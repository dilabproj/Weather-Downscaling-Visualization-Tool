# app.py
from flask import Flask, render_template, request, jsonify
import xarray as xr
import torch
import numpy as np
import os, sys, threading
import traceback

app = Flask(__name__)

from modules.downscaling_wind import single_time_downscale
from modules.visualization import ds_to_geojson
from basicsr.utils.options import parse_options
# from modules.downscaling_wind import inference_pipeline
from modules.downscaling_wind import load_dataloader_from_yml, load_model_from_yml

from rain_route import rain_bp
app.register_blueprint(rain_bp, url_prefix='/rain')

# 你的实验配置
EXP_FOLDER   = "/home/nycustd/system/flask_app/modules/Baseline/experiments/250408_UformerLSTM3Obs2TP3_CWA_org_ntp_16ts_obs16ts_96b_zscore_2e-4lr_0.7dr_1e-5wd_1gpu_train01"
ITERATION    = "latest"
MODEL_MODULE = "modules.Baseline.archs.climate_uformer_all_arch"
DATA_MODULE  = "modules.Baseline.data.ERA5_CWA_200_interpolation_ts_tp2_obs2_dataset"
# 運算裝置（GPU 或 CPU）
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 讀取 yml，並用 parse_options 建立 opt
yml_file = next(f for f in os.listdir(EXP_FOLDER) if f.endswith((".yml",".yaml")))
sys.argv = ["inference.py", "-opt", os.path.join(EXP_FOLDER, yml_file)]
opt, _ = parse_options(EXP_FOLDER, is_train=False)
# 2) Patch 模型權重路徑與裝置
opt["path"]["pretrain_network_g"] = os.path.join(
    EXP_FOLDER, "models", f"net_g_{ITERATION}.pth"
)
opt["device"] = DEVICE

# 3) 全域初始化 Dataset（只做索引，不做一次性讀取所有資料）
_dataset_cls = getattr(
    __import__(DATA_MODULE, fromlist=[""]), 
    opt["datasets"]["val"]["type"]
)
val_ds = _dataset_cls(opt["datasets"]["val"])

# 4) 全域載入模型（只做一次）
model = load_model_from_yml(opt, DEVICE, MODEL_MODULE)
model.eval()


@app.route('/')
def index():
    return render_template('index.html')

from datetime import datetime, timedelta

@app.route('/coarse')
def api_coarse():
    var_name = request.args.get('var')
    year_str  = request.args.get('year')
    month_str = request.args.get('month')
    day_str   = request.args.get('day')
    hour_str  = request.args.get('hour')

    # 檢查參數
    if not all([var_name, year_str, month_str, day_str, hour_str]):
        return jsonify({"error": "缺少必要參數 (var/year/month/day/hour)"}), 400

    # 先組出原始 datetime，再往前推 8 小時
    try:
        Y = int(year_str)
        M = int(month_str)
        D = int(day_str)
        H = int(hour_str)
        dt0 = datetime(Y, M, D, H)
    except ValueError:
        return jsonify({"error": "time format error"}), 400

    dt = dt0 - timedelta(hours=7)
    sel_year  = dt.year
    sel_month = dt.month
    sel_day   = dt.day
    sel_hour  = dt.hour
    # 用新的年作為檔案夾/檔名的路徑
    nc_path = (
        f"data/raw/1_Meteorological_Data/ERA5_corrdiff_v1/"
        f"{sel_year}/{sel_year}_ERA5_corrdiff_v1_{var_name}.nc"
    )
    if not os.path.exists(nc_path):
        return jsonify({"error": f"檔案不存在: {nc_path}"}), 400

    # 打開 NetCDF，選用 h5netcdf engine
    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    # 依據調整後的時間選取
    time_str = dt.strftime("%Y-%m-%dT%H:00:00")
    try:
        ds_sel = ds.sel(time=time_str)
    except KeyError:
        return jsonify({"error": f"找不到時間點 {time_str}"}), 400

    # 填補 NaN
    ds_sel = ds_sel.fillna(0)

    # 轉成 GeoJSON
    from modules.visualization import ds_to_geojson
    geojson_dict = ds_to_geojson(ds_sel, var_name=var_name)
    return jsonify(geojson_dict)


@app.route('/downscale')
def api_downscale():
    """
    單一時間點降尺度 API
    路徑範例: /rain/downscaleRain?var=u10&year=2023&month=7&day=28&hour=0
    """
    # 1) 參數檢查
    var = request.args.get("var", "u10")
    try:
        y, m, d, h = map(int, (
            request.args["year"],
            request.args["month"],
            request.args["day"],
            request.args["hour"]
        ))
    except Exception:
        return jsonify({"error": "缺少或格式錯誤的年/月/日/時參數"}), 400

    # 2) 組出 ISO 時間字串
    time_str = f"{y:04d}-{m:02d}-{d:02d}T{h:02d}:00:00"
    target64 = np.datetime64(time_str)

    # 3) 在 val_ds 中線性搜尋索引
    try:
        idx = next(
            i for i in range(len(val_ds))
            if np.datetime64(val_ds[i]["gt_time"]) == target64
        )
    except StopIteration:
        return jsonify({"error": f"找不到時間 {time_str}"}), 404

    # 4) 從 Dataset 取出該筆資料，並只將 tensor 欄位組成模型輸入
    item = val_ds[idx]
    inp = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            inp[k] = v.unsqueeze(0).to(DEVICE)

    # 5) 模型前向推理
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, tuple):
            out = out[0]
        arr = out.squeeze(0).cpu().numpy()

    # 6) 根據 var 選擇通道
    if var.lower().startswith("u"):
        ch = 0
    elif var.lower().startswith("v"):
        ch = 1
    else:
        ch = 0
    arr2d = arr[ch] if arr.ndim == 3 else arr[ch, -1]

    # 7) 經緯度直接從 Dataset 拿
    lats = val_ds.tp_lat
    lons = val_ds.tp_lon

    # 8) 打包成 xarray 再轉 GeoJSON
    da = xr.DataArray(arr2d, dims=("latitude", "longitude"),
                      coords={"latitude": lats, "longitude": lons})
    ds = da.to_dataset(name=var)
    geojson = ds_to_geojson(ds, var)
    return jsonify(geojson)


# @app.route('/downscale')
# def api_downscale():
#     var_name = request.args.get('var')
#     year_str = request.args.get('year')
#     month_str = request.args.get('month')
#     day_str   = request.args.get('day')
#     hour_str  = request.args.get('hour')

#     nc_path = f"data/raw/1_Meteorological_Data/ERA5_corrdiff_v1/{year_str}/{year_str}_ERA5_corrdiff_v1_{var_name}.nc"
#     if not os.path.exists(nc_path):
#         return jsonify({"error": f"File not found: {nc_path}"}), 400

#     ds = xr.open_dataset(nc_path)

#     try:
#         Y = int(year_str)
#         M = int(month_str)
#         D = int(day_str)
#         H = int(hour_str)
#         time_str = f"{Y:04d}-{M:02d}-{D:02d}T{H:02d}:00:00"
#     except:
#         return jsonify({"error": "time format error"}), 400

#     ds_sel = ds.sel(time=time_str, method='nearest')

#     # 真正做插值: [21.635..25.635], [118.885..122.885] 各延伸0.125 => 21.51~25.76, 118.76~123.01
#     ds_down = inference_pipeline(ds_sel, var_name=var_name)

#     ds_down = ds_down.fillna(0)

#     geojson_dict = ds_to_geojson(ds_down, var_name=var_name)
#     return jsonify(geojson_dict)

from datetime import datetime, timedelta

@app.route('/fine')
def api_fine():
    """
    顯示真正的細尺度資料 (CWA)
    e.g. data/processed/CWA/2018/2018_CWA_Taiwan_v2.4_u10.nc
    """
    var_name = request.args.get('var')     # e.g. 'u10','v10','t2m'...
    year_str  = request.args.get('year')   # e.g. '2018'
    month_str = request.args.get('month')  # e.g. '3'
    day_str   = request.args.get('day')    # e.g. '15'
    hour_str  = request.args.get('hour')   # e.g. '0' ~ '23'

    # 檢查參數完整性
    if not all([var_name, year_str, month_str, day_str, hour_str]):
        return jsonify({"error": "缺少必要參數 (var/year/month/day/hour)"}), 400

    # 先組出 datetime，再往前推 8 小時
    try:
        Y  = int(year_str)
        M  = int(month_str)
        D  = int(day_str)
        H  = int(hour_str)
        dt0 = datetime(Y, M, D, H)
    except ValueError:
        return jsonify({"error": "time format error"}), 400

    dt = dt0 - timedelta(hours=7)
    sel_year  = dt.year
    sel_month = dt.month
    sel_day   = dt.day
    sel_hour  = dt.hour

    # 組合檔案路徑（仍用原始細尺度的年份資料夾）
    nc_path = (
        f"data/processed/CWA/{sel_year}/"
        f"{sel_year}_CWA_Taiwan_v2.4_{var_name}.nc"
    )
    if not os.path.exists(nc_path):
        return jsonify({"error": f"File not found: {nc_path}"}), 400

    # 開啟 NetCDF
    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    # 使用調整後的時間字串去選 time
    time_str = dt.strftime("%Y-%m-%dT%H:00:00")
    try:
        ds_sel = ds.sel(time=time_str)
    except KeyError:
        return jsonify({"error": f"找不到時間點 {time_str}"}), 400

    # 填補 NaN
    ds_sel = ds_sel.fillna(0)

    # 轉成 GeoJSON
    from modules.visualization import ds_to_geojson
    geojson_dict = ds_to_geojson(ds_sel, var_name=var_name)
    return jsonify(geojson_dict)


@app.route('/rain')
def rain_page():
    return render_template('rain.html')


if __name__ == '__main__':
    app.run(debug=True)