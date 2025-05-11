from flask import Blueprint, render_template, request, jsonify
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import os

# 如果你已有 modules.visualization.ds_to_geojson, 請依你的模組路徑 import
from modules.visualization_rain import ds_to_geojson
from modules.downscaling_rain import get_rain_output

rain_bp = Blueprint('rain', __name__, template_folder='templates')

@rain_bp.route('/')
def rain_page():
    """ 顯示降水降尺度的前端頁面 (rain.html) """
    return render_template('rain.html')

@rain_bp.route('/coarseRain')
def api_coarse_rain():
    """
    讀取 TReADl_re 三個資料集 (train/val/test) 中對應時間的降水 (tp)。
    依 year 決定使用哪一個檔案：
      - year <= 2021 -> TReAD_train_16x16.nc
      - year == 2022 -> TReAD_val_16x16.nc
      - year >= 2023 -> TReAD_test_16x16.nc
    路由: /rain/coarseRain?year=YYYY&month=MM&day=DD&hour=HH
    """
    # 1. 參數檢查
    year_str  = request.args.get('year')
    month_str = request.args.get('month')
    day_str   = request.args.get('day')
    hour_str  = request.args.get('hour')
    if not all([year_str, month_str, day_str, hour_str]):
        return jsonify({"error": "缺少必要參數 (year/month/day/hour)"}), 400

    try:
        Y = int(year_str)
        M = int(month_str)
        D = int(day_str)
        H = int(hour_str)
        # 轉成 netCDF 裡的 ISO 時間字串
        time_str = f"{Y:04d}-{M:02d}-{D:02d}T{H:02d}:00:00"
    except ValueError:
        return jsonify({"error": "time format error"}), 400

    # 2. 選擇對應的檔案
    base_dir = '/home/nycustd/system/flask_app/data/raw/1_Meteorological_Data/TReADl_re'
    if Y <= 2021:
        fname = 'TReAD_train_16x16.nc'
    elif Y == 2022:
        fname = 'TReAD_val_16x16.nc'
    else:
        fname = 'TReAD_test_16x16.nc'
    nc_path = os.path.join(base_dir, fname)
    if not os.path.exists(nc_path):
        return jsonify({"error": f"File not found: {nc_path}"}), 400

    # 3. 開檔、選時間
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    if 'pr' not in ds:
        return jsonify({"error": "Dataset 中找不到變數 'pr'"}), 500

    try:
        ds_sel = ds.sel(time=time_str, method='nearest')
    except Exception as e:
        return jsonify({"error": f"選定的時間點在資料集中沒有對應: {time_str}"}), 500

    # 4. 填 NaN 並（若需要）調整單位
    ds_sel = ds_sel[['pr']].fillna(0)
    # 如果要把單位從 m 轉為 mm/hr, 可視資料單位決定
    # ds_sel['tp'] = ds_sel['tp'] * 1000

    # 5. 轉 GeoJSON
    geojson_dict = ds_to_geojson(ds_sel, var_name='pr')
    return jsonify(geojson_dict)

# 你若需要降尺度 (downscaleRain) 的功能, 也可保留:
# from inference import inference_pipeline_rain
@rain_bp.route('/downscaleRain')
def api_downscale_rain():
    # 1. 參數
    Y = int(request.args['year'])
    M = int(request.args['month'])
    D = int(request.args['day'])
    H = int(request.args['hour'])

    # 2. 產生 downscaled 陣列
    lat, lon, arr = get_rain_output(Y, M, D, H)

    # 3. 包成 xarray.DataArray 然後再轉 GeoJSON
    ds = xr.Dataset(
        {
            'pr': (('lat','lon'), arr)
        },
        coords={
            'lat': lat,
            'lon': lon
        }
    )
    # 4. 轉 geojson
    geojson = ds_to_geojson(ds, var_name='pr')
    return jsonify(geojson)

@rain_bp.route('/fineRain')
def api_fine_rain():
    """
    讀取 TReAD（三個檔案 train/val/test）中對應時間的細尺度降水 (pr)，
    路由: /rain/fineRain?year=YYYY&month=MM&day=DD&hour=HH
    """
    # 1. 取參數
    year_str  = request.args.get('year')
    month_str = request.args.get('month')
    day_str   = request.args.get('day')
    hour_str  = request.args.get('hour')
    if not all([year_str, month_str, day_str, hour_str]):
        return jsonify({"error": "缺少必要參數 (year/month/day/hour)"}), 400

    # 2. 轉 datetime string
    try:
        Y = int(year_str)
        M = int(month_str)
        D = int(day_str)
        H = int(hour_str)
        time_str = f"{Y:04d}-{M:02d}-{D:02d}T{H:02d}:00:00"
    except ValueError:
        return jsonify({"error": "time format error"}), 400

    # 3. 決定要讀哪一個檔案
    base_dir = '/home/nycustd/system/flask_app/data/raw/1_Meteorological_Data/TReAD'
    if Y <= 2021:
        fname = 'TReAD_train.nc'
    elif Y == 2022:
        fname = 'TReAD_val.nc'
    else:
        fname = 'TReAD_test.nc'

    nc_path = os.path.join(base_dir, fname)
    if not os.path.exists(nc_path):
        return jsonify({"error": f"File not found: {nc_path}"}), 400

    # 4. open & sel
    ds = xr.open_dataset(nc_path, engine='h5netcdf')
    if 'pr' not in ds:
        return jsonify({"error": "Dataset 中找不到變數 'pr'"}), 500

    try:
        ds_sel = ds.sel(time=time_str, method='nearest')
    except Exception:
        return jsonify({"error": f"指定時間在檔案中沒有對應: {time_str}"}), 400

    # 5. 填 NaN
    ds_sel = ds_sel[['pr']].fillna(0)
    # （如果單位需要轉換，可在此做）
    # 例如 ds_sel['pr'] *= 1000

    # 6. 轉 GeoJSON
    geojson_dict = ds_to_geojson(ds_sel, var_name='pr')
    return jsonify(geojson_dict)