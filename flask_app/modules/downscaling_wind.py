import os
import glob
import yaml
import torch
import os.path as osp
import logging
import numpy as np
from copy import deepcopy
from types import SimpleNamespace
from matplotlib.colors import TwoSlopeNorm
import xarray as xr
import importlib
import os, sys, torch
import time
import pandas as pd
from datetime import datetime, timedelta

from modules.Baseline import archs
from modules.Baseline import data
from modules.Baseline import losses
from modules.Baseline import models

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from torch.utils.data import DataLoader, Dataset


from tqdm import tqdm

from basicsr.utils.options import parse_options
# 假設以下函數由 basicsr 或您專案中提供，請根據實際情況調整 import 路徑
from basicsr.utils.registry import ARCH_REGISTRY, DATASET_REGISTRY
from basicsr.archs import build_network

def crop_center(data, target_size):
    """Crop the center region of the data to the target size."""
    # print(data)
    # print(data.shape)
    _, h, w = data.shape
    crop_h, crop_w = target_size
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return data[:, start_h:start_h + crop_h, start_w:start_w + crop_w]

def load_model_from_yml(opt, device, module_name):
    import importlib

    # 動態導入模型類型
    model_class = opt['network_g']['type']  # 從 YAML 獲取模型類型
    # model_module = "my_models"  # 替換為實際模型的模組名稱
    model_module = importlib.import_module(module_name)
    ModelClass = getattr(model_module, model_class)

    # 加載模型參數
    model_params = {k: v for k, v in opt['network_g'].items() if k != 'type'}  # 過濾掉 type
    model = ModelClass(**model_params)  # 自動解包參數

    # 加載模型權重
    model_path = opt['path']['pretrain_network_g']
    model.load_state_dict(torch.load(model_path, map_location=device)['params'], strict=False)
    model.eval()
    model = model.to(device)

    return model

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import importlib

def load_dataloader_from_yml(opt, module_name):
    # 動態導入數據集類型
    dataset_class = opt['datasets']['val']['type']  # 從 YAML 獲取數據集類型
    dataset_module = importlib.import_module(module_name)
    DatasetClass = getattr(dataset_module, dataset_class)

    # 加載數據集參數
    dataset = DatasetClass(opt['datasets']['val'])

    # 創建原始 dataloader
    raw_loader = DataLoader(
        dataset,
        batch_size       = 1,
        shuffle          = False,
        num_workers      = 0,     # 不启动任何子进程
        pin_memory       = False, # 不把数据刷到页锁定内存
        prefetch_factor  = 1,     # 每个 worker 只预读取 1 个 batch
        persistent_workers=False, # 工作进程用完就销毁
        drop_last        = False,
    )
    return loader, dataset

    # 包裝進 tqdm
    loader = tqdm(
        raw_loader,
        desc="Loading val data",
        unit="batch",
        leave=False    # 遍歷完後自動清除進度條
    )

    return loader, dataset

class InferenceSubset(Dataset):
    """把 base_ds 裡 gt_time 完全吻合 selected_times 的那些 index 挑出來"""
    def __init__(self, base_ds, selected_times):
        # 1) 把 base_ds 原先的所有時間都抓出來 (np.datetime64)
        all_times = [ np.datetime64(item['gt_time']) 
                      for item in base_ds ]
        # 2) 把 selected_times 全部轉成 np.datetime64
        want = [ np.datetime64(t) for t in selected_times ]
        # 3) 對每個 want 找 exact match 的 index
        idxs = []
        for t in want:
            matches = [i for i,tt in enumerate(all_times) if tt == t]
            if not matches:
                raise ValueError(f"時間 {t} 不在資料集中！")
            idxs.append(matches[0])
        self.base = base_ds
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.base[self.idxs[i]]


def single_time_downscale(
    exp_folder: str,
    iteration:  str,
    time_str:   str,   # "YYYY-MM-DDTHH:MM:SS"
    var_name:   str,   # 'u10','v10','tp',...
    n_steps:    int = 9, # 前面 8 小時 + 當下
    device:     str = "cpu"
):
    import os, sys, time
    import numpy as np
    import torch
    import pandas as pd
    from datetime import datetime, timedelta
    from basicsr.utils.options import parse_options

    t0 = time.time()
    print(f"[01] start                           {t0:.3f}s")

    # ----------------------------------------------------------------
    # 1) 先讀 yml、patch opt
    # ----------------------------------------------------------------
    yml = next(f for f in os.listdir(exp_folder) if f.endswith(('.yml','.yaml')))
    sys.argv = ['inference.py','-opt',os.path.join(exp_folder,yml)]
    opt, _ = parse_options(exp_folder, is_train=False)
    opt['path']['pretrain_network_g'] = os.path.join(
        exp_folder, 'models', f"net_g_{iteration}.pth"
    )
    opt['device'] = device
    t1 = time.time()
    print(f"[02] load & patch opt               {t1-t0:.3f}s")

    # ----------------------------------------------------------------
    # 2) 動態把 val 的時間範圍縮小到「當月」＋（若跨月）「前一月」
    # ----------------------------------------------------------------
    ts = datetime.fromisoformat(time_str)
    prev = ts - timedelta(hours=n_steps-1)
    months = sorted({ (prev.year, prev.month), (ts.year, ts.month) })
    opt['datasets']['val']['start_year']  = months[0][0]
    opt['datasets']['val']['start_month'] = months[0][1]
    opt['datasets']['val']['end_year']    = months[-1][0]
    opt['datasets']['val']['end_month']   = months[-1][1]
    t2 = time.time()
    print(f"[03] adjust val date range          {t2-t1:.3f}s")

    # ----------------------------------------------------------------
    # 3) 再載 DataLoader & Model
    # ----------------------------------------------------------------
    model_module   = "modules.Baseline.archs.climate_uformer_all_arch"
    dataset_module = "modules.Baseline.data.ERA5_CWA_200_interpolation_ts_tp2_obs2_dataset"
    val_loader, val_ds   = load_dataloader_from_yml(opt, dataset_module)
    t3 = time.time()
    print(f"[04] load DataLoader                {t3-t2:.3f}s")
    model                = load_model_from_yml(opt, device, model_module)
    t4 = time.time()
    print(f"[05] load Model                     {t4-t3:.3f}s")

    # ----------------------------------------------------------------
    # 4) 找到這個 time_str 在 val_ds 中的 index
    # ----------------------------------------------------------------
    target = np.datetime64(time_str)
    idx = next(
        i for i in range(len(val_ds))
        if np.datetime64(val_ds[i]['gt_time']) == target
    )
    t5 = time.time()
    print(f"[06] find index in dataset         {t5-t4:.3f}s")

    # ----------------------------------------------------------------
    # 5) 取出那一筆資料，連同 tp、trnd、trns、adj…都進 inp
    # ----------------------------------------------------------------
    item = val_ds[idx]
    inp = {}
    for k,v in item.items():
        if isinstance(v, torch.Tensor):
            inp[k] = v.unsqueeze(0).to(device)
    t6 = time.time()
    print(f"[07] build input dict               {t6-t5:.3f}s")

    # ----------------------------------------------------------------
    # 6) forward
    # ----------------------------------------------------------------
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, tuple): out = out[0]
        arr = out.squeeze(0).cpu().numpy()
    t7 = time.time()
    print(f"[08] model forward                  {t7-t6:.3f}s")

    # ----------------------------------------------------------------
    # 7) pick channel
    # ----------------------------------------------------------------
    ch = 0 if var_name.startswith('u') else 1 if var_name.startswith('v') else 0
    if arr.ndim == 3:
        arr2d = arr[ch]
    else:
        arr2d = arr[ch, -1]
    t8 = time.time()
    print(f"[09] slice channel                  {t8-t7:.3f}s")

    # ----------------------------------------------------------------
    # 8) 經緯度從 dataset 拿
    # ----------------------------------------------------------------
    lats = val_ds.tp_lat
    lons = val_ds.tp_lon
    t9 = time.time()
    print(f"[10] grab coords                    {t9-t8:.3f}s")
    print(f"[11] TOTAL                          {t9-t0:.3f}s")

    return lats, lons, arr2d

# def find_config_file(exp_folder):
#     for fn in os.listdir(exp_folder):
#         if fn.endswith((".yml", ".yaml")):
#             return os.path.join(exp_folder, fn)
#     raise FileNotFoundError(f"在 {exp_folder} 找不到 .yml/.yaml 配置文件！")

# def read_obs_csv(path, device):
#     # 1) 只讀時間和數值兩欄
#     #    假設第 1 欄是 station_id、第 2 欄是 time、第 3 欄是真正的值
#     df = pd.read_csv(
#         path,
#         usecols=[1, 2],       # 只讀 time, value
#         header=0,
#         names=['time', 'val'],# 重命名
#         parse_dates=['time'], # 把它 parse 成 datetime
#         index_col='time'      # 當成 index
#     )
#     arr = df['val'].values.astype(np.float32)  # shape = (T,)
#     # 如果你的模型需要 (T,H,W) 或 (T,Nstations)，再 reshape
#     t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
#     return t

# def single_time_downscale(
#     exp_folder:  str,
#     iteration:   str,
#     time_str:    str,    # "YYYY-MM-DDTHH:MM:SS"
#     var_name:    str,    # LQ 数据变量名，比如 'tp'
#     device:      str = "cpu",
#     n_steps:      int = 8  # 前 n_lead 小时 + 当前时刻
# ):
#     # 1) load yml & parse opt
#     cfg_path = find_config_file(exp_folder)
#     with open(cfg_path, 'r') as f:
#         cfg = yaml.safe_load(f)
#     import sys
#     sys.argv = ['inference.py', '-opt', cfg_path]
#     opt, _ = parse_options(exp_folder, is_train=False)
#     opt['path']['pretrain_network_g'] = os.path.join(
#         exp_folder, "models", f"net_g_{iteration}.pth"
#     )
#     opt['device'] = device

#     # 2) build model
#     module_path = opt['network_g'].get('module',
#         'modules.Baseline.archs.climate_uformer_all_arch'
#     )
#     cls_name = opt['network_g']['type']
#     ModelMod = importlib.import_module(module_path)
#     ModelClass = getattr(ModelMod, cls_name)
#     model = ModelClass(**{k:v for k,v in opt['network_g'].items() if k!="type"})
#     # load weights
#     ckpt = torch.load(opt['path']['pretrain_network_g'], map_location=device)
#     sd = ckpt.get('params', ckpt)
#     for k,v in deepcopy(sd).items():
#         if k.startswith("module."):
#             sd[k[7:]] = v; del sd[k]
#     model.load_state_dict(sd, strict=True)
#     model = model.to(device).eval()

#     # 2) 计算需要的时间区间 ↓
#     # time_str 形如 "2023-07-28T02:00:00"
#     ts   = datetime.fromisoformat(time_str)
#     # 如果你要前 8 小時，一共 9 個時刻，就用 n_steps - 1
#     prev = ts - timedelta(hours=n_steps-1)

#     # months 會是一個 set，存前一筆與現在筆的 (year, month)。
#     # 再把它轉成排序好的 list：
#     months = sorted({ (prev.year, prev.month),
#                     ( ts.year,  ts.month) })
#     # 範例結果：[(2023, 6), (2023, 7)]
#     lq_root = opt['datasets']['val']['meteorological_data_root']
#     src_ver = opt['datasets']['val']['source_version']
#     input_vars = opt['datasets']['val']['input_params']  # e.g. ['t2m','ws10','u10','v10','q','d2m']
#     topo_root = opt['datasets']['val']['topographical_data_root']
#     topo_ver  = opt['datasets']['val']['topo_version']
#     topo_list = opt['datasets']['val']['topo_params']
#     obs_root  = opt['datasets']['val']['observational_data_root']
#     obs_ver   = opt['datasets']['val']['obs_version']
    

#      # 3) 载入所有 meteorological variables 序列，拼成 (T, n_vars, H, W)
#     seqs = []
#     for var in input_vars:
#         arrs = []
#         for (yr, mo) in months:
#             fn = os.path.join(
#                 lq_root, "ERA5_corrdiff_v1", str(yr),
#                 f"{yr}_{src_ver}_{var}.nc"
#             )
#             ds = xr.open_dataset(fn, engine="h5netcdf")
#             # slice 从 prev 到 ts
#             ds2 = ds.sel(
#                 time=slice(prev.strftime("%Y-%m-%dT%H:00:00"),
#                            ts .strftime("%Y-%m-%dT%H:00:00"))
#             ).fillna(0)
#             arrs.append(ds2[var].values.astype(np.float32))  # (??, H, W)
#             ds.close()

#         merged = np.concatenate(arrs, axis=0)      # 可能 >> n_steps…
#         # **取最后 n_steps 帧**，确保 merged.shape[0] == n_steps
#         if merged.shape[0] < n_steps:
#             raise ValueError(f"数据不足 {n_steps} 帧")
#         merged = merged[-n_steps:]                # (n_steps, H, W)
#         seqs.append(merged)

#     # stack 成 (T, n_vars, H, W)
#     seqs = np.stack(seqs, axis=1)
#     T, n_vars, H, W = seqs.shape

#     # reshape→(T*n_vars, H, W)，再加 batch 维
#     lq_seq = seqs.reshape(T * n_vars, H, W)
#     t_lq   = torch.from_numpy(lq_seq).unsqueeze(0).to(device)  # (1,96,16,16)


#     # 5) load all topo_params
#     topo_root   = cfg['datasets']['val']['topographical_data_root']
#     topo_ver    = cfg['datasets']['val']['topo_version']
#     topo_list   = cfg['datasets']['val']['topo_params']
#     inp = {'lq': t_lq}

#     for p in topo_list:
#         if p == 'terrain' or p == 'slope' or p == 'aspect':
#             # terrain 用 topo_filename
#             fp = os.path.join(topo_root, cfg['datasets']['val']['topo_filename'])
#         else:
#             # 其它 params: e.g. 2023_CWA_Taiwan_v3.2_delta_utan.nc
#             fp = os.path.join(
#                 topo_root, topo_ver,f"{ts.year}",
#                 f"{ts.year}_{topo_ver}_{p}.nc"
#             )
#         ds = xr.open_dataset(fp, engine="h5netcdf")
#         arr = ds[p].values.astype(np.float32)
#         ds.close()
#         # 每个 topo param 都变成一个 channel
#         inp[p] = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

#     # 6) 把 terrain 重命名到 tp，给模型拿
#     if 'terrain' in inp:
#         inp['tp'] = inp.pop('terrain')
#     else:
#         raise KeyError("必须在 topo_params 里包含 'terrain' 作为地形高度")

#     obs_root = cfg['datasets']['val']['observational_data_root']
#     obs_ver  = cfg['datasets']['val']['obs_version']

#     # station_adj -> adj
#     adj_csv = os.path.join(obs_root, obs_ver, f"station_adj_{obs_ver}.csv")
#     df_adj = pd.read_csv(adj_csv, index_col=0, parse_dates=True)
#     arr_adj = df_adj.values.astype(np.float32)
#     inp['adj'] = torch.from_numpy(arr_adj).unsqueeze(0).unsqueeze(0).to(device)

#     # station_dynamic -> trnd
#     trnd_csv = os.path.join(obs_root, obs_ver, f"station_dynamic_{obs_ver}.csv")
#     # 動態
#     inp['trnd'] = read_obs_csv(trnd_csv, device)

#     # station_static -> trns
#     trns_csv = os.path.join(obs_root, obs_ver, f"station_static_{obs_ver}.csv")
#     df_trns = pd.read_csv(trns_csv, index_col=0, parse_dates=True)
#     arr_trns = df_trns.values.astype(np.float32)
#     inp['trns'] = torch.from_numpy(arr_trns).unsqueeze(0).unsqueeze(0).to(device)

#     # 6) forward once
#     with torch.no_grad():
#         out = model(inp)
#         if isinstance(out, tuple):
#             out = out[0]
#         out = out.squeeze(0).cpu().numpy()  # (C, H, W)

#     # 7) pick var_name 对应的 channel
#     #    假设 network 输出第 0/1 通道分别是 u/v，否则可改这里
#     if var_name.startswith('u'): ch = 0
#     elif var_name.startswith('v'): ch = 1
#     else: ch = 0
#     result2d = out[ch]

#     # 8) read coords from any one of the LQ files
#     ref_fn = os.path.join(
#         lq_root, str(ts.year),
#         f"{ts.year}_{src_ver}_{var_name}.nc"
#     )
#     dsr = xr.open_dataset(ref_fn, engine="h5netcdf")
#     lats = dsr['latitude'].values
#     lons = dsr['longitude'].values
#     dsr.close()

#     return lats, lons, result2d


def inference_pipeline(root_path, model_path, output_dir, device, model_module, dataset_module, add_tp):
    # 初始化環境與日誌記錄器
    os.makedirs(output_dir, exist_ok=True)
    opt, args = parse_options(root_path, is_train=False)
    opt['root_path'] = root_path
    # root_path = osp.abspath(osp.join(__file__, osp.pardir))
    opt['path']['pretrain_network_g'] = model_path
    opt['device'] = device
    crop_size = (200, 200)

    # 設置 cuDNN 性能優化
    torch.backends.cudnn.benchmark = True

    # 加載數據集與數據載入器
    val_loader, dataset = load_dataloader_from_yml(opt, dataset_module)
    
    # 加載模型
    model = load_model_from_yml(opt, device, model_module)

    # 推理與可視化
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_loader, desc="Inference")):
            # data_timer.record()
            if add_tp:
                # 準備資料
                lq = data['lq'].to(device)
                tp = data['tp'].to(device)
                # 模型推理
                output, *_ = model({'lq': lq, 'tp': tp})
            else:
                # 準備資料
                lq = data['lq'].to(device)
                # 模型推理
                output, *_ = model({'lq': lq})

            if output.ndim == 3:  # shape 為 (2, 200, 200)
                output = output.unsqueeze(0)  # 新形狀為 (1, 2, 200, 200)
                
            gt = data['gt'].numpy()[0]
            output = output.cpu().numpy()[0]
            
            # print(f'gt.shape = {gt.shape}')
            # print(f'output.shape = {output.shape}')
            
            
            # if gt.ndim == 3:  # shape 為 (2, 200, 200)
            #     gt = gt.unsqueeze(0)  # 新形狀為 (1, 2, 200, 200)
                
            # 若最後兩個維度不是 (200, 200) 才進行裁切
            if output.shape[-2:] != crop_size:
                output = crop_center(output, crop_size)
            if gt.shape[-2:] != crop_size:
                gt = crop_center(gt, crop_size)
                
            # print(output.shape)

            # 讀取經緯度與時間
            lat = dataset.tp_lat
            lon = dataset.tp_lon
            time_str = data['lq_time'][0]
            
            # print(lat)
            # print(lon)
                
            # else:
            #     save_path = os.path.join(output_dir, f"{simplified_time_str(time_str)}_{param_name}.png")
            #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #     visualize_wind_results(output[0], gt[0], lat, lon, param_name, 
            #                         simplified_time_str(time_str),
            #                         save_path=os.path.join(output_dir, f"{simplified_time_str(time_str)}_{param_name}.png"))

#----------------------
# experiment_name = "250303_UformerLSTM_CWA_org_8ts_32b_zscore_1gpu_train01"  # 模型檔案路徑
# yml_file = "CWA_UformerLSTM_200_8ts32b1z.yml"
# iteration = "latest"
# # output_filename = experiment_name
# output_dir = experiment_name
# target_dataset = "CWA"
# model_module = "archs.climate_uformer_arch"
# dataset_module = "data.ERA5_CWA_200_interpolation_ts_dataset"
# add_tp = True
# vars = ["v10", "u10"]
# #----------------------

# # inference參數
# root_path = f"/home/hujiahuaii12/Weather/BaselineTest/Baseline/experiments/{experiment_name}"
# model_path = os.path.join(root_path, f"models/net_g_{iteration}.pth")
# yml_file_path = os.path.join(root_path, yml_file)
# inference_output_dir = f"./inference/{output_dir}/inference_result" os.path.join(root_path, "/in")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # tensorboard參數
# performance_output_dir = f"./inference/{output_dir}/performance"
# log_dir = f"tb_logger/{experiment_name}"

# import sys
# sys.argv = ['inference.py', '-opt', f'{yml_file_path}']

# ### 執行推理
# inference_pipeline(root_path, model_path, inference_output_dir, device, model_module, dataset_module, add_tp)
# interpolation_inference_pipeline(root_path, inference_output_dir, dataset_module, visualization)