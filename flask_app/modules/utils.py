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

from Baseline import archs
from Baseline import data
from Baseline import losses
from Baseline import models

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from torch.utils.data import DataLoader, Dataset

# 假設以下函數由 basicsr 或您專案中提供，請根據實際情況調整 import 路徑
from basicsr.utils.registry import ARCH_REGISTRY, DATASET_REGISTRY
from basicsr.archs import build_network

model = ARCH_REGISTRY.get('ClimateUformerMultiScaleHGTMultiScaleOut_TP2')

from datetime import datetime
def simplified_time_str(time_input):
    # 檢查輸入類型
    if isinstance(time_input, str):
        # 處理 ISO 8601 時間字串
        dt = datetime.strptime(time_input, '%Y-%m-%dT%H:%M:%S.%f000')
    elif isinstance(time_input, np.datetime64):
        # 處理 numpy.datetime64
        dt = datetime.utcfromtimestamp(time_input.astype('datetime64[s]').astype(int))
    else:
        raise ValueError("Unsupported time format. Input must be a string or numpy.datetime64.")
    
    # 返回格式化時間
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def load_yaml_config(experiment_name):
    """ 
        輔助函數：讀取指定實驗資料夾中的 YAML 設定檔
    """
    exp_folder = os.path.join("experiments", experiment_name)
    # 找出資料夾內的 YAML 檔（假設只有一個）
    yaml_files = [f for f in os.listdir(exp_folder) if f.endswith(".yml") or f.endswith(".yaml")]
    if not yaml_files:
        raise FileNotFoundError(f"在 {exp_folder} 找不到 YAML 檔案！")
    yaml_path = os.path.join(exp_folder, yaml_files[0])
    with open(yaml_path, 'r') as f:
        opt = yaml.safe_load(f)
    return opt

def load_model(experiment_name, iteration, opt):
    """
    根據實驗名稱、指定的 iteration 和配置 opt 建構模型並載入權重。
    
    若指定的 iteration 檔案不存在，則嘗試fallback到最新可用的檔案。
    """
    # 建立網絡結構
    model_config = opt['network_g']
    model = build_network(model_config)

    # 定義權重及訓練狀態檔存放位置
    model_dir = os.path.join("experiments", experiment_name, "models")
    state_dir = os.path.join("experiments", experiment_name, "trainging_states")  # 請確認資料夾名稱是否正確

    # 嘗試構造檔案路徑
    model_file = os.path.join(model_dir, f"net_g_{iteration}.pth")
    state_file = os.path.join(state_dir, f"{iteration}.state")

    load_path = None
    if os.path.exists(model_file):
        load_path = model_file
    elif os.path.exists(state_file):
        load_path = state_file
    else:
        # 指定 iteration 檔案不存在，fallback 機制：
        print(f"警告：既沒有 {model_file} 也沒有 {state_file}！嘗試從 {state_dir} 與 {model_dir} 搜尋最新的權重檔...")
        # 優先從 state 資料夾尋找最新可用檔案
        available_state_files = glob.glob(os.path.join(state_dir, "*.state"))
        available_model_files = glob.glob(os.path.join(model_dir, "net_g_*.pth"))
        
        candidate_files = available_state_files + available_model_files
        if not candidate_files:
            raise FileNotFoundError(f"找不到實驗 {experiment_name} 的任何權重檔！")
        
        # 從檔名中提取 iteration 數字，假設檔名格式為 "{iteration}.state" 或 "net_g_{iteration}.pth"
        def extract_iter(fp):
            base = os.path.basename(fp)
            if base.startswith('net_g_'):
                name = base[len('net_g_'):].split('.')[0]
            else:
                name, _ = os.path.splitext(base)
            try:
                return int(name)
            except:
                return -1
        
        candidate_files = sorted(candidate_files, key=extract_iter, reverse=True)
        load_path = candidate_files[0]
        print(f"fallback 到 {load_path}")

    # 載入檔案
    checkpoint = torch.load(load_path, map_location="cpu")
    
    # 若 checkpoint 為包含 state 的訓練狀態檔，預期結構為 { 'params': {...}, ... }
    param_key = 'params'
    if isinstance(checkpoint, dict):
        if param_key in checkpoint:
            checkpoint = checkpoint[param_key]
        else:
            # 如果 checkpoint 外層已是 state_dict，則不做額外處理
            pass
    else:
        raise RuntimeError("載入的 checkpoint 結構不正確，預期應為 dict！")
    
    # 移除 DistributedDataParallel 包裝所留下的 'module.' 前綴
    for k, v in deepcopy(checkpoint).items():
        if k.startswith('module.'):
            checkpoint[k[7:]] = v
            del checkpoint[k]

    # 載入 state dict 到模型中
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
    if missing_keys or unexpected_keys:
        print(f"載入模型時缺少的鍵：{missing_keys}")
        print(f"載入模型時多餘的鍵：{unexpected_keys}")
        raise RuntimeError("模型權重與定義不匹配！")
    
    model.eval()
    return model

def load_model_pth(experiment_name, iteration, opt):
    """
    依據實驗名稱、指定的 iteration 和配置 opt 建構模型並從 .pth 檔案中載入權重，
    此版本只依賴 .pth 檔案，適用於僅進行 inference 的情形。
    
    檔案預期放在:
      experiments/[experiment_name]/models/net_g_{iteration}.pth
    """
    # 建構模型
    model = build_network(opt['network_g'])
    
    # 定義 .pth 檔案位置
    model_dir = os.path.join("experiments", experiment_name, "models")
    pth_file = os.path.join(model_dir, f"net_g_{iteration}.pth")
    if not os.path.exists(pth_file):
        raise FileNotFoundError(f"{pth_file} 不存在！")
    
    # 載入檔案
    checkpoint = torch.load(pth_file, map_location="cpu")
    
    # 如果 checkpoint 為字典且包含 "params"，則提取其內容；否則直接當作 state dict 使用
    if isinstance(checkpoint, dict) and 'params' in checkpoint:
        checkpoint = checkpoint['params']
    
    # 移除由 DistributedDataParallel 導致的 'module.' 前綴
    for k, v in deepcopy(checkpoint).items():
        if k.startswith('module.'):
            checkpoint[k[7:]] = v
            del checkpoint[k]
    
    # 載入狀態字典到模型中
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
    if missing_keys or unexpected_keys:
        print(f"缺少的鍵：{missing_keys}")
        print(f"多餘的鍵：{unexpected_keys}")
        raise RuntimeError("模型權重與模型定義不匹配！")
    
    model.eval()
    return model

def load_mean_std_config():
    """
    從 dataset_opt["topographical_data_root"] 讀取 mean_std_config.yml，
    整理成一個 SimpleNamespace 物件，方便以屬性方式存取。
    """
    root = "/home/hujiahua/Weather/flask_app/data/raw/3_Topographical_Data"
    config_path = os.path.join(root, "mean_std_config.yml")
    # print(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到 mean_std_config.yml：{config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # 整理成物件形式
    mean_std = SimpleNamespace(
        cwa_mean = config["cwa"]["mean"],
        cwa_std  = config["cwa"]["std"],
        cwa_min  = config["cwa"]["min"],
        era5_mean = config["era5"]["mean"],
        era5_std  = config["era5"]["std"],
        era5_min  = config["era5"]["min"]
    )
    return mean_std

def inverse_transform(data, mean_std_config, param, method="zscore", domain='cwa'):
    """
    將已標準化的資料逆轉換回原始尺度。
    Args:
        data (torch.Tensor): 單一通道的資料 (H, W)
        param (str): 對應的參數名稱
        domain (str): 使用哪個領域的標準化參數，'cwa' 或 'era5'
    Returns:
        torch.Tensor: 逆轉換後的資料
    """
    if method == 'zscore':
        if domain == 'cwa':
            mean = mean_std_config.cwa_mean[param]
            std = mean_std_config.cwa_std[param]
        elif domain == 'era5':
            mean = mean_std_config.era5_mean[param]
            std = mean_std_config.era5_std[param]
        else:
            raise ValueError("Unknown domain for inverse transformation")
        return data * std + mean
    elif method == 'minshift':
        if domain == 'cwa':
            min = mean_std_config.cwa_min[param]
        elif domain == 'era5':
            min = mean_std_config.era5_min[param]
        else:
            raise ValueError("Unknown domain for inverse transformation")
        return data + min

# class InferenceDataset(Dataset):
#     def __init__(self, base_dataset, selected_times):
#         """
#         base_dataset: 例如 ERA5_CWA_200_interpolation_ts_tp2_obs2_Dataset 的實例，裡面已包含所有時間點資料
#         selected_times: list，包含要 inference 的時間點（型態需與 base_dataset中每筆資料['gt_time']相符，例如 np.datetime64 或字串）
#         """
#         self.base_dataset = base_dataset
#         # 如果 selected_times 是字串，則轉換成 np.datetime64
#         self.selected_times = [np.datetime64(t) if not isinstance(t, np.datetime64) else t for t in selected_times]

#         # 利用每筆資料的 'gt_time' 來篩選資料索引
#         self.indices = []
#         for i in range(len(self.base_dataset)):
#             item = self.base_dataset[i]
#             # 假設每筆資料中 'gt_time' 能轉換成 np.datetime64
#             gt_time = np.datetime64(item['gt_time'])
#             if gt_time in self.selected_times:
#                 self.indices.append(i)
#         if not self.indices:
#             # print("Dataset 中的 gt_time:")
#             # for i in range(len(self.base_dataset)):
#             #     print(np.datetime64(self.base_dataset[i]['gt_time']))
#             raise ValueError("選定的時間點在資料集中沒有對應！")
    
#     def __len__(self):
#         return len(self.indices)
    
#     def __getitem__(self, idx):
#         orig_idx = self.indices[idx]
#         return self.base_dataset[orig_idx]


# dataset_opt = {
#     'name': 'ERA5_CWA_val',
#     'type': 'ERA5_CWA_200_interpolation_ts_tp2_obs2_Dataset',
#     'meteorological_data_root': '../../0_Data/1_Meteorological_Data',
#     'observational_data_root': '../../0_Data/2_Observational_Data',
#     'topographical_data_root': '../../0_Data/3_Topographical_Data',
#     'start_year': 2023,
#     'start_month': 1,
#     'end_year': 2023,
#     'end_month': 2,

#     'source_version': 'ERA5_corrdiff_v1',
#     'target_version': 'CWA_Taiwan_v2.4',
#     'topo_version': 'CWA_Taiwan_v3.2',
#     'topo_filename': 'CWA_Taiwan_v3.1.2_terrain.nc',
#     'obs_version': 'obs_v5',

#     'input_params': ['t2m', 'ws10', 'u10', 'v10', 'q', 'd2m'],
#     'output_params': ['u10', 'v10'],
#     'obs_params': ["TX", "PS", "PP", "RH", "WD", "WS"],
#     'topo_params': ['terrain'],
#     'time_steps': 8,
#     'obs_time_steps': 16,
#     'apply_zscore': True,

#     'io_backend':
#       {'type': 'disk'},
#     'scale': 10
# } 

def ensure_dir(directory):
    """
    確保目錄存在，若不存在則創建
    :param directory: 目錄路徑
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def list_files(directory, extension=None):
    """
    列出指定目錄下的所有文件
    :param directory: 目錄路徑
    :param extension: 指定文件擴展名 (可選)
    :return: list, 文件列表
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))]
    if extension:
        files = [f for f in files if f.endswith(extension)]
    return files
