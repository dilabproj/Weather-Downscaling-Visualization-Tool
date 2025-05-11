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

from modules.Baseline import archs
from modules.Baseline import data
from modules.Baseline import losses
from modules.Baseline import models

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

def visualize_combined_wind_fields(time_key, res_dict, gt_lat, gt_lon, save_path):
    """
    產生風場圖像：每列代表一個實驗，左欄為 Ground Truth，右欄為模型 Output。
    第一行為標題列（'Ground Truth' 與 'Output'），第二行開始才是實際資料。
    在主標題下方顯示全圖最大最小值。Colorbar 與主繪圖區等高，使用 quiver 表示風向。
    """
    experiments = list(res_dict.keys())
    n_exp = len(experiments)
    
    # 提取每個實驗的 ground truth 與 output
    gt_list = []
    out_list = []
    for exp in experiments:
        gt = res_dict[exp]["gt"]
        out = res_dict[exp]["output"]
        if gt.shape[0] >= 2:
            u_gt = gt[0].cpu().numpy()
            v_gt = gt[1].cpu().numpy()
        else:
            u_gt = gt[0].cpu().numpy()
            v_gt = np.zeros_like(u_gt)
        if out.shape[0] >= 2:
            u_out = out[0].cpu().numpy()
            v_out = out[1].cpu().numpy()
        else:
            u_out = out[0].cpu().numpy()
            v_out = np.zeros_like(u_out)
        speed_gt = np.sqrt(u_gt**2 + v_gt**2)
        speed_out = np.sqrt(u_out**2 + v_out**2)
        gt_list.append((speed_gt, u_gt, v_gt))
        out_list.append((speed_out, u_out, v_out))
    
    # 統一色階
    all_speed = np.concatenate([
        np.array([s for (s, _, _) in gt_list]).flatten(),
        np.array([s for (s, _, _) in out_list]).flatten()
    ])
    vmin_speed, vmax_speed = np.min(all_speed), np.max(all_speed)
    
    # 建立圖像，n_exp+1 行，其中第一行為 header
    fig, axes = plt.subplots(
        n_exp+1, 2, 
        figsize=(16, 3*(n_exp+1)), 
        squeeze=False, 
        subplot_kw={'projection': ccrs.PlateCarree()},
        gridspec_kw={'height_ratios': [0.3] + [1]*n_exp}
    )
    
    # Header row
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[0, 0].text(0.5, 0.5, "Ground Truth", fontsize=14, ha='center', va='center')
    axes[0, 1].text(0.5, 0.5, "Output", fontsize=14, ha='center', va='center')
    
    stride = 10  
    last_im = None
    for i in range(n_exp):
        s_gt, u_gt, v_gt = gt_list[i]
        s_out, u_out, v_out = out_list[i]
        
        # 左欄：GT
        ax_gt = axes[i+1, 0]
        im = ax_gt.pcolormesh(
            gt_lon, gt_lat, s_gt, 
            cmap='Blues', shading='nearest', 
            vmin=vmin_speed, vmax=vmax_speed,
            transform=ccrs.PlateCarree()
        )
        ax_gt.quiver(
            gt_lon[::stride], gt_lat[::stride], 
            u_gt[::stride, ::stride], v_gt[::stride, ::stride],
            transform=ccrs.PlateCarree(), scale=50, scale_units='inches',
            width=0.003, color='black', alpha=0.7
        )
        ax_gt.coastlines(resolution='10m', linewidth=0.8)
        ax_gt.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_gt.set_extent([gt_lon[0], gt_lon[-1], gt_lat[0], gt_lat[-1]], ccrs.PlateCarree())
        
        # 右欄：Output
        ax_out = axes[i+1, 1]
        im = ax_out.pcolormesh(
            gt_lon, gt_lat, s_out, 
            cmap='Blues', shading='nearest', 
            vmin=vmin_speed, vmax=vmax_speed,
            transform=ccrs.PlateCarree()
        )
        ax_out.quiver(
            gt_lon[::stride], gt_lat[::stride], 
            u_out[::stride, ::stride], v_out[::stride, ::stride],
            transform=ccrs.PlateCarree(), scale=50, scale_units='inches',
            width=0.003, color='black', alpha=0.7
        )
        ax_out.coastlines(resolution='10m', linewidth=0.8)
        ax_out.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax_out.set_extent([gt_lon[0], gt_lon[-1], gt_lat[0], gt_lat[-1]], ccrs.PlateCarree())
        
        last_im = im
    
    # fig.suptitle(
    #     f"Wind Fields at {time_key}\nMax = {vmax_speed:.2f}, Min = {vmin_speed:.2f}", 
    #     fontsize=16, y=0.98
    # )
    
    # 調整整體排版：水平方向間距進一步縮小 (wspace)
    fig.subplots_adjust(left=0.03, right=0.94, top=0.9, bottom=0.05, wspace=0.01, hspace=0.001)
    
    # 調整 colorbar，使其與所有子圖高度一致，並設定 fraction 使其更寬
    cbar = fig.colorbar(last_im, ax=axes[1:,:], orientation='vertical', fraction=0.04, pad=0.02)
    cbar.set_label("Wind Speed (m/s)")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def load_mean_std_config():
    """
    從 dataset_opt["topographical_data_root"] 讀取 mean_std_config.yml，
    整理成一個 SimpleNamespace 物件，方便以屬性方式存取。
    """
    root = "/home/nycustd/system/flask_app/data/raw/3_Topographical_Data"
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
    
class InferenceDataset(Dataset):
    def __init__(self, base_dataset, selected_times):
        """
        base_dataset: 例如 ERA5_CWA_200_interpolation_ts_tp2_obs2_Dataset 的實例，裡面已包含所有時間點資料
        selected_times: list，包含要 inference 的時間點（型態需與 base_dataset中每筆資料['gt_time']相符，例如 np.datetime64 或字串）
        """
        self.base_dataset = base_dataset
        # 如果 selected_times 是字串，則轉換成 np.datetime64
        self.selected_times = [np.datetime64(t) if not isinstance(t, np.datetime64) else t for t in selected_times]

        # 利用每筆資料的 'gt_time' 來篩選資料索引
        self.indices = []
        for i in range(len(self.base_dataset)):
            item = self.base_dataset[i]
            # 假設每筆資料中 'gt_time' 能轉換成 np.datetime64
            gt_time = np.datetime64(item['gt_time'])
            if gt_time in self.selected_times:
                self.indices.append(i)
        if not self.indices:
            # print("Dataset 中的 gt_time:")
            # for i in range(len(self.base_dataset)):
            #     print(np.datetime64(self.base_dataset[i]['gt_time']))
            raise ValueError("選定的時間點在資料集中沒有對應！")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        return self.base_dataset[orig_idx]
        

def inference_pipeline(experiment_list=None, iteration_list=None, time_list=None):
    """
    - experiment_list: 實驗/模型名稱列表
    - iteration_list: 每個模型需載入的權重檔（順序與 experiment_list 對應）
    - time_list: 推理資料的時間點列表（例如 '2023-01-24T06:00'）
    """
    import torch
    from torch.utils.data import DataLoader
    from basicsr.utils.registry import DATASET_REGISTRY
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的 device: {device}")
    time_list = [np.datetime64(t) for t in time_list]
    if experiment_list is None or iteration_list is None or time_list is None:
        raise ValueError("請提供 experiment_list、iteration_list 與 time_list！")
    
    # 如果 iteration_list 長度不足，複製第一個值使其數量與 experiment_list 相同
    if len(iteration_list) < len(experiment_list):
        iteration_list = [iteration_list[0]] * len(experiment_list)

    norm_params = load_mean_std_config()  # 載入 normalization 參數

    # ------------------ Phase 1: Inference ------------------
    results_dict = {t: {} for t in time_list}
    
    for exp, iter_ in zip(experiment_list, iteration_list):
        opt = load_yaml_config(exp)
        model = load_model_pth(exp, iter_, opt).to(device)
        dataset_opt = opt['datasets']['train']
        dataset_opt['start_year'] = 2023
        dataset_opt['end_year'] = 2023
        dataset_opt['start_month'] = 1
        dataset_opt['end_month'] = 12
        dataset_cls = DATASET_REGISTRY.get(dataset_opt['type'])
        base_dataset = dataset_cls(dataset_opt)
        inf_dataset = InferenceDataset(base_dataset, selected_times=time_list)
        dataloader = DataLoader(inf_dataset, batch_size=1, shuffle=False)
        gt_lat = base_dataset.tp_lat
        gt_lon = base_dataset.tp_lon
        apply_wt_minshift = dataset_opt.get('apply_wt_minshift', False)
        apply_zscore = dataset_opt.get('apply_zscore', False)
        input_params = dataset_opt.get('input_params', [])
        output_params = dataset_opt.get('output_params', [])
        
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                input_data = {
                    'lq': data['lq'].to(device),
                    'tp': data['tp'].to(device) if 'tp' in data else None,
                    'trnd': data['trnd'].to(device) if 'trnd' in data else None,
                    'trns': data['trns'].to(device) if 'trns' in data else None,
                    'adj': data['adj'].to(device) if 'adj' in data else None
                }
                output = model(input_data)
                if isinstance(output, tuple):
                    output = output[0]
                data['lq'] = data['lq'].squeeze(0)
                data['gt'] = data['gt'].squeeze(0)
                output = output.squeeze(0)
                
                # Inverse transformation
                lq_data_inv = data['lq'].clone()
                if apply_wt_minshift:
                    for i, param in enumerate(input_params):
                        lq_data_inv[i] = inverse_transform(lq_data_inv[i], norm_params, param, method='minshift', domain='era5')
                elif apply_zscore:
                    for i, param in enumerate(input_params):
                        lq_data_inv[i] = inverse_transform(lq_data_inv[i], norm_params, param, method='zscore', domain='era5')
                output_data_inv = output.clone()
                if apply_wt_minshift:
                    for i, param in enumerate(output_params):
                        output_data_inv[i] = inverse_transform(output_data_inv[i], norm_params, param, method='minshift', domain='cwa')
                elif apply_zscore:
                    for i, param in enumerate(output_params):
                        output_data_inv[i] = inverse_transform(output_data_inv[i], norm_params, param, method='zscore', domain='cwa')
                if data['gt'] is not None:
                    gt_data_inv = data['gt'].clone()
                    if apply_wt_minshift:
                        for i, param in enumerate(output_params):
                            gt_data_inv[i] = inverse_transform(gt_data_inv[i], norm_params, param, method='minshift', domain='cwa')
                    elif apply_zscore:
                        for i, param in enumerate(output_params):
                            gt_data_inv[i] = inverse_transform(gt_data_inv[i], norm_params, param, method='zscore', domain='cwa')
                else:
                    gt_data_inv = None
                
                time_key = time_list[batch_idx]
                results_dict[time_key][exp] = {"gt": gt_data_inv.clone(), "output": output_data_inv.clone(), "lq": lq_data_inv.clone()}
                print(f"Experiment {exp} at time {time_key} inference done.")
        del model, base_dataset, inf_dataset, dataloader
        torch.cuda.empty_cache()
    
    # ------------------ Phase 2: Visualization ------------------
    # 對於每個 time_key，生成 Combined Wind Fields 與 Combined Differences 圖像
    for time_key in time_list:
        res_t = results_dict[time_key]  # dict { exp: {...} }
        combined_dir = os.path.join("visualizations", "combined")
        os.makedirs(combined_dir, exist_ok=True)
        save_path1 = os.path.join(combined_dir, f"combined_wind_{time_key}.png")
        
        visualize_combined_wind_fields(time_key, res_t, gt_lat, gt_lon, save_path1)
    
    return results_dict

dataset_opt = {
    'name': 'ERA5_CWA_val',
    'type': 'ERA5_CWA_200_interpolation_ts_tp2_obs2_Dataset',
    'meteorological_data_root': '../../0_Data/1_Meteorological_Data',
    'observational_data_root': '../../0_Data/2_Observational_Data',
    'topographical_data_root': '../../0_Data/3_Topographical_Data',
    'start_year': 2023,
    'start_month': 1,
    'end_year': 2023,
    'end_month': 2,

    'source_version': 'ERA5_corrdiff_v1',
    'target_version': 'CWA_Taiwan_v2.4',
    'topo_version': 'CWA_Taiwan_v3.2',
    'topo_filename': 'CWA_Taiwan_v3.1.2_terrain.nc',
    'obs_version': 'obs_v5',

    'input_params': ['t2m', 'ws10', 'u10', 'v10', 'q', 'd2m'],
    'output_params': ['u10', 'v10'],
    'obs_params': ["TX", "PS", "PP", "RH", "WD", "WS"],
    'topo_params': ['terrain'],
    'time_steps': 8,
    'obs_time_steps': 16,
    'apply_zscore': True,

    'io_backend':
      {'type': 'disk'},
    'scale': 10
}

# # input
# experiment_list = [
#     '/home/hujiahua/Weather/flask_app/modules/Baseline/experiments/250408_UformerLSTM3Obs2TP3_CWA_org_ntp_16ts_obs16ts_96b_zscore_2e-4lr_0.7dr_1e-5wd_1gpu_train01',
# ]
# # experiment_name = [
# #     'SOTA',
# #     'Model1',
# #     'Model2'
# # ]
# iteration_list = [
#     'latest'
# ]
# time_list = [
#     '2023-01-24T06:00',
#     '2023-04-14T06:00',
#     '2023-07-28T00:00',
#     '2023-11-11T12:00'
# ]

# inference_pipeline(experiment_list=experiment_list,
#                 #    dataset_opt = dataset_opt,
#                    iteration_list=iteration_list,
#                    time_list=time_list)