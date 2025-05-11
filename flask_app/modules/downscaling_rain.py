import torch
from torch import nn as nn
from basicsr.archs.arch_util import Upsample, make_layer
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from datetime import datetime


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), 
            nn.ReLU(True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),  # 改成 LeakyReLU
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x

# class RCAB(nn.Module):
#     def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
#         super(RCAB, self).__init__()
#         self.res_scale = res_scale

#         self.rcab = nn.Sequential(
#             nn.Conv2d(num_feat, num_feat, 3, 1, 1),
#             nn.BatchNorm2d(num_feat),  # 加入 BatchNorm
#             nn.ReLU(True),
#             nn.Conv2d(num_feat, num_feat, 3, 1, 1),
#             nn.BatchNorm2d(num_feat),  # 加入 BatchNorm
#             ChannelAttention(num_feat, squeeze_factor)
#         )

#     def forward(self, x):
#         res = self.rcab(x) * self.res_scale
#         return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


class RCAN_2(nn.Module):
    """Residual Channel Attention Networks.

    ``Paper: Image Super-Resolution Using Very Deep Residual Channel Attention Networks``

    Reference: https://github.com/yulunzhang/RCAN

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=20,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(RCAN_2, self).__init__()

        self.img_range = img_range
        if len(rgb_mean) == 3:
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        elif len(rgb_mean) == 1:
            self.mean = torch.Tensor(rgb_mean).view(1, 1, 1, 1)
        else:
            raise ValueError(f"Invalid rgb_mean: {rgb_mean}. Must be a list or tuple with 1 or 3 elements.")

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(25, 25), mode='bicubic', align_corners=False)
        self.mean = self.mean.type_as(x)

        if x.size(1) == 1:
            x = (x - self.mean[:, :1, :, :]) * self.img_range  # 單通道情況
        else:
            x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean
        
        return x

class WT_dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.gt_path = opt['dataroot_gt']
        self.lq_path = opt['dataroot_lq']
        self.gt_files = sorted([f for f in os.listdir(self.gt_path) if f.endswith('.nc')])
        self.lq_files = sorted([f for f in os.listdir(self.lq_path) if f.endswith('.nc')])
        self.time_coords = self._load_time_coords(self.gt_path, self.gt_files)

        # 一次性讀取所有數據到記憶體，並統一維度名稱
        self.gt_data = self._load_all_xarray(self.gt_path, self.gt_files, 'pr', is_era5=False)
        self.lq_data = self._load_all_xarray(self.lq_path, self.lq_files, 'pr', is_era5=True)
        # 將 list 轉換為 4D 張量 (time, height, width)
        self.gt_data = torch.cat(self.gt_data, dim=0).unsqueeze(0)  # [1, T, H, W]
        self.lq_data = torch.cat(self.lq_data, dim=0).unsqueeze(0)  # [1, T, H, W]

        print(self.gt_data.shape) # torch.Size([52583, 200, 200])

    def __getitem__(self, index): 
        return {'gt': self.gt_data[:, index, :, :], 'lq': self.lq_data[:, index, :, :], 'lq_path': self.lq_path, 'gt_path': self.gt_path, 'time_value': str(self.time_coords[index])}

    def __len__(self):
        # 返回資料集的長度，即時間步的數量
        return self.lq_data.shape[1]

    def _load_time_coords(self, path, files):
        time_list = []
        for filename in files:
            dataset = xr.open_dataset(os.path.join(path, filename))
            time_values = dataset['time'].values
            time_list.extend(time_values)
            dataset.close()
        return time_list

    def _load_all_xarray(self, path, files, variable_name, is_era5):
        data_list = []
        for filename in files:
            dataset = xr.open_dataset(os.path.join(path, filename))

            data_array = dataset[variable_name].transpose('time', 'lat', 'lon')
            data_tensor = torch.tensor(data_array.values, dtype=torch.float32)
            
            data_list.append(data_tensor)
            dataset.close()
        
        return data_list
    
def load_model(model_path):
    model = RCAN_2(
        num_in_ch=1,
        num_out_ch=1,
        num_feat=64,
        num_group=10,
        num_block=20,
        squeeze_factor=16,
        upscale=8,
        res_scale=0.8,
        img_range=216.34548950195312,
        rgb_mean=[0.20883285999298096]
    )
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.cuda() 
    return model

def load_test_dataset(batch_size):

    test_dataset_opt = {
        'dataroot_gt': '/home/nycustd/system/flask_app/data/raw/1_Meteorological_Data/TReAD',
        'dataroot_lq': '/home/nycustd/system/flask_app/data/raw/1_Meteorological_Data/TReADl_re'
    }
    dataset = WT_dataset(test_dataset_opt)
    def custom_collate_fn(batch):
        collated = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([b[key] for b in batch])
            else:
                collated[key] = [b[key] for b in batch]
        return collated

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    return dataloader


def inference(model, dataloader, target_time, device='cuda'):
    model.eval()
    model.to(device)

    for i, data in enumerate(dataloader):
        current_time = pd.to_datetime(data['time_value'][0])  # 取 batch 中第一筆的時間
        if current_time == target_time:
            lq = data['lq'].to(device)
            gt = data['gt'].to(device)
            with torch.no_grad():
                output = model(lq)

            return output.squeeze().cpu().numpy() # 拿output取消註解這行

def load_lq_slice(year: int, month: int, day: int, hour: int):
    """
    只讀取 TReADl_re 中對應時間那一格的降水資料 (pr)，
    並且回傳一個 shape=(1,1,H,W) 的 torch.Tensor。
    """
    # 1) 決定用哪一個檔（train/val/test）
    base_dir = '/home/nycustd/system/flask_app/data/raw/1_Meteorological_Data/TReADl_re'
    if year <= 2021:
        fname = 'TReAD_train_16x16.nc'
    elif year == 2022:
        fname = 'TReAD_val_16x16.nc'
    else:
        fname = 'TReAD_test_16x16.nc'
    path = os.path.join(base_dir, fname)

    # 2) 打開 netCDF，只 sel 那個時間點
    ds = xr.open_dataset(path, engine='h5netcdf')
    time_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00"
    da = ds['pr'].sel(time=time_str, method='nearest').fillna(0)
    ds.close()

    # 3) 維度順序保證是 (lat, lon)，轉 numpy 再包成 tensor(1,1,lat,lon)
    arr = da.values.astype(np.float32)[None,None,:,:]
    return torch.from_numpy(arr)


def get_rain_output(year: int, month: int, day: int, hour: int,
                    model_path: str = '/home/nycustd/system/flask_app/modules/topic_2/models/net_g_latest.pth'):
    """
    執行模型推理，回傳 (lat, lon, downscaled_array)。
    downscaled_array shape = (Nlat_gt, Nlon_gt)
    """
    # --- (A) 模型載入 ---
    model = load_model(model_path)
    model.cuda().eval()

    # --- (B) 只載入單一時步的 LQ tensor ---
    lq = load_lq_slice(year, month, day, hour).cuda()  # shape=(1,1,H,W)

    # --- (C) 推理 ---
    with torch.no_grad():
        out = model(lq)          # shape still (1,1,H_out,W_out)
    out = out.squeeze().cpu().numpy()  # -> (H_out, W_out)

    # --- (D) 讀 GT sample 的經緯度 (只拿 coords，不丟整個 dataset) ---
    gt_dir = '/home/nycustd/system/flask_app/data/raw/1_Meteorological_Data/TReAD'
    sample_nc = sorted(f for f in os.listdir(gt_dir) if f.endswith('.nc'))[0]
    ds_gt = xr.open_dataset(os.path.join(gt_dir, sample_nc))
    lat = ds_gt['lat'].values
    lon = ds_gt['lon'].values
    ds_gt.close()

    return lat, lon, out
