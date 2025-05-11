# modules/visualization.py

import numpy as np
import math

def ds_to_geojson(ds, var_name='u10'):
    """
    將 xarray Dataset ds 中的指定變量 (u10, v10, speed 或其他單一屬性)
    轉換成網格型 GeoJSON，並輸出 Nlat×Nlon 個多邊形 (Polygon) cell，
    同時僅保留在指定紅框範圍內的區塊（將超出部分裁剪）。

    參數:
        ds (xarray.Dataset or DataArray): 必須含有 'latitude' 和 'longitude' 座標，
            若為 'speed' 將分別取 ds['u10']、ds['v10'] 計算風速。
        var_name (str): 欲輸出的變量名稱；'speed'、'u10'、'v10' 或其他。

    回傳:
        dict: GeoJSON FeatureCollection。
    """
    # --- 經緯度座標重命名 (若必要) ---
    if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
        if 'XLAT' in ds.variables and 'XLONG' in ds.variables:
            ds = ds.set_coords(['XLAT', 'XLONG']) \
                   .rename({'XLAT': 'latitude', 'XLONG': 'longitude'})
        elif 'lat' in ds.variables and 'lon' in ds.variables:
            ds = ds.set_coords(['lat', 'lon']) \
                   .rename({'lat': 'latitude', 'lon': 'longitude'})
        else:
            raise ValueError("Dataset 不含 'latitude/longitude' 或 'XLAT/XLONG'，無法轉換 GeoJSON！")

    # 取出經緯度 array
    lat = ds['latitude'].values  # shape: (Nlat,)
    lon = ds['longitude'].values # shape: (Nlon,)

    # 計算半步距 (假設等距)
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("latitude/longitude 長度需 >= 2 才能計算步距。")
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    hlat = dlat / 2.0
    hlon = dlon / 2.0

    # 定義紅框範圍 (可依需求微調)
    LAT_MIN, LAT_MAX = 21.624, 25.626
    LON_MIN, LON_MAX = 118.874, 122.876

    features = []

    # 逐一為每個中心格點建立 Polygon，並裁剪到紅框範圍
    for i in range(len(lat)):
        for j in range(len(lon)):
            # --- 1) 讀取原始數值 & 計算 value, raw_u, raw_v ---
            if var_name == 'speed':
                u = float(ds['u10'].values[i, j])
                v = float(ds['v10'].values[i, j])
                val = math.sqrt(u*u + v*v)
                raw_u, raw_v = u, v
            else:
                d = float(ds[var_name].values[i, j])
                if var_name.lower().startswith('u'):
                    raw_u, raw_v, val = d, 0.0, abs(d)
                elif var_name.lower().startswith('v'):
                    raw_u, raw_v, val = 0.0, d, abs(d)
                else:
                    raw_u, raw_v, val = 0.0, 0.0, abs(d)

            if np.isnan(val):
                val = None

            # --- 2) 原始 cell 四角 (中心 ± 半步距) ---
            lat0 = lat[i]
            lon0 = lon[j]
            tile_lat_min = lat0 - hlat
            tile_lat_max = lat0 + hlat
            tile_lon_min = lon0 - hlon
            tile_lon_max = lon0 + hlon

            # --- 3) 裁剪到紅框範圍內 ---
            #   將外側部分 clamp 到 [LAT_MIN, LAT_MAX] / [LON_MIN, LON_MAX]
            tile_lat_min = max(tile_lat_min, LAT_MIN)
            tile_lat_max = min(tile_lat_max, LAT_MAX)
            tile_lon_min = max(tile_lon_min, LON_MIN)
            tile_lon_max = min(tile_lon_max, LON_MAX)

            # 如果裁剪後已無面積，就跳過
            if tile_lat_min >= tile_lat_max or tile_lon_min >= tile_lon_max:
                continue

            # --- 4) 建立裁剪後的多邊形座標 ---
            coords = [
                [tile_lon_min, tile_lat_min],
                [tile_lon_max, tile_lat_min],
                [tile_lon_max, tile_lat_max],
                [tile_lon_min, tile_lat_max],
                [tile_lon_min, tile_lat_min],
            ]

            features.append({
                "type": "Feature",
                "properties": {
                    "value": val,
                    "raw_u": raw_u,
                    "raw_v": raw_v
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            })

    return {
        "type": "FeatureCollection",
        "features": features
    }
