# # modules/visualization.py

# import numpy as np

# def ds_to_geojson(ds, var_name='u10'):
#     """
#     將 ds 中的指定變量（u10, v10, speed）轉換成網格型 GeoJSON，
#     並只輸出指定範圍內的多邊形：
#         Latitude: [21.75, 25.50]
#         Longitude: [119.0, 122.76]

#     - 如果資料集的經緯度名稱為 'XLAT','XLONG'，會先重命名為 'latitude','longitude'。
#     - 當 var_name 為 'speed' 時，從 ds['u10'] 與 ds['v10'] 計算風速與原始分量；
#       否則只讀取 ds[var_name] 並根據名稱判斷 raw_u/raw_v。
#     - 每個 Feature.properties 包含：
#         - value: 絕對風速或分量值
#         - raw_u, raw_v: 用於前端繪製箭頭
#     """
#     import numpy as np, math
#     # 重新命名經緯度
#     if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
#         if 'XLAT' in ds.variables and 'XLONG' in ds.variables:
#             ds = ds.set_coords(['XLAT', 'XLONG'])
#             ds = ds.rename({'XLAT': 'latitude', 'XLONG': 'longitude'})
#         if 'lat' in ds.variables and 'lon' in ds.variables:
#             ds = ds.set_coords(['lat', 'lon'])
#             ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
#         else:
#             raise ValueError("Dataset 不含 'latitude/longitude' 或 'XLAT/XLONG'，無法轉換 GeoJSON！")

#     lat_vals = ds['latitude'].values
#     lon_vals = ds['longitude'].values

#     # 若 var_name 為 speed，分別取 u10, v10；否則只讀取對應變量
#     if var_name == 'speed':
#         u_vals = ds['u10'].values
#         v_vals = ds['v10'].values
#     else:
#         data_vals = ds[var_name].values

#     LAT_MIN, LAT_MAX = 21.624, 25.626
#     LON_MIN, LON_MAX = 118.874, 122.876

#     features = []
#     for i in range(len(lat_vals) - 1):
#         for j in range(len(lon_vals) - 1):
#             # 計算 raw_u, raw_v 與 value
#             if var_name == 'speed':
#                 raw_u = float(u_vals[i, j])
#                 raw_v = float(v_vals[i, j])
#                 val = math.sqrt(raw_u**2 + raw_v**2)
#             else:
#                 raw = float(data_vals[i, j])
#                 if var_name.lower().startswith('u'):
#                     raw_u, raw_v = raw, 0.0
#                     val = abs(raw_u)
#                 elif var_name.lower().startswith('v'):
#                     raw_u, raw_v = 0.0, raw
#                     val = abs(raw_v)
#                 else:
#                     raw_u, raw_v = 0.0, 0.0
#                     val = abs(raw)

#             if np.isnan(val):
#                 val = None

#             lat1, lat2 = lat_vals[i], lat_vals[i+1]
#             lon1, lon2 = lon_vals[j], lon_vals[j+1]
#             tile_lat_min, tile_lat_max = min(lat1, lat2), max(lat1, lat2)
#             tile_lon_min, tile_lon_max = min(lon1, lon2), max(lon1, lon2)

#             # 跳過固定範圍外
#             if (tile_lat_max < LAT_MIN or tile_lat_min > LAT_MAX
#                or tile_lon_max < LON_MIN or tile_lon_min > LON_MAX):
#                 continue

#             coords = [
#                 [lon1, lat1],
#                 [lon2, lat1],
#                 [lon2, lat2],
#                 [lon1, lat2],
#                 [lon1, lat1],
#             ]

#             feature = {
#                 "type": "Feature",
#                 "properties": {"value": val, "raw_u": raw_u, "raw_v": raw_v},
#                 "geometry": {"type": "Polygon", "coordinates": [coords]}
#             }
#             features.append(feature)

#     return {"type": "FeatureCollection", "features": features}

# modules/visualization.py

import numpy as np
import math

def ds_to_geojson(ds, var_name='u10'):
    """
    將 xarray Dataset ds 中的指定變量 (u10, v10, speed 或其他單一屬性)
    轉換成網格型 GeoJSON，並輸出 Nlat×Nlon 個多邊形 (Polygon) cell。

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

    # 計算「半步距」，假設每個維度等距
    # 若實際不等距，可改成 per‐cell 計算 (lat[i+1]-lat[i])/2
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("latitude/longitude 長度需 >= 2 才能計算步距。")
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    hlat = dlat / 2.0
    hlon = dlon / 2.0

    features = []
    # 對每一個格點中心 (i, j) 建立一個 Polygon
    for i in range(len(lat)):
        for j in range(len(lon)):
            # 1) 讀取原始數值 & 計算 value, raw_u, raw_v
            if var_name == 'speed':
                u = float(ds['u10'].values[i, j])
                v = float(ds['v10'].values[i, j])
                val   = math.sqrt(u*u + v*v)
                raw_u = u
                raw_v = v
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

            # 2) 計算 cell 四角 (中心 ± 半步距)
            lat_min = lat[i] - hlat
            lat_max = lat[i] + hlat
            lon_min = lon[j] - hlon
            lon_max = lon[j] + hlon

            coords = [
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min],
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

