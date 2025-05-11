# Weather Downscaling Visualization Tool

一個基於 Flask + Leaflet 的氣象降尺度可視化網頁工具，可對原始風場與降水資料進行降尺度處理，並將結果顯示在互動式地圖上。

---

## 功能特色

- **降尺度模組**  
  - `modules/downscaling_wind.py`：風場降尺度  
  - `modules/downscaling_rain.py`：降水降尺度

- **可視化模組**  
  - `modules/visualization.py`、`modules/visualization_rain.py`：將降尺度後資料轉換為 GeoJSON  
  - 供前端 Leaflet 地圖直接讀取並動態疊加

- **Web API**  
  - `app.py`：Flask 主程式  
  - `rain_route.py`：降水專用 API 
  - `wsgi.py`：WSGI 介面，用於生產環境部署

- **前端資源**  
  - `templates/`：HTML 模板  
  - `static/`：CSS、JavaScript、圖片等靜態檔案

---

## 安裝步驟

1. Clone 本專案：
   ```bash
   git clone https://github.com/dilabproj/Weather-Downscaling-Visualization-Tool.git
   cd Weather-Downscaling-Visualization-Tool/flask_app
   ```
2. 建立並啟動虛擬環境：
  ```bash
  python3 -m venv venv
  source venv/bin/activate   # Windows: venv\Scripts\activate
  ```
3. 安裝套件：
   ```bash
   pip install -r requirements.txt
   ```


