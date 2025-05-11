Weather Downscaling Visualization Tool

一個基於 Flask + Leaflet 的氣象降尺度可視化網頁工具，可對原始風場與降水資料進行降尺度處理，並將結果顯示在互動式地圖上。

功能特色

降尺度模組：

modules/downscaling_wind.py：單時刻風場降尺度

modules/downscaling_rain.py：單時刻降水降尺度

可視化模組：

modules/visualization.py、modules/visualization_rain.py：將降尺度後資料轉換為 GeoJSON

供前端 Leaflet 地圖直接讀取並動態疊加

Web API：

app.py：Flask 主程式，定義各路由

rain_route.py：降水專用 API 路由

wsgi.py：WSGI 介面，用於生產環境部署

前端資源：

templates/：HTML 模板

static/：CSS、JavaScript、圖片等靜態檔案

環境需求

Python 3.8+

建議使用虛擬環境 (venv 或 Conda)

安裝步驟

Clone 本專案：

git clone https://github.com/dilabproj/Weather-Downscaling-Visualization-Tool.git
cd Weather-Downscaling-Visualization-Tool/flask_app

建立並啟動虛擬環境：

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

安裝相依套件：

pip install -r requirements.txt

執行服務

# 本機開發模式
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000

開啟瀏覽器，前往 http://<SERVER_IP>:5000 即可使用。

專案目錄結構

flask_app/
├── modules/               # 核心運算與可視化模組
│   ├── downscaling_wind.py
│   ├── downscaling_rain.py
│   ├── visualization.py
│   ├── visualization_rain.py
│   └── utils.py
├── static/                # CSS/JS/圖片等前端靜態資源
├── templates/             # HTML 模板
├── app.py                 # Flask 主程式
├── rain_route.py          # 降水專用 API 路由
├── wsgi.py                # WSGI 介面
├── requirements.txt       # 相依套件列表
└── .gitignore

參數設定

mean_std_config.yml（若有）：放在 data/ 目錄，用於指定資料正規化參數。

可在 modules/utils.py 中調整預設參數，如網格範圍、投影設定等。

貢獻與回報

歡迎提交 Issue 或 Pull Request，如有建議或 bug 回報，請至 GitHub Repo 提交。

