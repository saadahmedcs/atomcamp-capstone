import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Weed & Crop Detection Engine")
model = YOLO("best.pt")

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Autonomous Weed Detection Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px auto; max-width: 800px; padding: 0 20px; color: #222; }
        .box { border: 2px dashed #bbb; border-radius: 8px; padding: 30px; text-align: center; background: #fafafa; }
        button { background: #2e7d32; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 4px; cursor: pointer; margin-top: 15px; }
        button:hover { background: #1b5e20; }
    </style>
</head>
<body>
    <h2>🌾 Crop & Weed Detection Dashboard</h2>
    <p>Upload top-down field imagery to classify crop rows and calculate weed pressure.</p>
    <div class="box">
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br>
            <button type="submit">Analyze Field Image</button>
        </form>
    </div>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    results = model.predict(image, conf=0.35)
    result = results[0]
    
    classes = result.boxes.cls.tolist()
    crops = classes.count(0)
    weeds = classes.count(1)
    total = len(classes)
    weed_ratio = (weeds / total * 100) if total > 0 else 0.0
    
    if weed_ratio > 30.0:
        alert_class = "alert-high"
        alert_text = f"⚠️ HIGH WEED PRESSURE ({weed_ratio:.1f}%) — Targeted Spot Spraying Required."
    else:
        alert_class = "alert-low"
        alert_text = f"✅ OPTIMAL CANOPY HEALTH ({weed_ratio:.1f}% weed ratio) — Blanket Spraying Suppressed."

    annotated_array = result.plot()
    annotated_img = Image.fromarray(annotated_array[..., ::-1])
    buf = io.BytesIO()
    annotated_img.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Field Diagnostics</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px auto; max-width: 800px; padding: 0 20px; color: #222; }}
        .card {{ padding: 20px; border-radius: 8px; background: #f1f8e9; border: 1px solid #c8e6c9; }}
        .metrics-grid {{ display: flex; justify-content: space-around; margin: 15px 0; text-align: center; }}
        .metric-val {{ font-size: 26px; font-weight: bold; color: #1b5e20; }}
        .metric-label {{ font-size: 13px; color: #555; text-transform: uppercase; }}
        .alert {{ padding: 10px 15px; border-radius: 4px; font-weight: bold; margin-bottom: 15px; }}
        .alert-high {{ background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }}
        .alert-low {{ background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }}
        a {{ display: inline-block; margin-top: 20px; text-decoration: none; color: #1565c0; font-weight: bold; }}
        img {{ max-width: 100%; border-radius: 8px; margin-top: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h2>Diagnostic Results</h2>
    <div class="alert {alert_class}">
        {alert_text}
    </div>
    <div class="card">
        <div class="metrics-grid">
            <div>
                <div class="metric-val">{total}</div>
                <div class="metric-label">Total Plants</div>
            </div>
            <div>
                <div class="metric-val">{crops}</div>
                <div class="metric-label">Crops Detected</div>
            </div>
            <div>
                <div class="metric-val">{weeds}</div>
                <div class="metric-label">Weeds Detected</div>
            </div>
            <div>
                <div class="metric-val">{weed_ratio:.1f}%</div>
                <div class="metric-label">Infestation Pressure</div>
            </div>
        </div>
    </div>
    <h3>Annotated Detection Field:</h3>
    <img src="data:image/jpeg;base64,{img_b64}" alt="Inference Output">
    <br>
    <a href="/">← Analyze Another Image</a>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.post("/api/v1/detect")
async def detect_api(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model.predict(image, conf=0.35)
    r = results[0]
    
    classes = r.boxes.cls.tolist()
    crops = classes.count(0)
    weeds = classes.count(1)
    total = len(classes)
    weed_ratio = (weeds / total * 100) if total > 0 else 0.0
    
    return {
        "total_detected": total,
        "crops": crops,
        "weeds": weeds,
        "infestation_pressure_percent": round(weed_ratio, 2),
        "spray_recommended": weed_ratio > 30.0
    }