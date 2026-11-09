import io
import base64
import time
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = FastAPI(
    title="Autonomous Crop & Weed Detection API",
    description="Edge-ready inference engine for Site-Specific Weed Management",
    version="1.0.0"
)

# Load trained weights once on startup
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# ==========================================
# 1. HEADLESS REST API (Used by Web UI & Robots)
# ==========================================
@app.post("/api/v1/detect")
async def detect_crops_and_weeds(
    file: UploadFile = File(...),
    conf: float = Query(0.35, ge=0.05, le=1.0),
    iou: float = Query(0.45, ge=0.1, le=1.0)
):
    start_time = time.perf_counter()
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_width, img_height = image.size

    # Run YOLOv8 inference
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    result = results[0]
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 1)

    crop_count = 0
    weed_count = 0
    boxes_data = []

    # Parse detections
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        label = model.names[cls_id]
        if cls_id == 0:
            crop_count += 1
        elif cls_id == 1:
            weed_count += 1

        boxes_data.append({
            "class_id": cls_id,
            "label": label,
            "confidence": round(confidence, 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        })

    total_plants = crop_count + weed_count
    weed_pressure = round((weed_count / total_plants * 100), 1) if total_plants > 0 else 0.0
    spray_recommended = weed_pressure > 30.0

    # Render annotated image to base64
    annotated_arr = result.plot()  # BGR numpy array
    annotated_img = Image.fromarray(annotated_arr[..., ::-1])  # to RGB
    buffer = io.BytesIO()
    annotated_img.save(buffer, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return JSONResponse(content={
        "status": "success",
        "telemetry": {
            "inference_time_ms": inference_time_ms,
            "image_resolution": [img_width, img_height]
        },
        "agronomics": {
            "crop_count": crop_count,
            "weed_count": weed_count,
            "total_plants": total_plants,
            "weed_pressure_pct": weed_pressure,
            "spray_recommended": spray_recommended
        },
        "detections": boxes_data,
        "annotated_image_base64": f"data:image/jpeg;base64,{img_b64}"
    })


# ==========================================
# 2. MODERN DASHBOARD (GET /)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    # Return the HTML dashboard generated in the canvas
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()