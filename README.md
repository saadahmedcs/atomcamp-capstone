# Autonomous Crop and Weed Detection System

An end-to-end computer vision pipeline and edge-ready serving engine designed for **Site-Specific Weed Management (SSWM)**. This system fine-tunes a lightweight **YOLOv8n** architecture to discriminate between cash crops and invasive weeds in top-down field imagery, calculates real-time **Weed Infestation Pressure (%)**, and serves predictions via an asynchronous **FastAPI** web dashboard and REST API.

---

## Key Performance Highlights

- **Detection Accuracy:** 81.9% mAP@50 (Crop: 81.2%, Weed: 82.6%)
- **Precision:** 83.9%
- **Recall:** 76.4%
- **Inference Latency:** 4.7 ms on Tesla T4 GPU | ~240 ms on Intel Core i5 CPU
- **Export Format:** PyTorch (`.pt`) and optimized ONNX runtime (`.onnx`)

---

## Repository Structure
```
weed-detector-capstone/
├── .gitignore               # Excludes datasets, caches, venv, and binary weights
├── requirements.txt         # Production and training dependency specifications
├── data.yaml                # YOLO dataset configuration and class mapping
├── split_dataset.py         # 70/20/10 train-val-test partitioning script
├── check_labels.py          # Bounding-box coordinate sanity visualizer
├── weed_detector.ipynb      # Cloud training and evaluation workflow (Google Colab)
└── app.py                   # FastAPI inference backend & diagnostic dashboard
```

---

## Dataset Overview & Preparation

The model was trained on 1,300 high-resolution top-down agricultural field images annotated in standard YOLO bounding-box format across two target classes:
- `0: Crop`
- `1: Weed`

### Partitioning & Splits
- **Train Set:** 909 images (70%)
- **Validation Set:** 260 images (20%)
- **Test Set:** 131 images (10%)

### Preprocessing & Augmentation
Images are standardized to 640x640 pixels. Real-world lighting variations, occlusions, and canopy shadows are simulated via Ultralytics native augmentations:
- Mosaic augmentation (1.0)
- HSV color jitter (hue: 0.015, saturation: 0.7, value: 0.4)
- Random horizontal flips (0.5) and minor affine rotations

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10 to 3.12 (or WSL Ubuntu / Windows 11 PowerShell)
- Git

### 2. Clone Repository & Setup Virtual Environment

git clone https://github.com/saadahmedcs/weed-detector-capstone.git
cd weed-detector-capstone

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS / WSL
source .venv/bin/activate

### 3. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

---

## Data Validation & Splitting

If training from raw data archives:

# 1. Partition raw dataset into 70/20/10 splits
python split_dataset.py

# 2. Run visual sanity check on bounding boxes
python check_labels.py

---

## Model Training & Export

Training is executed using Ultralytics YOLOv8n. To run on a GPU-enabled machine or Google Colab:

from ultralytics import YOLO

# Initialize model
model = YOLO("yolov8n.pt")

# Train for 30 epochs
results = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    project="weed_detector_runs",
    name="yolov8n_capstone",
    seed=42,
    save=True
)

# Export weights to ONNX for optimized serving
best_model = YOLO("weed_detector_runs/yolov8n_capstone/weights/best.pt")
best_model.export(format="onnx")

---

## Serving & Inference Engine

Ensure `best.pt` is placed in the project root directory.

### Launch Local Server

uvicorn app:app --reload --port 8000

Once started:
- **Interactive Web Dashboard:** Navigate to http://localhost:8000 to upload field images, inspect bounding box overlays, and view calculated infestation ratios.
- **Swagger API Documentation:** Navigate to http://localhost:8000/docs to test endpoints interactively.

### Programmatic REST API Usage

#### Endpoint
`POST /api/v1/detect`

#### cURL Request
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/field_sample.jpg"

#### JSON Response Schema
{
  "total_detected": 4,
  "crops": 3,
  "weeds": 1,
  "infestation_pressure_percent": 25.0,
  "spray_recommended": false
}

---

## Operational Spray Logic

The inference engine calculates canopy pressure using the following ratio:

Weed Infestation Pressure (%) = (Weeds Detected / Total Detected Plants) * 100

- **Pressure > 30.0%:** Triggers `HIGH WEED PRESSURE` alert; spot-spraying mechanism engaged.
- **Pressure <= 30.0%:** Triggers `OPTIMAL CANOPY HEALTH` alert; chemical application suppressed.