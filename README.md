# Real-Time Multi-Camera Person Detection & Clothing Compliance System

A comprehensive computer vision system that performs real-time person detection, multi-camera tracking, and automatic clothing compliance monitoring using deep learning models.

## Features

- **Multi-Camera Person Detection**: Real-time detection using YOLOv8/YOLOv11
- **Advanced Tracking**: Deep OC-SORT algorithm for robust person tracking
- **Multi-Camera Re-Identification**: OSNet-based feature extraction for consistent identity across cameras
- **Clothing Detection**: Automated dress code compliance monitoring
- **Real-Time Dashboard**: Streamlit web interface for live monitoring
- **Audit Logging**: Comprehensive violation tracking with MongoDB integration

## Prerequisites

- Python 3.8 - Python 3.10 version above not support torchreid
- NVIDIA GPU (recommended) with CUDA support for optimal performance
- MongoDB Atlas account or local MongoDB instance
- Webcam(s) or video files for testing

## Installation

run the install.bat file in cmd 

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Person Detection"
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have an NVIDIA GPU, you may want to install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

### 4. Download Models

Place the following model files in the `model/` directory:

- **Person Detection**: `yolov8s.pt` or `yolov8m.pt` (automatically downloaded by Ultralytics)
- **Re-Identification**: `osnet_duke_reid.pth` (OSNet model)
- **Clothing Detection**: `best.pt` (custom YOLO model for clothing)
- **Shoe Detection**: `shoelast.pt` (custom YOLO model for shoes)

You can download YOLO models automatically, but you'll need to provide the Re-ID and custom models.

### 5. Configure MongoDB

#### Option A: MongoDB Atlas (Cloud)

1. Create a MongoDB Atlas account at https://www.mongodb.com/cloud/atlas
2. Create a new cluster and database
3. Create two collections: `evaluation` and `violation`
4. Get your connection string

#### Option B: Set Environment Variable

Set your MongoDB connection string as an environment variable:

```bash
# Windows (PowerShell)
$env:ATLAS_URI="mongodb+srv://username:password@cluster.mongodb.net/?appName=Real-timeClothingSystem"

# Windows (CMD)
set ATLAS_URI=mongodb+srv://username:password@cluster.mongodb.net/?appName=Real-timeClothingSystem

# Linux/Mac
export ATLAS_URI="mongodb+srv://username:password@cluster.mongodb.net/?appName=Real-timeClothingSystem"
```

#### Option C: Streamlit Secrets (For Web Dashboard)

Create a `.streamlit/secrets.toml` file:

```toml
atlas_uri = "mongodb+srv://username:password@cluster.mongodb.net/?appName=Real-timeClothingSystem"
```

Alternatively, you can modify the connection string directly in `py/db.py` (not recommended for production).

## Usage

### Command Line Interface

Run the main detection script:

```bash
python py/main.py --sources 0 --track --reid --cloth-detect
```

#### Common Arguments

- `--sources`: Camera indices or video file paths (e.g., `0` or `0 1` or `footage/c0.avi`)
- `--model`: Path to YOLO detection model (default: `./model/yolov8s.pt`)
- `--conf`: Confidence threshold (default: 0.75)
- `--track`: Enable Deep OC-SORT tracking
- `--reid`: Enable multi-camera re-identification (requires `--track`)
- `--cloth-detect`: Enable clothing detection (requires `--track`)
- `--reid-model`: Path to Re-ID model (default: `./model/osnet_duke_reid.pth`)
- `--cloth-model`: Path to clothing detection model (default: `./model/best.pt`)
- `--shoe-model`: Path to shoe detection model (default: `./model/shoelast.pt`)
- `--device`: Computation device (`cpu` or `cuda`, default: auto-detect)
- `--log-metadata`: Save detection metadata to CSV files

#### Example Commands

```bash
# Basic detection with single camera
python py/main.py --sources 0

# Multi-camera with tracking and re-identification
python py/main.py --sources 0 1 --track --reid --reid-model ./model/osnet_duke_reid.pth

# Full pipeline with clothing detection
python py/main.py --sources 0 --track --reid --cloth-detect \
    --cloth-model ./model/best.pt --shoe-model ./model/shoelast.pt

# Process video files
python py/main.py --sources footage/c0.avi footage/c1.avi --track --reid --cloth-detect

# Use GPU with custom confidence threshold
python py/main.py --sources 0 --device cuda --conf 0.8 --track --reid --cloth-detect
```

### Web Dashboard (Streamlit)

Launch the web interface:

```bash
streamlit run frontend/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### Dashboard Features

1. **Real-Time Monitoring** (`pages/1_Real_Time_Dashboard.py`)
   - Live multi-camera mosaic view
   - Real-time metrics (detections, violations, active cameras)
   - Performance monitoring (latency metrics)
   - Configurable parameters via sidebar
   - Color-coded bounding boxes:
     - 🟢 Green: Appropriate clothing
     - 🔴 Red: Violation detected
     - 🟡 Yellow: Pending evaluation

2. **Audit Log** (`pages/2_Audit_Log.py`)
   - Filter evaluations by date range and status
   - View violation details with images
   - Search functionality
   - Sortable records

## Project Structure

```
Person Detection/
├── py/                          # Backend processing modules
│   ├── main.py                  # Main CLI entry point
│   ├── PersonDetection.py       # Person detection utilities
│   ├── tracking.py             # Deep OC-SORT tracking implementation
│   ├── re_id.py                # Person re-identification (OSNet)
│   ├── db.py                   # MongoDB database operations
│   └── cloth_detection.py      # Clothing detection service
├── frontend/                    # Streamlit web application
│   ├── app.py                  # Main Streamlit app
│   └── pages/
│       ├── 1_Real_Time_Dashboard.py  # Real-time monitoring
│       └── 2_Audit_Log.py           # Audit log viewer
├── model/                       # Model files directory
│   ├── yolov8s.pt             # Person detection model
│   ├── osnet_duke_reid.pth    # Re-identification model
│   ├── best.pt                # Clothing detection model
│   └── shoelast.pt            # Shoe detection model
├── footage/                     # Sample video files
├── logo/                        # Application logos
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

### Tracking Parameters

- `--max-age`: Maximum frames to keep lost tracks alive (default: 30)
- `--min-hits`: Minimum consecutive hits before reporting a track (default: 3)
- `--track-iou`: IoU threshold for association (default: 0.3)
- `--similarity-lambda`: Blend factor between IoU and Re-ID similarity (default: 0.5)

### Re-Identification Parameters

- `--reid-threshold`: Cosine similarity threshold for matching (default: 0.7)
- `--reid-interval`: Run Re-ID every N frames (default: 10)

### Clothing Detection Parameters

- `--cloth-conf`: Confidence threshold for clothing detection (default: 0.25)
- `--cloth-interval`: Process every N frames per person (default: 30)
- `--cloth-min-size`: Minimum bounding box size in pixels (default: 70)
- `--cloth-edge-margin`: Edge margin as fraction of frame (default: 0.03)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce image size: `--imgsz 480`
   - Process fewer cameras simultaneously
   - Use smaller YOLO model: `--model ./model/yolov8s.pt`

2. **Camera Not Detected**
   - Use `--probe` to list available cameras: `python py/main.py --probe`
   - Check camera permissions
   - Try different camera indices: `--sources 0 1 2`

3. **MongoDB Connection Error**
   - Verify your connection string
   - Check network connectivity
   - Ensure MongoDB Atlas IP whitelist includes your IP

4. **Model Not Found**
   - Verify model files exist in `model/` directory
   - Check file paths in command arguments
   - YOLO models will auto-download if not found

5. **Low FPS Performance**
   - Enable GPU: `--device cuda`
   - Reduce image size: `--imgsz 480`
   - Disable unnecessary features (Re-ID, clothing detection)
   - Use TensorRT optimized models

## Performance Tips

- **GPU Acceleration**: Always use `--device cuda` if available
- **Batch Processing**: Process multiple frames together when possible
- **Model Selection**: Use smaller models (yolov8s) for faster inference
- **Image Size**: Smaller images (480) process faster than larger (960)
- **Feature Toggles**: Only enable features you need (tracking, re-id, clothing)

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add your contact information]

## Acknowledgments

- Ultralytics for YOLO models
- OSNet for person re-identification
- Deep OC-SORT algorithm implementation
- Streamlit for web framework

