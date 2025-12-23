# Real-Time Multi-Camera Person Detection & Clothing Compliance System

A comprehensive computer vision system that performs real-time person detection, multi-camera tracking, and automatic clothing compliance monitoring using deep learning models.

## Features

- **Multi-Camera Person Detection**: Real-time detection using YOLOv8/YOLOv11
- **Advanced Tracking**: Deep OC-SORT algorithm for robust person tracking with Kalman filtering
- **Multi-Camera Re-Identification**: OSNet-based feature extraction for consistent identity (Global IDs) across cameras
- **Automatic Memory Management**: Re-ID feature cleanup to prevent memory leaks
- **Clothing Detection**: Automated dress code compliance monitoring with configurable violation rules
- **Real-Time Dashboard**: Streamlit web interface for live monitoring with video upload support
- **Model Management**: Upload and manage multiple detection, Re-ID, and clothing models via web interface
- **Violation Settings**: Configurable violation types with dynamic updates
- **Audit Logging**: Comprehensive violation tracking with MongoDB integration
- **Multi-Format Support**: Supports PyTorch (.pt, .pth), ONNX (.onnx), and TensorRT (.engine, .plan) models

## Prerequisites

- Python 3.8 - 3.10 (Python 3.11+ not supported due to torchreid compatibility)
- NVIDIA GPU (recommended) with CUDA support for optimal performance
- MongoDB Atlas account or local MongoDB instance
- Webcam(s) or video files for testing

## Installation

### Quick Install (Windows)

Run the `install.bat` file in Command Prompt:

```bash
install.bat
```

### Manual Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Cloth Detection"
```

#### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have an NVIDIA GPU, you may want to install PyTorch with CUDA support:

```bash
# For CUDA 11.8/12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Download Models

Place the following model files in the `model/` directory:

- **Person Detection**: `yolov8s.pt` or `yolov8m.pt` (automatically downloaded by Ultralytics)
- **Re-Identification**: `osnet_duke_reid.pth` or `osnet_reid.pth` (OSNet model)
- **Clothing Detection**: `clothing_detection.pt` or `bestclothingmodel.pt` (custom YOLO model)

Alternatively, you can upload models through the **Settings** page in the web dashboard (recommended).

You can download YOLO models automatically, but you'll need to provide the Re-ID and custom clothing detection models.

#### 5. Configure MongoDB

##### Option A: MongoDB Atlas (Cloud)

1. Create a MongoDB Atlas account at https://www.mongodb.com/cloud/atlas
2. Create a new cluster and database
3. Create two collections: `evaluation` and `violation`
4. Get your connection string

##### Option B: Set Environment Variable

Set your MongoDB connection string as an environment variable:

```bash
# Windows (PowerShell)
$env:ATLAS_URI="mongodb+srv://username:password@cluster.mongodb.net/?appName=Real-timeClothingSystem"

# Windows (CMD)
set ATLAS_URI=mongodb+srv://username:password@cluster.mongodb.net/?appName=Real-timeClothingSystem

# Linux/Mac
export ATLAS_URI="mongodb+srv://username:password@cluster.mongodb.net/?appName=Real-timeClothingSystem"
```

##### Option C: Streamlit Secrets (For Web Dashboard)

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
- `--imgsz`: Image size for inference (default: 640, options: 480, 640, 736, 960)
- `--track`: Enable Deep OC-SORT tracking (shows Tracking IDs)
- `--reid`: Enable multi-camera re-identification (requires `--track`, shows Global IDs)
- `--cloth-detect`: Enable clothing detection (requires `--track`)
- `--reid-model`: Path to Re-ID model (default: `./model/osnet_duke_reid.pth`)
- `--reid-threshold`: Cosine similarity threshold for Re-ID matching (default: 0.7)
- `--reid-interval`: Run Re-ID every N frames per camera (default: 10)
- `--cloth-model`: Path to clothing detection model (default: `./model/clothing_detection.pt`)
- `--cloth-conf`: Confidence threshold for clothing detection (default: 0.50)
- `--cloth-interval`: Process clothing detection every N frames per person (default: 30)
- `--max-age`: Maximum frames to keep lost tracks alive (default: 10)
- `--min-hits`: Minimum consecutive hits before reporting a track (default: 3)
- `--track-iou`: IoU threshold for association (default: 0.3)
- `--similarity-lambda`: Blend factor between IoU and Re-ID similarity (default: 0.5)
- `--device`: Computation device (`cpu` or `cuda`, default: auto-detect)
- `--log-metadata`: Save detection metadata to CSV files

#### Example Commands

```bash
# Basic detection with single camera
python py/main.py --sources 0

# Multi-camera with tracking and re-identification
python py/main.py --sources 0 1 --track --reid --reid-model ./model/osnet_reid.pth

# Full pipeline with clothing detection
python py/main.py --sources 0 --track --reid --cloth-detect \
    --cloth-model ./model/clothing_detection.pt

# Process video files
python py/main.py --sources footage/c0.avi footage/c1.avi --track --reid --cloth-detect

# Use GPU with custom confidence threshold
python py/main.py --sources 0 --device cuda --conf 0.8 --track --reid --cloth-detect

# Probe available cameras
python py/main.py --probe --max-index 10
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
   - Support for camera inputs and video file uploads
   - Real-time metrics (evaluations, violations, active cameras, current time)
   - Performance monitoring (latency metrics)
   - Configurable parameters via sidebar
   - Model selection dropdowns (detection, Re-ID, clothing)
   - Color-coded bounding boxes:
     - 🟢 Green: Appropriate clothing
     - 🔴 Red: Violation detected
     - 🟡 Yellow: Pending evaluation

2. **Audit Log** (`pages/2_Audit_Log.py`)
   - Filter evaluations by date range and status
   - View violation details
   - Search functionality
   - Sortable records
   - Export capabilities

3. **Settings** (`pages/3_Settings.py`)
   - **Model Management**:
     - Upload detection, Re-ID, and clothing models
     - Manage uploaded models (view, delete)
     - Support for multiple model formats (.pt, .pth, .onnx, .engine, .plan)
   - **Violation Configuration**:
     - Configure which clothing items trigger violations
     - Dynamic violation settings (takes effect immediately)
     - Reset to default settings

## Project Structure

```
Cloth Detection/
├── py/                          # Backend processing modules
│   ├── main.py                  # Main CLI entry point
│   ├── PersonDetection.py       # Person detection utilities
│   ├── tracking.py              # Deep OC-SORT tracking implementation
│   ├── re_id.py                 # Person re-identification (OSNet) with cleanup
│   ├── clothing_analysis.py     # Clothing detection and violation analysis
│   ├── db.py                    # MongoDB database operations
│   ├── config.py                # Centralized configuration parameters
│   ├── model_manager.py         # Model metadata management
│   ├── violation_settings.py    # Violation settings management
│   └── migrate_ids.py           # Database migration utilities
├── frontend/                    # Streamlit web application
│   ├── app.py                   # Main Streamlit app (landing page)
│   └── pages/
│       ├── 1_Real_Time_Dashboard.py  # Real-time monitoring
│       ├── 2_Audit_Log.py            # Audit log viewer
│       └── 3_Settings.py             # Model and violation settings
├── model/                       # Model files directory
│   ├── model_metadata.json      # Model metadata (auto-generated)
│   ├── violation_settings.json  # Violation configuration (auto-generated)
│   ├── yolov8s.pt               # Person detection model (example)
│   ├── osnet_reid.pth           # Re-identification model (example)
│   └── clothing_detection.pt    # Clothing detection model (example)
├── data/                        # Data directory
│   └── uploaded_videos/         # Temporary storage for uploaded videos
├── footage/                     # Sample video files
├── logo/                        # Application logos
├── requirements.txt             # Python dependencies
├── install.bat                  # Windows installation script
└── README.md                    # This file
```

## Configuration

### Centralized Configuration (`py/config.py`)

All key parameters are centralized in `py/config.py` for consistency:

#### Detection Configuration
- `DETECTION_CONFIDENCE`: Default confidence threshold (0.75)
- `DETECTION_IMAGE_SIZE`: Default inference image size (640)

#### Tracking Parameters
- `TRACK_MAX_AGE`: Maximum frames to keep lost tracks alive (default: 10)
- `TRACK_MIN_HITS`: Minimum consecutive hits before reporting a track (default: 3)
- `TRACK_IOU_THRESHOLD`: IoU threshold for association (default: 0.3)
- `SIMILARITY_LAMBDA`: Blend factor between IoU and Re-ID similarity (default: 0.5)

#### Re-Identification Parameters
- `REID_THRESHOLD`: Cosine similarity threshold for matching (default: 0.7)
- `REID_INTERVAL`: Run Re-ID every N frames per camera (default: 30 in Streamlit, 10 in CLI)

#### Clothing Detection Parameters
- `CLOTH_CONFIDENCE_THRESHOLD`: Confidence threshold for clothing detection (default: 0.50)
- `CLOTH_INTERVAL`: Process every N frames per person (default: 30)
- `CLOTH_MIN_SIZE`: Minimum bounding box size in pixels (default: 70)
- `CLOTH_EDGE_MARGIN`: Edge margin as fraction of frame (default: 0.03 = 3%)
- `MAX_CLOTHING_ITEMS`: Maximum clothing items to report (default: 2)

#### Violation Keywords
- Configurable via Settings page or `violation_settings.json`
- Default banned items: shorts, skirt, flipflops, sandals, vest, sling_dress, sling

### Re-ID Memory Management

The system automatically cleans up inactive Global IDs to prevent memory leaks:

- **Cleanup Interval**: Every 300 frames (configurable)
- **Cleanup Threshold**: `max_age * max_age_multiplier` (default: 10 * 10 = 100 frames)
- **Active Protection**: Global IDs currently associated with active tracks are never removed

## Workflow

### Person Detection → Tracking → Re-ID → Clothing Analysis

1. **Detection Phase**: YOLO model detects persons in video frames
2. **Tracking Phase**: Deep OC-SORT assigns Tracking IDs (TID) and tracks persons within each camera
3. **Re-ID Phase** (if enabled): 
   - OSNet extracts feature vectors from person crops
   - PersonReIDManager assigns Global IDs (GID) across cameras based on feature similarity
   - Features stored in gallery for cross-camera matching
   - Periodic cleanup removes inactive Global IDs
4. **Clothing Analysis Phase** (if enabled):
   - Person crops analyzed by clothing detection model
   - Complete coverage validation (top+bottom OR dress)
   - Violation detection based on configurable banned keywords
   - Results saved to MongoDB with status (Appropriate/Not Appropriate/Pending)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce image size: `--imgsz 480`
   - Process fewer cameras simultaneously
   - Use smaller YOLO model: `--model ./model/yolov8s.pt`
   - Reduce batch processing

2. **Camera Not Detected**
   - Use `--probe` to list available cameras: `python py/main.py --probe`
   - Check camera permissions
   - Try different camera indices: `--sources 0 1 2`
   - In Streamlit dashboard, click "🔄 Refresh camera list"

3. **MongoDB Connection Error**
   - Verify your connection string in environment variable or `.streamlit/secrets.toml`
   - Check network connectivity
   - Ensure MongoDB Atlas IP whitelist includes your IP (if using Atlas)
   - Verify database and collection names are correct

4. **Model Not Found**
   - Verify model files exist in `model/` directory
   - Check file paths in command arguments
   - YOLO models will auto-download if not found
   - Use Settings page to upload models via web interface

5. **Low FPS Performance**
   - Enable GPU: `--device cuda`
   - Reduce image size: `--imgsz 480`
   - Disable unnecessary features (Re-ID, clothing detection)
   - Use TensorRT optimized models (.engine, .plan)
   - Increase Re-ID and clothing detection intervals

6. **torchreid Installation Issues**
   - Ensure Python version is 3.8 - 3.10 (not 3.11+)
   - Install torchreid separately: `pip install torchreid`
   - Check compatibility with your PyTorch version

7. **Re-ID Memory Leak (High RAM Usage)**
   - System automatically cleans up inactive Global IDs every 300 frames
   - Verify cleanup is running (check logs)
   - Reduce `max_age_multiplier` if needed (in `re_id.py`)

## Performance Tips

- **GPU Acceleration**: Always use `--device cuda` if available
- **Batch Processing**: Process multiple frames together when possible (PyTorch models)
- **Model Selection**: Use smaller models (yolov8s) for faster inference
- **Image Size**: Smaller images (480) process faster than larger (960)
- **Feature Toggles**: Only enable features you need (tracking, re-id, clothing)
- **Interval Tuning**: Increase Re-ID and clothing detection intervals for better performance
- **TensorRT Optimization**: Convert models to TensorRT format for maximum performance

## Model Formats Supported

- **PyTorch**: `.pt`, `.pth` (standard YOLO and OSNet models)
- **ONNX**: `.onnx` (cross-platform inference)
- **TensorRT**: `.engine`, `.plan` (optimized for NVIDIA GPUs)

Models can be uploaded and managed via the Settings page in the web dashboard.

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add your contact information]

## Acknowledgments

- **Ultralytics** for YOLO models
- **OSNet** for person re-identification
- **Deep OC-SORT** algorithm implementation
- **Streamlit** for web framework
- **MongoDB** for database solution
