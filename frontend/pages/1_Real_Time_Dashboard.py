import os
import re
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

try:
    import cv2
except Exception as e:
    st.error(
        "OpenCV (cv2) is required. Install it with:\n\n`pip install opencv-python`"
    )
    raise

try:
    import torch
except Exception:
    st.error("PyTorch is required. Install it with:\n\n`pip install torch`")
    raise

try:
    from ultralytics import YOLO
except Exception:
    st.error("Ultralytics is required. Install it with:\n\n`pip install ultralytics`")
    raise


# Make sure we can import from the existing `py/` modules
# pages/1_Real_Time_Dashboard.py -> frontend/ -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PY_DIR = os.path.join(BASE_DIR, "py")
if PY_DIR not in sys.path:
    sys.path.append(PY_DIR)

from PersonDetection import parse_sources, open_captures, load_model  # type: ignore
from re_id import PersonReIDManager, init_reid  # type: ignore
from tracking import init_tracker  # type: ignore
from model_manager import (  # type: ignore
    get_models_by_type,
    get_model_path,
)
from db import (  # type: ignore
    get_total_evaluations,
    get_total_violations,
    get_evaluation_status_by_gid,
    get_latest_global_id,
)
from clothing_analysis import analyze_clothing_and_log  # type: ignore
from config import SHOE_CONFIDENCE_THRESHOLD  # type: ignore


def get_model_options(model_type: str) -> Tuple[List[str], List[str]]:
    """
    Get model options for dropdown.
    
    Returns:
        Tuple of (model_names, model_paths) lists
    """
    models = get_models_by_type(model_type)
    if not models:
        return [], []
    
    names = [m["name"] for m in models]
    paths = [m["path"] for m in models]
    return names, paths


def get_default_model_path(model_type: str, default_path: str) -> str:
    """
    Get default model path, checking if it exists in uploaded models first.
    
    Args:
        model_type: Type of model (detection, reid, cloth, shoe)
        default_path: Default path to use if no models uploaded
    
    Returns:
        Model path string
    """
    models = get_models_by_type(model_type)
    if models:
        # Return the first uploaded model's path as default
        return models[0]["path"]
    return default_path


@st.cache_resource
def cached_load_model(model_path: str, device: str):
    """Cache model loading to avoid reloading on every Streamlit rerun."""
    return load_model(model_path, device=device)


@st.cache_resource
def cached_load_shoe_model(model_path: str, device: str):
    """Cache shoe model loading."""
    return YOLO(model_path)


@st.cache_data(show_spinner=False)
def probe_cameras(max_index: int = 5) -> List[int]:
    """
    Detect available camera indices up to max_index.
    Returns a list of indices that successfully open and read a frame.
    """
    available: List[int] = []
    for idx in range(max(0, max_index) + 1):
        cap = cv2.VideoCapture(idx)
        ok, _ = cap.read()
        cap.release()
        if ok:
            available.append(idx)
    return available


def persist_uploaded_videos(files: List) -> List[str]:
    """
    Persist uploaded video files to a temp folder and return their paths.
    """
    if not files:
        return []
    upload_dir = Path(BASE_DIR) / "data" / "uploaded_videos"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    for f in files:
        suffix = Path(f.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(
            delete=False, dir=upload_dir, suffix=suffix
        ) as tmp:
            tmp.write(f.getbuffer())
            saved_paths.append(tmp.name)
    return saved_paths


def build_mosaic(frames: List[np.ndarray]) -> np.ndarray:
    """Combine frames into a simple 2x2-style mosaic grid."""
    if not frames:
        raise ValueError("No frames provided for mosaic.")

    num_frames = len(frames)
    cols = int(np.ceil(np.sqrt(num_frames)))
    rows = int(np.ceil(num_frames / cols))

    base_h, base_w = frames[0].shape[:2]
    normalized_frames: List[np.ndarray] = []
    for frame in frames:
        if frame.shape[:2] != (base_h, base_w):
            normalized_frames.append(cv2.resize(frame, (base_w, base_h)))
        else:
            normalized_frames.append(frame)

    blank_tile = np.zeros_like(normalized_frames[0])
    tiles: List[np.ndarray] = []
    idx = 0
    for _ in range(rows):
        row_tiles = []
        for _ in range(cols):
            if idx < num_frames:
                row_tiles.append(normalized_frames[idx])
            else:
                row_tiles.append(blank_tile)
            idx += 1
        tiles.append(np.hstack(row_tiles))

    mosaic = np.vstack(tiles)
    return mosaic


def run_streaming_dashboard() -> None:
    st.set_page_config(
        page_title="Person Detection Dashboard",
        layout="wide",
    )

    # Custom CSS styling from reference
    st.markdown(
        """
        <style>
        .dashboard-root {
            background-color: #0f0f0f;
            padding: 1.25rem 2rem;
            color: #f5f5f5;
        }
        .metric-card {
            background: #181818;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #2a2a2a;
        }
        .metric-title {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 600;
        }
        .camera-card {
            background: #1b1b1b;
            border-radius: 12px;
            padding: 0.75rem;
            border: 1px solid #292929;
            min-height: 320px;
        }
        .camera-label {
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #f2f2f2;
        }
        .camera-img {
            border-radius: 10px;
            width: 100%;
            border: 1px solid #333;
        }
        body {
            background-color: #0f0f0f;
            color: #f5f5f5;
        }
        .stApp {
            background-color: #0f0f0f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Real-Time Monitoring Dashboard")
    st.caption("Watch multiple camera feeds and tag violations on the fly.")

    has_cuda = torch.cuda.is_available()

    with st.sidebar:
        st.subheader("Configuration")

        default_device = "cuda" if has_cuda else "cpu"

        st.markdown("### Sources")
        source_mode = st.radio(
            "Choose source type",
            options=["Camera", "Video"],
            index=0,
            horizontal=True,
            help="Use cameras detected on this machine or upload/select video files.",
        )

        # Camera mode: auto-detect and allow multi-select
        camera_max_index = st.number_input(
            "Max camera index to scan",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
        )
        if "available_cameras" not in st.session_state:
            st.session_state["available_cameras"] = probe_cameras(int(camera_max_index))
            st.session_state["selected_cameras"] = [str(i) for i in st.session_state["available_cameras"]]

        if st.button("🔄 Refresh camera list", help="Re-scan connected cameras"):
            st.session_state["available_cameras"] = probe_cameras(int(camera_max_index))
            st.session_state["selected_cameras"] = [str(i) for i in st.session_state["available_cameras"]]

        available_cams = st.session_state.get("available_cameras", [])
        camera_choices = [str(idx) for idx in available_cams]

        # Auto-select all detected cameras; remember selection across reruns
        selected_cameras = st.multiselect(
            "Select camera(s)",
            options=camera_choices,
            default=st.session_state.get("selected_cameras", camera_choices),
            help="Detected cameras you can stream from.",
            disabled=source_mode != "Camera",
        )
        st.session_state["selected_cameras"] = selected_cameras

        # Video mode: upload files (supports multiple)
        uploaded_videos = st.file_uploader(
            "Upload video file(s)",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True,
            disabled=source_mode != "Video",
        )

        selected_sources: List[str] = []
        if source_mode == "Camera":
            selected_sources = selected_cameras
            if not available_cams:
                st.warning("No cameras detected. Try refreshing or increase max index.")
        else:
            uploaded_paths = persist_uploaded_videos(uploaded_videos) if uploaded_videos else []
            selected_sources = uploaded_paths
            if not selected_sources:
                st.info("Upload a video file to start monitoring.")
        
        # Detection model dropdown
        detection_models, detection_paths = get_model_options("detection")
        if detection_models:
            default_detection_path = get_default_model_path("detection", "./model/yolov8s.pt")
            default_idx = detection_paths.index(default_detection_path) if default_detection_path in detection_paths else 0
            selected_detection = st.selectbox(
                "Detection Model",
                options=detection_models,
                index=default_idx,
                help="Select a detection model uploaded in Settings",
            )
            model_path = get_model_path(selected_detection, "detection") or "./model/yolov8s.pt"
        else:
            st.info("💡 No detection models uploaded. Go to Settings to upload models.")
            model_path = st.text_input(
                "YOLO model path or name",
                value="./model/yolov8s.pt",
                help="Or upload a model in Settings page",
            )
        conf = st.slider("Detection confidence", 0.1, 1.0, 0.75, 0.05)
        imgsz = st.selectbox("Image size", [480, 640, 736, 960], index=1)
        enable_tracking = st.checkbox("Enable DeepSort tracking (shows TID)", value=True)
        enable_reid = False
        reid_model_path = "./model/osnet_duke_reid.pth"  # Default initialization
        if enable_tracking:
            enable_reid = st.checkbox("Enable Multi-Camera Re-ID (shows TID and GID)", value=True)
        else:
            st.info("Enable tracking first to use Re-ID")
        
        # Re-ID model dropdown
        if enable_reid:
            reid_models, reid_paths = get_model_options("reid")
            if reid_models:
                default_reid_path = get_default_model_path("reid", "./model/osnet_duke_reid.pth")
                default_idx = reid_paths.index(default_reid_path) if default_reid_path in reid_paths else 0
                selected_reid = st.selectbox(
                    "Re-ID Model",
                    options=reid_models,
                    index=default_idx,
                    help="Select a Re-ID model uploaded in Settings",
                    disabled=not enable_reid,
                )
                reid_model_path = get_model_path(selected_reid, "reid") or "./model/osnet_duke_reid.pth"
            else:
                st.info("💡 No Re-ID models uploaded. Go to Settings to upload models.")
                reid_model_path = st.text_input(
                    "Re-ID model path",
                    value="./model/osnet_duke_reid.pth",
                    disabled=not enable_reid,
                    help="Or upload a model in Settings page",
                )
        reid_threshold = st.slider(
            "Re-ID similarity threshold",
            0.3,
            0.95,
            0.7,
            0.05,
            help="Higher threshold = stricter matching across cameras",
            disabled=not enable_reid,
        )
        reid_interval = st.number_input(
            "Re-ID interval (frames)",
            min_value=1,
            max_value=60,
            value=10,
            disabled=not enable_reid,
        )
        enable_cloth = False
        cloth_model_path = "./model/bestclothingmodel.pt"  # Default initialization
        shoe_model_path = "./model/shoelast.pt"  # Default initialization
        if enable_tracking:
            enable_cloth = st.checkbox(
                "Enable Cloth Detection (YOLO bestclothinmodel.pt, logs clothing attributes)", value=True
            )
        else:
            st.info("Enable tracking to use cloth detection")
        # Cloth model dropdown
        if enable_cloth:
            cloth_models, cloth_paths = get_model_options("cloth")
            if cloth_models:
                default_cloth_path = get_default_model_path("cloth", "./model/bestclothingmodel.pt")
                default_idx = cloth_paths.index(default_cloth_path) if default_cloth_path in cloth_paths else 0
                selected_cloth = st.selectbox(
                    "Cloth Detection Model",
                    options=cloth_models,
                    index=default_idx,
                    help="Select a cloth detection model uploaded in Settings",
                    disabled=not enable_cloth,
                )
                cloth_model_path = get_model_path(selected_cloth, "cloth") or "./model/bestclothingmodel.pt"
            else:
                st.info("💡 No cloth models uploaded. Go to Settings to upload models.")
                cloth_model_path = st.text_input(
                    "Cloth detection model path",
                    value="./model/bestclothingmodel.pt",
                    disabled=not enable_cloth,
                    help="Or upload a model in Settings page",
                )
            
            # Shoe model dropdown
            shoe_models, shoe_paths = get_model_options("shoe")
            if shoe_models:
                default_shoe_path = get_default_model_path("shoe", "./model/Shoebest.pt")
                default_idx = shoe_paths.index(default_shoe_path) if default_shoe_path in shoe_paths else 0
                selected_shoe = st.selectbox(
                    "Shoe Detection Model",
                    options=shoe_models,
                    index=default_idx,
                    help="Select a shoe detection model uploaded in Settings",
                    disabled=not enable_cloth,
                )
                shoe_model_path = get_model_path(selected_shoe, "shoe") or "./model/shoelast.pt"
            else:
                st.info("💡 No shoe models uploaded. Go to Settings to upload models.")
                shoe_model_path = st.text_input(
                    "Shoe detection model path",
                    value="./model/shoelast.pt",
                    disabled=not enable_cloth,
                    help="Path to shoe detection model (shoelast.pt). Or upload a model in Settings page",
                )
        cloth_conf = st.slider(
            "Cloth detection confidence",
            0.1,
            1.0,
            0.35,
            0.05,
            disabled=not enable_cloth,
        )

        shoe_conf = st.slider(
        "Shoe detection confidence",
        0.1,
        1.0,
        SHOE_CONFIDENCE_THRESHOLD,
        0.05,
        disabled=not enable_cloth,
        )

        # Cloth detection interval is set to 1 (process immediately on first detection)
        cloth_interval = 1
        cloth_min_size = st.number_input(
            "Minimum bounding box size (pixels)",
            min_value=50,
            max_value=200,
            value=70,
            help="Minimum person size for cloth detection",
            disabled=not enable_cloth,
        )
        cloth_edge_margin = st.slider(
            "Edge margin (fraction)",
            0.0,
            0.1,
            0.03,
            0.01,
            help="Margin from frame edges (0.03 = 3%%)",
            disabled=not enable_cloth,
        )
        cloth_max_retries = st.number_input(
            "Max retry attempts",
            min_value=0,
            max_value=10,
            value=3,
            help="Maximum retries for failed detections",
            disabled=not enable_cloth,
        )
        max_age = st.number_input("Tracker max age", min_value=1, max_value=120, value=10)
        min_hits = st.number_input("Tracker min hits", min_value=1, max_value=10, value=3)
        track_iou = st.slider("Tracker IoU threshold", 0.05, 0.9, 0.3, 0.05)
        similarity_lambda = st.slider(
            "Similarity blend (IoU vs Re-ID)",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Higher = more weight on IoU, lower = more weight on Re-ID embeddings",
            disabled=not enable_reid,
        )

        # Device selection with safety for environments without CUDA
        if has_cuda:
            device = st.selectbox("Device", ["cpu", "cuda"], index=0 if default_device == "cpu" else 1)
        else:
            device = "cpu"
            st.info("CUDA is not available on this system. Using CPU.")

        use_amp = st.checkbox(
            "Mixed precision (AMP, CUDA only)",
            value=(device == "cuda"),
            help="Only effective when running on CUDA. Ignored on CPU.",
        )

        start = st.button("Start Monitoring", type="primary")
        stop = st.button("Stop")

        if start:
            st.session_state["run_streams"] = True
        if stop:
            st.session_state["run_streams"] = False

        st.markdown("---")
        st.subheader("Performance Targets")
        st.markdown(
            "- **Detector**: ≤ 30 ms/frame"
        )
        
        st.markdown("---")
        if st.button("⚙️ Go to Settings", help="Manage uploaded models"):
            st.switch_page("pages/3_Settings.py")

    # Create metrics container that will be updated dynamically
    metrics_container = st.container()
    
    with metrics_container:
        col_metrics = st.columns(4)
        
        # Get initial counts from database
        total_evaluations = get_total_evaluations()
        total_violations = get_total_violations()
        
        # Create empty containers for metric values that can be updated in the loop
        metric_value_slots = []
        metric_title_slots = []
        metric_note_slots = []
        
        # Dynamic metrics - start with initial values
        sample_metrics = [
            ("Person Detected", str(total_evaluations), "Total detections"),
            ("Violations", str(total_violations), "Total violations"),
            ("Cameras Online", "0/0", "Active cameras"),
            ("Last Updated", datetime.now().strftime("%I:%M %p"), "Auto refresh"),
        ]
        for col, (title, value, note) in zip(col_metrics, sample_metrics):
            with col:
                col.markdown('<div class="metric-card">', unsafe_allow_html=True)
                title_slot = col.empty()
                value_slot = col.empty()
                note_slot = col.empty()
                title_slot.markdown(f'<div class="metric-title">{title}</div>', unsafe_allow_html=True)
                value_slot.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
                note_slot.caption(note)
                col.markdown('</div>', unsafe_allow_html=True)
                metric_title_slots.append(title_slot)
                metric_value_slots.append(value_slot)
                metric_note_slots.append(note_slot)

    # Placeholders for video
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if not st.session_state.get("run_streams", False):
        st.info("Click **Start Monitoring** in the sidebar to begin.")
        return

    # Parse and open sources
    raw_sources = selected_sources
    if not raw_sources:
        st.error("Please select at least one camera or video source.")
        return

    sources = parse_sources(raw_sources)
    caps = open_captures(sources)
    if not caps:
        st.error("No valid sources could be opened. Check camera indices/paths.")
        return

    # Safety: if user somehow selected CUDA but it is not actually available, fall back to CPU
    if device != "cpu" and not has_cuda:
        st.warning("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    # Load detection model (cached to avoid reloading on reruns)
    try:
        model = cached_load_model(model_path, device)
    except Exception as e:
        st.error(f"Failed to load model `{model_path}`: {e}")
        return
    
    # Detect TensorRT engine (for detection model)
    is_tensorrt = model_path.lower().endswith(('.plan', '.engine'))
    if is_tensorrt:
        st.info("TensorRT engine detected. Processing one frame at a time.")

    # Load clothing and shoe models if cloth detection is enabled
    clothing_model = None
    shoe_model = None
    if enable_cloth:
        try:
            clothing_model = YOLO(cloth_model_path)
            if os.path.exists(shoe_model_path):
                shoe_model = cached_load_shoe_model(shoe_model_path, device)
            else:
                st.warning(f"Shoe model not found at {shoe_model_path}. Shoe detection will be disabled.")
        except Exception as e:
            st.error(f"Failed to load cloth/shoe models: {e}")
            return

    last_timestamps = {name: time.time() for name, _ in caps}
    smoothed_fps = {name: 0.0 for name, _ in caps}
    
    # Track last evaluation time per camera for interval-based processing
    last_eval_time: Dict[str, float] = {name: 0.0 for name, _ in caps}
    # Track processed identities to avoid duplicates (per camera)
    processed_identities: Dict[str, Dict[str, bool]] = {name: {} for name, _ in caps}
    # Track frame counters per identity for interval-based processing
    identity_frame_counters: Dict[str, Dict[str, int]] = {name: {} for name, _ in caps}

    # Create a parameter hash to detect changes (include sources to detect camera changes)
    camera_names = tuple(name for name, _ in caps)
    param_hash = hash((
        enable_tracking, enable_reid, enable_cloth,
        max_age, min_hits, track_iou, similarity_lambda,
        reid_threshold, reid_interval, reid_model_path,
        cloth_model_path, shoe_model_path, cloth_conf, cloth_min_size,
        cloth_edge_margin, cloth_max_retries,
        device, camera_names
    ))
    
    # Initialize or reuse trackers from session state
    if "trackers" not in st.session_state or "param_hash" not in st.session_state or st.session_state["param_hash"] != param_hash:
        # Parameters changed or first run - recreate trackers
        st.session_state["trackers"] = {}
        st.session_state["reid_manager"] = None
        st.session_state["reid_extractor"] = None
        st.session_state["reid_frame_counters"] = {}
        st.session_state["param_hash"] = param_hash
        
        if enable_tracking:
            # Initialize Re-ID only if requested
            if enable_reid:
                try:
                    st.session_state["reid_extractor"] = init_reid(model_path=reid_model_path, device=device)
                except Exception as e:
                    st.error(f"Failed to initialize Re-ID model: {e}")
                    return
                # Get the latest GID from MongoDB to ensure persistence across restarts
                latest_gid = get_latest_global_id()
                st.session_state["reid_manager"] = PersonReIDManager(
                    similarity_threshold=reid_threshold,
                    initial_global_id=latest_gid
                )
            # Initialize tracker (with or without Re-ID embeddings)
            for window_name, _ in caps:
                st.session_state["trackers"][window_name] = init_tracker(
                    reid_extractor=st.session_state["reid_extractor"],  # None if Re-ID not enabled
                    max_age=int(max_age),
                    min_hits=int(min_hits),
                    iou_threshold=float(track_iou),
                    similarity_lambda=float(similarity_lambda),
                )
                if enable_reid:
                    st.session_state["reid_frame_counters"][window_name] = 0
    
    # Use session state trackers
    trackers = st.session_state["trackers"]
    reid_manager = st.session_state["reid_manager"]
    reid_extractor = st.session_state["reid_extractor"]
    reid_frame_counters = st.session_state["reid_frame_counters"]
    
    # Ensure trackers exist for all current cameras (in case cameras were added)
    if enable_tracking:
        for window_name, _ in caps:
            if window_name not in trackers:
                trackers[window_name] = init_tracker(
                    reid_extractor=reid_extractor,
                    max_age=int(max_age),
                    min_hits=int(min_hits),
                    iou_threshold=float(track_iou),
                    similarity_lambda=float(similarity_lambda),
                )
                if enable_reid and window_name not in reid_frame_counters:
                    reid_frame_counters[window_name] = 0
    
    # Update Re-ID manager threshold if it exists and threshold changed
    if reid_manager is not None and hasattr(reid_manager, 'similarity_threshold'):
        if reid_manager.similarity_threshold != reid_threshold:
            reid_manager.similarity_threshold = reid_threshold

    try:
        # Initialize frame counter for Re-ID cleanup
        if "reid_frame_counter" not in st.session_state:
            st.session_state["reid_frame_counter"] = 0
        
        while st.session_state.get("run_streams", False):
            frames: List[np.ndarray] = []
            owners: List[str] = []
            tracker_latency_ms = 0.0
            
            # Increment frame counter for Re-ID cleanup
            if enable_reid and reid_manager:
                reid_manager.increment_frame_count()
                st.session_state["reid_frame_counter"] += 1

            for window_name, cap in caps:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frames.append(frame)
                owners.append(window_name)

            if not frames:
                time.sleep(0.02)
                continue

            # Update dynamic metrics
            now = time.time()
            total_evaluations = get_total_evaluations()
            total_violations = get_total_violations()
            cameras_online = len([name for name, _ in caps])
            cameras_total = len(caps)
            
            if len(metric_value_slots) >= 4:
                metric_value_slots[0].markdown(f'<div class="metric-value">{total_evaluations}</div>', unsafe_allow_html=True)
                metric_value_slots[1].markdown(f'<div class="metric-value">{total_violations}</div>', unsafe_allow_html=True)
                metric_value_slots[2].markdown(f'<div class="metric-value">{cameras_online}/{cameras_total}</div>', unsafe_allow_html=True)
                metric_value_slots[3].markdown(f'<div class="metric-value">{datetime.now().strftime("%I:%M %p")}</div>', unsafe_allow_html=True)

            # Detector
            det_start = time.time()
            results: List = []
            # For TensorRT, process frames one at a time to avoid batching issues
            # For PyTorch, process all frames together
            if is_tensorrt:
                for frame in frames:
                    frame_result = model.predict(
                        frame,
                        imgsz=imgsz,
                        conf=conf,
                        classes=[0],
                        device=device,
                        half=(use_amp and device.startswith("cuda")),
                        verbose=False,
                        stream=False,
                    )
                    # model.predict returns a list of Results, get the first one
                    if isinstance(frame_result, list) and len(frame_result) > 0:
                        results.append(frame_result[0])
                    elif frame_result is not None:
                        results.append(frame_result)
            else:
                results = model.predict(
                    frames,
                    imgsz=imgsz,
                    conf=conf,
                    classes=[0],
                    device=device,
                    half=(use_amp and device.startswith("cuda")),
                    verbose=False,
                    stream=False,
                )
                # Ensure results is a list
                if not isinstance(results, list):
                    results = [results]
            det_end = time.time()
            detector_latency_ms = (det_end - det_start) * 1000.0

            display_frames: List[np.ndarray] = []

            for window_name, result, frame in zip(owners, results, frames):
                frame_to_show: np.ndarray | None = None
                run_reid_now = False

                if enable_tracking:
                    if enable_reid:
                        reid_frame_counters[window_name] += 1
                        run_reid_now = (
                            reid_frame_counters[window_name] % max(1, int(reid_interval)) == 0
                        )
                    else:
                        run_reid_now = False
                    frame_to_show = frame.copy()
                    detections_for_tracker: List[Dict[str, np.ndarray]] = []
                    if result and result.boxes is not None and len(result.boxes) > 0:
                        for b in result.boxes:
                            score = float(b.conf[0]) if b.conf is not None else 0.0
                            xyxy = b.xyxy[0].cpu().numpy().astype(np.float32)
                            detections_for_tracker.append({"bbox": xyxy, "score": score})

                    trk_start = time.time()
                    track_outputs = trackers[window_name].update(
                        detections_for_tracker, frame=frame
                    )
                    tracker_latency_ms = max(
                        tracker_latency_ms, (time.time() - trk_start) * 1000.0
                    )

                    for trk in track_outputs:
                        x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]
                        global_id = None
                        if enable_reid and reid_manager and reid_extractor:
                            global_id = reid_manager.get_existing_global_id(
                                window_name, trk["track_id"]
                            )
                            if run_reid_now:
                                embedding = reid_extractor.extract_from_frame(frame, trk["bbox"])
                                global_id = reid_manager.assign_global_id(
                                    window_name, trk["track_id"], embedding
                                )

                        # Check MongoDB for existing evaluation status by GID
                        status = "Pending"
                        status_color = (255, 255, 0)  # Yellow for pending
                        
                        # If GID exists, check MongoDB for existing evaluation status
                        if enable_cloth and global_id is not None:
                            existing_status = get_evaluation_status_by_gid(global_id)
                            if existing_status:
                                status = existing_status
                                if status == "Appropriate":
                                    status_color = (0, 255, 0)  # Green
                                elif status == "Not Appropriate":
                                    status_color = (0, 0, 255)  # Red
                        
                        # Check if person is fully visible for cloth detection
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1
                        frame_h, frame_w = frame.shape[:2]
                        is_visible = (
                            bbox_w >= cloth_min_size
                            and bbox_h >= cloth_min_size
                            and x1 >= int(frame_w * cloth_edge_margin)
                            and y1 >= int(frame_h * cloth_edge_margin)
                            and x2 <= int(frame_w * (1 - cloth_edge_margin))
                            and y2 <= int(frame_h * (1 - cloth_edge_margin))
                        )

                        # Process cloth detection with interval and deduplication
                        if enable_cloth and clothing_model is not None and is_visible:
                            identity_key = f"GID_{global_id}" if global_id is not None else f"TID_{trk['track_id']}"
                            
                            # FIRST: Check database - if GID already has an evaluation, skip processing entirely
                            already_in_db = False
                            if global_id is not None:
                                existing_status = get_evaluation_status_by_gid(global_id)
                                if existing_status is not None:
                                    # Already evaluated in database, skip processing
                                    already_in_db = True
                            
                            # SECOND: Check if already processed in this session
                            if global_id is not None:
                                # For GID: check across ALL cameras (global deduplication)
                                already_processed = any(
                                    processed_identities[cam_name].get(identity_key, False) 
                                    for cam_name in processed_identities.keys()
                                )
                            else:
                                # For TID: check only within this camera (per-camera deduplication)
                                already_processed = processed_identities[window_name].get(identity_key, False)
                            
                            # Skip if already in database OR already processed in this session
                            if not (already_in_db or already_processed):
                                # Initialize frame counter for this identity if not exists
                                if identity_key not in identity_frame_counters[window_name]:
                                    identity_frame_counters[window_name][identity_key] = 0
                                
                                identity_frame_counters[window_name][identity_key] += 1
                                
                                # Check if we should process now (interval-based, per identity)
                                should_process = (
                                    identity_frame_counters[window_name][identity_key] % cloth_interval == 0
                                )
                                
                                if should_process:
                                    # Crop person region
                                    person_crop = frame[
                                        max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)
                                    ]
                                    if person_crop.size > 0:
                                        try:
                                            # Extract camera index from window name (e.g., "Camera 0" -> 0, "footage/c0.avi" -> 0)
                                            cam_idx = 1
                                            if window_name.split()[-1].isdigit():
                                                cam_idx = int(window_name.split()[-1]) + 1
                                            elif any(char.isdigit() for char in window_name):
                                                # Try to extract number from filename
                                                numbers = re.findall(r'\d+', window_name)
                                                if numbers:
                                                    cam_idx = int(numbers[0]) + 1
                                            
                                            # Check status BEFORE calling analyze_clothing_and_log
                                            # to detect if a new evaluation was created
                                            status_before = None
                                            if global_id is not None:
                                                status_before = get_evaluation_status_by_gid(global_id)
                                            
                                            status, violation_desc, violation_type = analyze_clothing_and_log(
                                                person_crop,
                                                cam_idx=cam_idx,
                                                clothing_model=clothing_model,
                                                shoe_model=shoe_model,
                                                global_id=global_id,
                                                tracking_id=trk["track_id"],
                                                shoe_conf=shoe_conf,
                                            )

                                            # Check if evaluation was actually saved to database
                                            # If clothing_category was "Unknown", nothing was saved and we should allow retry
                                            evaluation_saved = False
                                            if global_id is not None:
                                                # Check if status changed (new evaluation was created)
                                                status_after = get_evaluation_status_by_gid(global_id)
                                                # If status_before was None and status_after is not None, new evaluation was created
                                                # If status_before was not None and status_after is different, new evaluation was created
                                                # If status_before == status_after and both are not None, no new evaluation (already existed)
                                                if status_before is None and status_after is not None:
                                                    evaluation_saved = True  # New evaluation created
                                                elif status_before is not None and status_after is not None and status_before != status_after:
                                                    evaluation_saved = True  # New evaluation created (status changed)
                                                elif status_before is not None and status_after == status_before:
                                                    evaluation_saved = False  # No new evaluation (already existed)
                                                else:
                                                    # status_before is None and status_after is None - nothing was saved (Unknown case)
                                                    evaluation_saved = False
                                            else:
                                                # For TID-only cases, we can't check database easily
                                                # Use heuristic: if violation_desc and violation_type are both empty,
                                                # it might be Unknown case, but non-violations also have empty desc/type
                                                # To be safe, we'll check: if status is "Appropriate" and both are empty,
                                                # it's likely Unknown (nothing saved). Otherwise assume saved.
                                                if status == "Appropriate" and violation_desc == "" and violation_type == "":
                                                    # Likely Unknown case - don't mark as processed to allow retry
                                                    evaluation_saved = False
                                                else:
                                                    # Assume saved for TID-only to avoid infinite retries
                                                    evaluation_saved = True
                                            
                                            # Only mark as processed if evaluation was saved to database
                                            # This allows retry for "Unknown" cases where nothing was saved
                                            if evaluation_saved:
                                                # Mark as processed: for GID, mark in ALL cameras; for TID, mark only in current camera
                                                if global_id is not None:
                                                    # Global deduplication: mark in all cameras
                                                    for cam_name in processed_identities.keys():
                                                        processed_identities[cam_name][identity_key] = True
                                                else:
                                                    # Per-camera deduplication: mark only in current camera
                                                    processed_identities[window_name][identity_key] = True
                                                last_eval_time[window_name] = now
                                            # If evaluation_saved is False, don't mark as processed - allows retry on next interval
                                            
                                            # Set status color
                                            if status == "Appropriate":
                                                status_color = (0, 255, 0)  # Green
                                            elif status == "Not Appropriate":
                                                status_color = (0, 0, 255)  # Red
                                            else:
                                                status_color = (255, 255, 0)  # Yellow
                                        except Exception as e:
                                            st.warning(f"Cloth detection error: {e}")

                        cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), status_color, 2)
                        # Build label: TID always shown when tracking, GID only when Re-ID enabled, status
                        label_parts = []
                        if enable_tracking:
                            label_parts.append(f"TID {trk['track_id']}")
                        if enable_reid and global_id is not None:
                            label_parts.append(f"GID {global_id}")
                        if enable_cloth and status != "Pending":
                            label_parts.append(status)
                        label = " | ".join(label_parts) if label_parts else ""
                        if label:
                            ((tw, th), baseline) = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )
                            frame_h = frame_to_show.shape[0]
                            label_y = min(y2 + th + baseline + 4, frame_h - 1)
                            cv2.rectangle(
                                frame_to_show,
                                (x1, y2),
                                (x1 + tw + 6, label_y),
                                status_color,
                                -1,
                            )
                            cv2.putText(
                                frame_to_show,
                                label,
                                (x1 + 3, min(y2 + th + baseline, frame_h - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )
                else:
                    frame_to_show = result.plot()

                # FPS overlay (per camera)
                dt = max(1e-6, now - last_timestamps[window_name])
                last_timestamps[window_name] = now
                alpha = 0.1
                smoothed_fps[window_name] = (1.0 / dt) * alpha + smoothed_fps[window_name] * (1 - alpha)
                fps_text = f"{window_name}: {smoothed_fps[window_name]:.1f} FPS"
                # Get text size for background rectangle
                ((text_width, text_height), baseline) = cv2.getTextSize(
                    fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                # Draw grey background rectangle
                cv2.rectangle(
                    frame_to_show,
                    (5, 5),
                    (5 + text_width + 6, 5 + text_height + baseline + 4),
                    (128, 128, 128),  # Grey color (BGR)
                    -1,  # Filled rectangle
                )
                # Draw text on top
                cv2.putText(
                    frame_to_show,
                    fps_text,
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                display_frames.append(frame_to_show)
            
            # Periodic cleanup of inactive Re-ID Global IDs
            if enable_reid and reid_manager and st.session_state["reid_frame_counter"] % reid_manager.cleanup_interval == 0:
                removed_count = reid_manager.cleanup_inactive_gids(
                    max_age=int(max_age),
                    active_global_ids=None  # Will be built from track_to_global internally
                )
                if removed_count > 0:
                    st.session_state.get("reid_cleanup_log", []).append(
                        f"Cleaned up {removed_count} inactive Global ID(s)"
                    )

            if display_frames:
                mosaic = build_mosaic(display_frames)
                mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
                # Use width='stretch' to avoid media file storage issues
                video_placeholder.image(mosaic_rgb, channels="RGB", width="stretch")

            # Performance metrics panel
            if enable_tracking or enable_reid:
                col1, col2 = metrics_placeholder.columns(2)
                col1.metric("Detector (ms)", f"{detector_latency_ms:.1f}", help="Target ≤ 30 ms")
                col2.metric("Tracker (ms)", f"{tracker_latency_ms:.1f}", help="Deep OC-SORT latency")
            else:
                metrics_placeholder.metric("Detector (ms)", f"{detector_latency_ms:.1f}", help="Target ≤ 30 ms")

            # Small sleep to avoid pegging the CPU
            time.sleep(0.01)

    finally:
        for _, cap in caps:
            try:
                cap.release()
            except Exception:
                pass


if __name__ == "__main__":
    if "run_streams" not in st.session_state:
        st.session_state["run_streams"] = False
    run_streaming_dashboard()