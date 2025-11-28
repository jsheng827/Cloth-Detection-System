import os
import re
import sys
import time
from datetime import datetime
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
from db import (  # type: ignore
    save_evaluation,
    save_violation,
    get_next_evaluation_and_cloth_ids,
    get_next_violation_id,
    get_total_evaluations,
    get_total_violations,
    get_evaluation_status_by_gid,
    get_latest_global_id,
)

# Parameters for shoe detection filtering
SHOE_CONFIDENCE_THRESHOLD = 0.55  # confidence required to consider a shoe detection


@st.cache_resource
def cached_load_model(model_path: str, device: str):
    """Cache model loading to avoid reloading on every Streamlit rerun."""
    return load_model(model_path, device=device)


@st.cache_resource
def cached_load_shoe_model(model_path: str, device: str):
    """Cache shoe model loading."""
    return YOLO(model_path)


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


def analyze_clothing_and_log(
    person_crop: np.ndarray,
    cam_idx: int,
    clothing_model: YOLO,
    shoe_model: Optional[YOLO],
    global_id: Optional[int] = None,
    tracking_id: Optional[int] = None,
) -> Tuple[str, str, str]:
    """
    Run cloth and shoe models on a cropped person image, decide status, save into MongoDB,
    and return (status, description, violation_type).
    """
    cloth_results = clothing_model(person_crop, verbose=False)[0]

    # collect all classes detected on this person (clothing) with confidence scores
    labels = []
    label_confidences = {}  # Store label -> confidence mapping
    for det in cloth_results.boxes:
        cls_id = int(det.cls)
        label = cloth_results.names.get(cls_id, "cloth")
        confidence = float(det.conf)  # Get confidence score
        labels.append(label)
        # Store highest confidence if label appears multiple times
        if label not in label_confidences or confidence > label_confidences[label]:
            label_confidences[label] = confidence

    # run shoe detection model on the lower half of the person crop (no tracking)
    if shoe_model is not None:
        h, w = person_crop.shape[:2]
        shoe_roi = person_crop[int(h * 0.5) :, :]

        shoe_results = shoe_model(
            shoe_roi,
            conf=SHOE_CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        if shoe_results and shoe_results.boxes is not None:
            for det in shoe_results.boxes:
                cls_id = int(det.cls)
                label = shoe_results.names.get(cls_id, "shoe")
                confidence = float(det.conf)
                labels.append(label)
                if label not in label_confidences or confidence > label_confidences[label]:
                    label_confidences[label] = confidence

    if not labels:
        clothing_category = "Unknown"
    else:
        clothing_category = ", ".join(sorted(set(labels)))

    # very simple rule: mark some classes as violation
    # Only match exact words, not substrings (e.g., "short" should match "shorts" or "short" but not "short_sleeve_top")
    banned_keywords = ["short", "skirt", "crop", "flipflops", "sandals", "vest"]
    
    def is_banned(label_lower):
        """Check if label contains any banned keyword as a whole word (not substring)."""
        # Check exact match first
        if label_lower in banned_keywords:
            return True
        
        # Handle plural forms (e.g., "shorts" should match banned "short")
        if label_lower == "shorts" and "short" in banned_keywords:
            return True
        
        # Split by underscores to get individual words
        words = label_lower.split("_")
        # Check if any word exactly matches a banned keyword
        # But exclude cases where "short" is part of "short_sleeve" (it's a modifier, not the item)
        for word in words:
            if word in banned_keywords:
                # Special case: "short" in "short_sleeve_top" should NOT match
                # Only match if "short" is the main item (like "short_pants" or standalone)
                if word == "short" and "sleeve" in words:
                    continue  # Skip "short" when it's part of "short_sleeve"
                return True
        
        return False
    
    # Find which specific label(s) triggered the violation
    banned_labels = [label for label in labels if is_banned(label.lower())]
    is_violation = len(banned_labels) > 0

    status = "Not Appropriate" if is_violation else "Appropriate"

    # Incremental IDs based on existing records in MongoDB
    eval_id, cloth_id = get_next_evaluation_and_cloth_ids()
    details = "..." if is_violation else "-"

    evaluation_doc = save_evaluation(
        evaluation_id=eval_id,
        clothing_detection_id=cloth_id,
        clothing_category=clothing_category,
        status=status,
        details=details,
        global_id=global_id,
        tracking_id=tracking_id,
    )

    violation_desc = ""
    violation_type = ""
    if is_violation:
        # Get all banned labels with their confidence scores (remove duplicates, keep unique)
        unique_banned_labels = []
        seen = set()
        for label in banned_labels:
            if label not in seen:
                unique_banned_labels.append(label)
                seen.add(label)
        
        # Build violation type string (comma-separated if multiple)
        violation_type = ", ".join(unique_banned_labels) if unique_banned_labels else "unknown"
        
        # Build description with all violation types and their confidence scores
        violation_parts = []
        for label in unique_banned_labels:
            confidence = label_confidences.get(label, 0.0)
            confidence_pct = f"{confidence * 100:.1f}%"
            violation_parts.append(f"{label}({confidence_pct})")
        
        # Format: "Detected {violation_type}(Confidence score), and {violation_type} (confidence score) on camera {cam_idx}"
        if len(violation_parts) == 1:
            violation_desc = f"Detected {violation_parts[0]} on camera {cam_idx}"
        elif len(violation_parts) == 2:
            violation_desc = f"Detected {violation_parts[0]}, and {violation_parts[1]} on camera {cam_idx}"
        else:
            # For 3+ violations: "Detected X, Y, and Z on camera N"
            all_but_last = ", ".join(violation_parts[:-1])
            violation_desc = f"Detected {all_but_last}, and {violation_parts[-1]} on camera {cam_idx}"
        
        vio_id = get_next_violation_id()
        save_violation(
            violation_id=vio_id,
            evaluation_id=evaluation_doc["evaluation_id"],
            evaluation_status=status,
            violation_type=violation_type,
            description=violation_desc,
            image=person_crop,
        )

    return status, violation_desc, violation_type


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

        sources_str = st.text_input(
            "Sources (space-separated indices/paths)",
            value="0",
            help="Example: `0` or `0 1` or `footage/c0.avi footage/c1.avi`",
        )
        model_path = st.text_input(
            "YOLO model path or name",
            value="./model/yolov8s.pt",
        )
        conf = st.slider("Detection confidence", 0.1, 1.0, 0.75, 0.05)
        imgsz = st.selectbox("Image size", [480, 640, 736, 960], index=1)
        enable_tracking = st.checkbox("Enable DeepSort tracking (shows TID)", value=True)
        enable_reid = False
        if enable_tracking:
            enable_reid = st.checkbox("Enable Multi-Camera Re-ID (shows TID and GID)", value=True)
        else:
            st.info("Enable tracking first to use Re-ID")
        
        reid_model_path = st.text_input(
            "Re-ID model path",
            value="./model/osnet_duke_reid.pth",
            disabled=not enable_reid,
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
        if enable_tracking:
            enable_cloth = st.checkbox(
                "Enable Cloth Detection (YOLO best.pt, logs clothing attributes)", value=True
            )
        else:
            st.info("Enable tracking to use cloth detection")
        cloth_model_path = st.text_input(
            "Cloth detection model path",
            value="./model/best.pt",
            disabled=not enable_cloth,
        )
        shoe_model_path = st.text_input(
            "Shoe detection model path",
            value="./model/shoelast.pt",
            disabled=not enable_cloth,
            help="Path to shoe detection model (shoelast.pt)",
        )
        cloth_conf = st.slider(
            "Cloth detection confidence",
            0.1,
            1.0,
            0.25,
            0.05,
            disabled=not enable_cloth,
        )
        # Cloth detection interval is set to 1 (process immediately on first detection)
        cloth_interval = 1
        cloth_min_size = st.number_input(
            "Minimum bounding box size (pixels)",
            min_value=50,
            max_value=200,
            value=60,
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
        max_age = st.number_input("Tracker max age", min_value=1, max_value=120, value=30)
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
    raw_sources = sources_str.split()
    if not raw_sources:
        st.error("Please provide at least one source.")
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
        while st.session_state.get("run_streams", False):
            frames: List[np.ndarray] = []
            owners: List[str] = []
            tracker_latency_ms = 0.0

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
                            
                            # Check if already processed - if yes, skip all interval/visibility checks
                            already_processed = processed_identities[window_name].get(identity_key, False)
                            
                            if not already_processed:
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
                                            
                                            status, violation_desc, violation_type = analyze_clothing_and_log(
                                                person_crop,
                                                cam_idx=cam_idx,
                                                clothing_model=clothing_model,
                                                shoe_model=shoe_model,
                                                global_id=global_id,
                                                tracking_id=trk["track_id"],
                                            )
                                            processed_identities[window_name][identity_key] = True
                                            last_eval_time[window_name] = now
                                            
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
