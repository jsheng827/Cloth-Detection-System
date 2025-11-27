import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PY_DIR = os.path.join(BASE_DIR, "py")
if PY_DIR not in sys.path:
    sys.path.append(PY_DIR)

from PersonDetection import parse_sources, open_captures, load_model  # type: ignore
from cloth_detection import ClothDetectionService  # type: ignore
from re_id import PersonReIDManager, init_reid  # type: ignore
from tracking import init_tracker  # type: ignore


@st.cache_resource
def cached_load_model(model_path: str, device: str):
    """Cache model loading to avoid reloading on every Streamlit rerun."""
    return load_model(model_path, device=device)


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

    # Simple dark styling to roughly match guidelines
    st.markdown(
        """
        <style>
        body {
            background-color: #1A1A1A;
            color: #E0E0E0;
        }
        .stApp {
            background-color: #1A1A1A;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Person Detection System – Multi-Camera Dashboard")

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
            enable_reid = st.checkbox("Enable Multi-Camera Re-ID (shows TID and GID)", value=False)
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
                "Enable Cloth Detection (YOLO best.pt, logs clothing attributes)", value=False
            )
        else:
            st.info("Enable tracking to use cloth detection")
        cloth_model_path = st.text_input(
            "Cloth detection model path",
            value="./model/best.pt",
            disabled=not enable_cloth,
        )
        cloth_conf = st.slider(
            "Cloth detection confidence",
            0.1,
            1.0,
            0.25,
            0.05,
            disabled=not enable_cloth,
        )
        cloth_interval = st.number_input(
            "Cloth detection interval (frames)",
            min_value=1,
            max_value=120,
            value=30,
            help="Process cloth detection every N frames per person",
            disabled=not enable_cloth,
        )
        cloth_min_size = st.number_input(
            "Minimum bounding box size (pixels)",
            min_value=50,
            max_value=200,
            value=80,
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
        log_metadata = st.checkbox(
            "Log metadata CSV",
            value=False,
            help="Saves detection metadata to data/metadata_output/ and shows recent rows.",
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

    # Placeholders for video and metrics
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    metadata_placeholder = st.empty()
    cloth_placeholder = st.empty()

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

    last_timestamps = {name: time.time() for name, _ in caps}
    smoothed_fps = {name: 0.0 for name, _ in caps}

    # Metadata logging (aligned with CLI CSV format)
    csv_writer = None
    csv_file = None
    metadata_recent: List[Dict[str, str]] = []
    frame_seq_nums: Dict[str, int] = {name: 0 for name, _ in caps}
    if log_metadata:
        metadata_dir = os.path.join("data", "metadata_output")
        os.makedirs(metadata_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(metadata_dir, f"{ts_str}_metadata.csv")
        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "timestamp",
                "camera_name",
                "frame_sequence_num",
                "track_id",
                "global_id",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "confidence_score",
            ]
        )

    # Create a parameter hash to detect changes (include sources to detect camera changes)
    camera_names = tuple(name for name, _ in caps)
    param_hash = hash((
        enable_tracking, enable_reid, enable_cloth,
        max_age, min_hits, track_iou, similarity_lambda,
        reid_threshold, reid_interval, reid_model_path,
        cloth_model_path, cloth_conf, cloth_interval, cloth_min_size,
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
        st.session_state["cloth_service"] = None
        st.session_state["param_hash"] = param_hash
        
        if enable_tracking:
            # Initialize Re-ID only if requested
            if enable_reid:
                try:
                    st.session_state["reid_extractor"] = init_reid(model_path=reid_model_path, device=device)
                except Exception as e:
                    st.error(f"Failed to initialize Re-ID model: {e}")
                    return
                st.session_state["reid_manager"] = PersonReIDManager(similarity_threshold=reid_threshold)
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
            if enable_cloth:
                try:
                    st.session_state["cloth_service"] = ClothDetectionService(
                        model_path=cloth_model_path,
                        conf=cloth_conf,
                        process_interval=int(cloth_interval),
                        min_size=int(cloth_min_size),
                        edge_margin=float(cloth_edge_margin),
                        max_retries=int(cloth_max_retries),
                    )
                except Exception as e:
                    st.error(f"Failed to initialize cloth detection model: {e}")
                    return
        else:
            st.session_state["cloth_service"] = None
    
    # Use session state trackers
    trackers = st.session_state["trackers"]
    reid_manager = st.session_state["reid_manager"]
    reid_extractor = st.session_state["reid_extractor"]
    reid_frame_counters = st.session_state["reid_frame_counters"]
    cloth_service = st.session_state.get("cloth_service")
    
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

                        cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Build label: TID always shown when tracking, GID only when Re-ID enabled
                        label_parts = []
                        if enable_tracking:
                            label_parts.append(f"TID {trk['track_id']}")
                        if enable_reid and global_id is not None:
                            label_parts.append(f"GID {global_id}")
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
                                (0, 255, 0),
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

                        if enable_cloth and cloth_service is not None:
                            cloth_service.process(
                                tracking_id=trk["track_id"],
                                frame=frame,
                                bbox=trk["bbox"],
                                global_id=global_id,
                            )

                        if csv_writer is not None:
                            frame_seq_nums[window_name] += 1
                            timestamp = datetime.now().isoformat()
                            csv_writer.writerow(
                                [
                                    timestamp,
                                    window_name,
                                    frame_seq_nums[window_name],
                                    trk["track_id"],
                                    global_id if global_id is not None else "",
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    f"{trk['score']:.3f}",
                                ]
                            )
                            metadata_recent.append(
                                {
                                    "timestamp": timestamp,
                                    "camera": window_name,
                                    "track_id": trk["track_id"],
                                    "global_id": global_id if global_id is not None else "",
                                    "confidence": f"{trk['score']:.2f}",
                                }
                            )
                            metadata_recent = metadata_recent[-100:]
                else:
                    frame_to_show = result.plot()

                    # Optional metadata logging
                    if csv_writer is not None and result and result.boxes is not None and len(result.boxes) > 0:
                        for b in result.boxes:
                            score = float(b.conf[0]) if b.conf is not None else 0.0
                            xyxy = b.xyxy[0].tolist()
                            x1, y1, x2, y2 = [int(v) for v in xyxy]
                            frame_seq_nums[window_name] += 1
                            timestamp = datetime.now().isoformat()
                            csv_writer.writerow(
                                [
                                    timestamp,
                                    window_name,
                                    frame_seq_nums[window_name],
                                    "",
                                    "",
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    f"{score:.3f}",
                                ]
                            )
                            metadata_recent.append(
                                {
                                    "timestamp": timestamp,
                                    "camera": window_name,
                                    "track_id": "",
                                    "global_id": "",
                                    "confidence": f"{score:.2f}",
                                }
                            )
                            metadata_recent = metadata_recent[-100:]

                # FPS overlay (per camera)
                now = time.time()
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

            if csv_writer is not None:
                if metadata_recent:
                    recent_df = pd.DataFrame(metadata_recent[-20:][::-1])
                    metadata_placeholder.dataframe(
                        recent_df,
                        width="stretch",
                    )
                else:
                    metadata_placeholder.info("Metadata logging enabled. Waiting for detections...")
            else:
                metadata_placeholder.empty()

            if enable_cloth and cloth_service is not None:
                try:
                    cloth_df = pd.read_csv(cloth_service.log_path)
                    if not cloth_df.empty:
                        cloth_placeholder.dataframe(
                            cloth_df.tail(20),
                            width="stretch",
                        )
                    else:
                        cloth_placeholder.info("Cloth detection enabled. Waiting for detections...")
                except FileNotFoundError:
                    cloth_placeholder.info("Cloth detection log not created yet.")
            else:
                cloth_placeholder.empty()

            # Small sleep to avoid pegging the CPU
            time.sleep(0.01)

    finally:
        for _, cap in caps:
            try:
                cap.release()
            except Exception:
                pass
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    if "run_streams" not in st.session_state:
        st.session_state["run_streams"] = False
    run_streaming_dashboard()


