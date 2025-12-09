import argparse
import csv
import math
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except Exception:
    print("OpenCV (cv2) is required. Install with: pip install opencv-python", file=sys.stderr)
    raise

try:
    import torch
except Exception:
    print("PyTorch is required. Install with: pip install torch", file=sys.stderr)
    raise

try:
    from ultralytics import YOLO
except Exception:
    print("Ultralytics is required. Install with: pip install ultralytics", file=sys.stderr)
    raise

try:
    import numpy as np
except Exception:
    print("NumPy is required. Install with: pip install numpy", file=sys.stderr)
    raise

from PersonDetection import parse_sources, open_captures, load_model
from re_id import PersonReIDManager, init_reid
from tracking import init_tracker
from db import (
    get_evaluation_status_by_gid,
    get_latest_global_id,
)
from clothing_analysis import analyze_clothing_and_log
from config import (
    CLOTH_CONFIDENCE_THRESHOLD,
    CLOTH_EDGE_MARGIN,
    CLOTH_INTERVAL,
    CLOTH_MAX_RETRIES,
    CLOTH_MIN_SIZE,
)
from typing import Optional


def build_mosaic(frames: List[np.ndarray], cols: int | None = None) -> np.ndarray:
    """Combine frames into a mosaic grid."""
    if not frames:
        raise ValueError("No frames provided for mosaic.")

    num_frames = len(frames)
    if cols is None or cols <= 0:
        cols = math.ceil(math.sqrt(num_frames))
    rows = math.ceil(num_frames / cols)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time person detection on multiple sources "
        "with optional Deep OC-SORT tracking and metadata logging"
    )

    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument(
        "--sources",
        nargs="+",
        default=["0"],
        help="List of camera indices, file paths, or RTSP/HTTP streams (default: 0)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Probe and list available camera indices, then exit",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=5,
        help="Maximum camera index to probe (used with --probe)",
    )
    parser.add_argument(
        "--model",
        default="./model/yolov8s.pt",
        help="Path to YOLO model (.pt). Default: ./model/yolov8s.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.75,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (square). Typical values: 640, 736, 960",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay approximate FPS in window title",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable Deep OC-SORT tracking (detection-based, shows TID)",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable Multi-Camera Re-ID (requires --track, shows TID and GID)",
    )
    parser.add_argument(
        "--cloth-detect",
        action="store_true",
        help="Enable cloth detection from person crops (requires --track)",
    )
    parser.add_argument(
        "--log-metadata",
        action="store_true",
        help=(
            "Log standardized detection metadata to a CSV file under "
            "metadata_output/ (aligned with project documentation)"
        ),
    )
    parser.add_argument(
        "--reid-model",
        type=str,
        default="./model/osnet_duke_reid.pth",
        help="Path to OSNet Re-ID checkpoint (default: ./model/osnet_duke_reid.pth)",
    )
    parser.add_argument(
        "--cloth-model",
        type=str,
        default="./model/clothing_detection.pt",
        help="Path to cloth detection YOLO checkpoint (default: ./model/clothing_detection.pt)",
    )
    parser.add_argument(
        "--shoe-model",
        type=str,
        default="./model/Shoebest.pt",
        help="Path to shoe detection YOLO checkpoint (default: ./model/Shoebest.pt)",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Maximum frames to keep lost tracks alive (Deep OC-SORT)",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum consecutive hits before reporting a track",
    )
    parser.add_argument(
        "--track-iou",
        type=float,
        default=0.3,
        help="IoU threshold used for association in Deep OC-SORT",
    )
    parser.add_argument(
        "--similarity-lambda",
        type=float,
        default=0.5,
        help="Blend factor between IoU and Re-ID similarity (0-1). Higher favors IoU.",
    )
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for assigning the same Global ID across cameras.",
    )
    parser.add_argument(
        "--reid-interval",
        type=int,
        default=10,
        help="Run Re-ID every N frames per camera (default: 10).",
    )
    parser.add_argument(
        "--cloth-conf",
        type=float,
        default=CLOTH_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for cloth detection model.",
    )
    parser.add_argument(
        "--cloth-interval",
        type=int,
        default=CLOTH_INTERVAL,
        help="Process cloth detection every N frames per person (default: 30).",
    )
    parser.add_argument(
        "--cloth-min-size",
        type=int,
        default=CLOTH_MIN_SIZE,
        help="Minimum bounding box size (pixels) for cloth detection (default: 70).",
    )
    parser.add_argument(
        "--cloth-edge-margin",
        type=float,
        default=CLOTH_EDGE_MARGIN,
        help="Edge margin as fraction of frame dimension (default: 0.03 = 3%%).",
    )
    parser.add_argument(
        "--cloth-max-retries",
        type=int,
        default=CLOTH_MAX_RETRIES,
        help="Maximum retry attempts for failed cloth detections (default: 3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Computation device for inference (e.g., cpu, cuda, cuda:0). Default uses GPU if available.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (FP16) inference. Defaults to enabled when using CUDA.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    use_amp = args.amp or (args.device.startswith("cuda") and torch.cuda.is_available())

    if args.probe:
        available = []
        for idx in range(max(0, args.max_index + 1)):
            cap = cv2.VideoCapture(idx)
            ok, _ = cap.read()
            cap.release()
            if ok:
                available.append(idx)
        if available:
            print("Available camera indices:", " ".join(str(i) for i in available))
        else:
            print("No cameras detected in range 0..", args.max_index)
        return

    try:
        model = load_model(args.model, device=args.device)
    except Exception as e:
        print(f"Failed to load model '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    sources = parse_sources(args.sources)
    caps = open_captures(sources)
    if not caps:
        print("No valid sources could be opened.", file=sys.stderr)
        sys.exit(2)

    # Detect if using TensorRT engine (for detection model)
    is_tensorrt = args.model.lower().endswith(('.plan', '.engine'))

    cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)

    # Optional CSV metadata logger (doc-aligned: /data/metadata_output/...)
    csv_writer = None
    csv_file = None
    if args.log_metadata:
        # Create output directory similar to documented structure
        metadata_dir = os.path.join("data", "metadata_output")
        os.makedirs(metadata_dir, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(metadata_dir, f"{ts_str}_metadata.csv")
        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        # Header aligns with documentation (simplified – can be extended later)
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

    try:
        import time

        last_timestamps = {name: time.time() for name, _ in caps}
        smoothed_fps = {name: 0.0 for name, _ in caps}

        # Per-camera frame sequence counters (for metadata logging)
        frame_seq_nums: Dict[str, int] = {name: 0 for name, _ in caps}

        # Tracker + Re-ID setup
        trackers: Dict[str, object] = {}
        reid_manager: PersonReIDManager | None = None
        reid_extractor = None
        reid_frame_counters: Dict[str, int] = {}
        if args.reid and not args.track:
            print("Error: --reid requires --track to be enabled", file=sys.stderr)
            sys.exit(3)
        if args.cloth_detect and not args.track:
            print("Error: --cloth-detect requires --track to be enabled", file=sys.stderr)
            sys.exit(4)
        
        # Cloth detection models (MongoDB integration)
        clothing_model: Optional[YOLO] = None
        shoe_model: Optional[YOLO] = None
        if args.cloth_detect:
            try:
                clothing_model = YOLO(args.cloth_model)
                if os.path.exists(args.shoe_model):
                    shoe_model = YOLO(args.shoe_model)
                else:
                    print(f"Warning: Shoe model not found at {args.shoe_model}. Shoe detection will be disabled.", file=sys.stderr)
            except Exception as e:
                print(f"Failed to initialize cloth/shoe detection models: {e}", file=sys.stderr)
                sys.exit(5)
        
        # Track processed identities and frame counters for cloth detection
        processed_identities: Dict[str, Dict[str, bool]] = {name: {} for name, _ in caps}
        identity_frame_counters: Dict[str, Dict[str, int]] = {name: {} for name, _ in caps}
        
        if args.track:
            # Initialize Re-ID only if requested
            if args.reid:
                try:
                    reid_extractor = init_reid(model_path=args.reid_model, device=args.device)
                except Exception as e:
                    print(f"Failed to initialize Re-ID extractor: {e}", file=sys.stderr)
                    sys.exit(3)
                # Get the latest GID from MongoDB to ensure persistence across restarts
                latest_gid = get_latest_global_id()
                print(f"[ReID] Starting from GID {latest_gid + 1} (latest in DB: {latest_gid})")
                reid_manager = PersonReIDManager(
                    similarity_threshold=args.reid_threshold,
                    initial_global_id=latest_gid
                )
            # Initialize tracker (with or without Re-ID embeddings)
            for window_name, _ in caps:
                trackers[window_name] = init_tracker(
                    reid_extractor=reid_extractor if args.reid else None,  # None if Re-ID not enabled
                    max_age=args.max_age,
                    min_hits=args.min_hits,
                    iou_threshold=args.track_iou,
                    similarity_lambda=args.similarity_lambda,
                )
                if args.reid:
                    reid_frame_counters[window_name] = 0

        while True:
            frames: List = []
            owners: List[str] = []

            for window_name, cap in caps:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frames.append(frame)
                owners.append(window_name)

            if not frames:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            # Detector timing (basic latency measurement)
            det_start = time.time()
            results: List = []
            # For TensorRT, process frames one at a time to avoid batching alignment issues
            # For PyTorch, process all frames together
            if is_tensorrt:
                for frame in frames:
                    frame_result = model.predict(
                        frame,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        classes=[0],
                        device=args.device,
                        half=use_amp,
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
                    imgsz=args.imgsz,
                    conf=args.conf,
                    classes=[0],
                    device=args.device,
                    half=use_amp,
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
                if args.track:
                    if args.reid:
                        reid_frame_counters[window_name] += 1
                        run_reid_now = (
                            reid_frame_counters[window_name] % max(1, args.reid_interval) == 0
                        )
                    else:
                        run_reid_now = False
                    frame_to_show = frame.copy()
                    detections_for_tracker: List[Dict[str, np.ndarray]] = []
                    if result and result.boxes is not None and len(result.boxes) > 0:
                        for b in result.boxes:
                            score = float(b.conf[0]) if b.conf is not None else 0.0
                            xyxy = b.xyxy[0].cpu().numpy().astype(np.float32)
                            detections_for_tracker.append(
                                {
                                    "bbox": xyxy,
                                    "score": score,
                                }
                            )

                    track_outputs = trackers[window_name].update(detections_for_tracker, frame=frame)
                    for trk in track_outputs:
                        x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]
                        global_id = None
                        if args.reid and reid_manager and reid_extractor:
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
                        if args.cloth_detect and global_id is not None:
                            existing_status = get_evaluation_status_by_gid(global_id)
                            if existing_status:
                                status = existing_status
                                if status == "Appropriate":
                                    status_color = (0, 255, 0)  # Green
                                elif status == "Not Appropriate":
                                    status_color = (0, 0, 255)  # Red
                        
                        # Build label: TID always shown when tracking, GID only when Re-ID enabled
                        label_parts = []
                        if args.track:
                            label_parts.append(f"TID {trk['track_id']}")
                        if args.reid and global_id is not None:
                            label_parts.append(f"GID {global_id}")

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
                                    trk["score"],
                                ]
                            )
                        # Cloth detection with MongoDB integration
                        if args.cloth_detect and clothing_model is not None:
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
                                    identity_frame_counters[window_name][identity_key] % args.cloth_interval == 0
                                )
                                
                                # Check if person is fully visible
                                bbox_w = x2 - x1
                                bbox_h = y2 - y1
                                frame_h, frame_w = frame.shape[:2]
                                is_visible = (
                                    bbox_w >= args.CLOTH_MIN_SIZE
                                    and bbox_h >= args.CLOTH_MIN_SIZE
                                    and x1 >= int(frame_w * args.cloth_edge_margin)
                                    and y1 >= int(frame_h * args.cloth_edge_margin)
                                    and x2 <= int(frame_w * (1 - args.cloth_edge_margin))
                                    and y2 <= int(frame_h * (1 - args.cloth_edge_margin))
                                )
                                
                                if should_process and is_visible:
                                    # Crop person region
                                    person_crop = frame[
                                        max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)
                                    ]
                                    if person_crop.size > 0:
                                        try:
                                            # Extract camera index from window name
                                            cam_idx = 1
                                            if window_name.split()[-1].isdigit():
                                                cam_idx = int(window_name.split()[-1]) + 1
                                            elif any(char.isdigit() for char in window_name):
                                                numbers = re.findall(r'\d+', window_name)
                                                if numbers:
                                                    cam_idx = int(numbers[0]) + 1
                                            
                                            # Analyze clothing and save to MongoDB
                                            status, violation_desc, violation_type = analyze_clothing_and_log(
                                                person_crop,
                                                cam_idx=cam_idx,
                                                clothing_model=clothing_model,
                                                shoe_model=shoe_model,
                                                global_id=global_id,
                                                tracking_id=trk["track_id"],
                                            )
                                            processed_identities[window_name][identity_key] = True
                                            
                                            # Set status color based on result
                                            if status == "Appropriate":
                                                status_color = (0, 255, 0)  # Green
                                            elif status == "Not Appropriate":
                                                status_color = (0, 0, 255)  # Red
                                            else:
                                                status_color = (255, 255, 0)  # Yellow
                                            
                                            # Add status to label if available
                                            if status != "Pending":
                                                label_parts.append(status)
                                        except Exception as e:
                                            print(f"Cloth detection error: {e}", file=sys.stderr)
                        
                        # Draw bounding box and label with appropriate color
                        cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), status_color, 2)
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
                                    score,
                                ]
                            )

                if frame_to_show is None:
                    continue

                if args.show_fps:
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
                cv2.imshow("Mosaic", mosaic)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    except KeyboardInterrupt:
        pass
    finally:
        for window_name, cap in caps:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


