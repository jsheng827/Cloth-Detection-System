import argparse
import os
import shutil
import sys
from typing import List, Tuple, Optional

try:
    import cv2
except Exception as cv2_error:  # pragma: no cover
    print("OpenCV (cv2) is required. Install with: pip install opencv-python", file=sys.stderr)
    raise

try:
    from ultralytics import YOLO
except Exception as yolo_error:  # pragma: no cover
    print(
        "Ultralytics is required for YOLOv11. Install with: pip install ultralytics",
        file=sys.stderr,
    )
    raise


def parse_sources(raw_sources: List[str]) -> List[Tuple[int, str]]:
    """
    Convert CLI sources into a list of tuples (kind, value_str) where kind=0 means integer cam index,
    and kind=1 means URL/path string. Keeping both forms allows better VideoCapture construction and window naming.
    """
    parsed: List[Tuple[int, str]] = []
    for s in raw_sources:
        # Try to interpret as integer camera index
        try:
            idx = int(s)
            parsed.append((0, str(idx)))
            continue
        except ValueError:
            pass
        parsed.append((1, s))
    return parsed


def open_captures(sources: List[Tuple[int, str]]) -> List[Tuple[str, cv2.VideoCapture]]:
    """
    Open cv2.VideoCapture for each source. Returns list of (window_name, capture).
    Window name is derived from camera index or URL basename.
    """
    captures: List[Tuple[str, cv2.VideoCapture]] = []
    for kind, value in sources:
        if kind == 0:
            cap = cv2.VideoCapture(int(value))
            window_name = f"Camera {value}"
        else:
            cap = cv2.VideoCapture(value)
            # Shorten long URLs/paths for window naming
            short = value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            window_name = f"Source {short}"
        if not cap.isOpened():
            print(f"Warning: Unable to open source: {value}", file=sys.stderr)
            continue
        # Try to set a sane FPS buffer for webcams (best effort)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        captures.append((window_name, cap))
    return captures


# Flexible model loader to make swapping detection backends easy
def load_model(model_identifier: str, device: Optional[str] = None) -> YOLO:
    """
    Load a detection model. Accepts either a local path to a .pt file or a known YOLO model name.

    Examples:
    - "./model/yolov8m.pt" (local file in model folder)
    - "yolov8m" or "yolo11m" (Ultralytics hub names)

    device: Optional device string like "cpu" or "cuda:0". If None, Ultralytics picks automatically.
    """
    identifier = model_identifier

    # Allow TensorRT plans (.plan) by mirroring to .engine extension expected by Ultralytics
    root, ext = os.path.splitext(identifier)
    if ext.lower() == ".plan":
        engine_path = root + ".engine"
        try:
            if (not os.path.exists(engine_path)) or (
                os.path.getmtime(identifier) > os.path.getmtime(engine_path)
            ):
                shutil.copyfile(identifier, engine_path)
        except Exception as copy_err:
            print(f"Failed to prepare TensorRT engine copy: {copy_err}", file=sys.stderr)
            raise
        identifier = engine_path
    # If given a bare name without extension and a matching .pt file exists locally, use it
    if not os.path.splitext(identifier)[1]:
        candidate = identifier + ".pt"
        if os.path.exists(candidate):
            identifier = candidate
    # If given a relative or absolute path ensure it exists, else fall back to name as-is
    if os.path.splitext(identifier)[1] and not os.path.exists(identifier):
        # Still allow non-local weight identifiers resolvable by Ultralytics (e.g., repo names)
        pass

    model = YOLO(identifier)
    # Set device if provided
    if device is not None:
        try:
            model.to(device)
        except Exception:
            # Some Ultralytics versions set device via model.predict(..., device=)
            pass
    return model


# The runnable main entrypoint has been moved to main.py


