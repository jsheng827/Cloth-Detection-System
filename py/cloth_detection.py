import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


class ClothDetectionService:
    """
    Runs a secondary YOLO model (best.pt) on person crops to detect clothing attributes.
    Keeps track of which Global IDs (GID) have already been processed to avoid duplicates.
    Falls back to Tracking ID (TID) when GID is not available.
    """

    def __init__(
        self,
        model_path: str = "./model/best.pt",
        output_dir: str = "data/cloth_detections",
        conf: float = 0.25,
        process_interval: int = 30,
        min_size: int = 80,
        edge_margin: float = 0.03,
        max_retries: int = 3,
    ) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        self.conf = conf
        self.process_interval = process_interval  # Process every N frames per person
        self.min_size = min_size  # Minimum bounding box size (pixels)
        self.edge_margin = edge_margin  # Edge margin as fraction of frame dimension
        self.max_retries = max_retries  # Maximum retry attempts for failed detections
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.output_dir, f"{timestamp}_cloth_detection.csv")
        self._init_log()

        # Track processed identities (successful detections only)
        self.processed_ids: set[str] = set()
        # Track both GID and TID to avoid conflicts
        self.gid_to_tid: Dict[int, int] = {}  # Map GID to TID
        self.tid_to_gid: Dict[int, int] = {}  # Map TID to GID
        
        # Frame counters per identity for interval-based processing
        self.frame_counters: Dict[str, int] = {}
        
        # Track failed attempts and visibility for retry logic
        self.failed_attempts: Dict[str, int] = {}  # identity_key -> attempt count
        self.last_bbox_size: Dict[str, Tuple[int, int]] = {}  # identity_key -> (width, height)

    def _init_log(self) -> None:
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("timestamp,tracking_id,global_id,clothing,confidence,image_path\n")

    def has_processed(self, identity_key: str) -> bool:
        """Check if identity has been successfully processed."""
        return identity_key in self.processed_ids
    
    def _get_identity_keys(self, tracking_id: int, global_id: Optional[int]) -> Tuple[str, str]:
        """Get both GID and TID identity keys to check for conflicts."""
        gid_key = f"GID_{global_id}" if global_id is not None else None
        tid_key = f"TID_{tracking_id}"
        return gid_key, tid_key
    
    def _should_process_now(self, identity_key: str) -> bool:
        """Check if enough frames have passed since last attempt (interval-based)."""
        if identity_key not in self.frame_counters:
            self.frame_counters[identity_key] = 0
            return True
        self.frame_counters[identity_key] += 1
        return self.frame_counters[identity_key] >= self.process_interval

    def _is_person_fully_visible(
        self, frame, bbox: Tuple[int, int, int, int], identity_key: str
    ) -> Tuple[bool, bool]:
        """
        Check if person is fully visible (not partially entering/exiting frame).
        
        Args:
            frame: Full frame image
            bbox: Bounding box (x1, y1, x2, y2)
            identity_key: Identity key for tracking visibility improvements
        
        Returns:
            (is_visible, visibility_improved) tuple
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Check minimum size (configurable)
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width < self.min_size or bbox_height < self.min_size:
            return False, False
        
        # Check if bounding box is too close to frame edges (configurable margin)
        margin_x = int(w * self.edge_margin)
        margin_y = int(h * self.edge_margin)
        
        # Person is considered partially visible if bbox touches frame edges
        if x1 < margin_x or y1 < margin_y or x2 > (w - margin_x) or y2 > (h - margin_y):
            return False, False
        
        # Check if visibility has improved (for retry logic)
        visibility_improved = False
        if identity_key in self.last_bbox_size:
            prev_width, prev_height = self.last_bbox_size[identity_key]
            # Consider improved if bbox is significantly larger
            if bbox_width > prev_width * 1.2 or bbox_height > prev_height * 1.2:
                visibility_improved = True
        
        # Update last known bbox size
        self.last_bbox_size[identity_key] = (bbox_width, bbox_height)
        
        return True, visibility_improved

    def _crop_person(self, frame, bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, max(x1 + 1, x2))
        y2 = min(h, max(y1 + 1, y2))
        return frame[y1:y2, x1:x2]

    def process(
        self,
        tracking_id: int,
        frame,
        bbox: List[float],
        global_id: Optional[int] = None,
    ) -> Optional[str]:
        """
        Run cloth detection on the provided crop if this Global ID (or Tracking ID) has not been processed.
        Only processes if person is fully visible in the frame.
        Returns the log line written, or None if skipped.
        """
        # Get both GID and TID keys to check for conflicts
        gid_key, tid_key = self._get_identity_keys(tracking_id, global_id)
        
        # Use GID for primary identity when available, fallback to TID
        primary_key = gid_key if gid_key is not None else tid_key
        
        # Check if already successfully processed (using primary key)
        if self.has_processed(primary_key):
            return None
        
        # Check for identity conflicts: if processed with TID but now has GID, or vice versa
        if gid_key and tid_key:
            # Update mappings
            if global_id is not None:
                self.gid_to_tid[global_id] = tracking_id
            self.tid_to_gid[tracking_id] = global_id
            
            # Check if either identity was already processed
            if self.has_processed(gid_key) or self.has_processed(tid_key):
                # If one was processed, mark the other as processed too
                if self.has_processed(gid_key):
                    self.processed_ids.add(tid_key)
                else:
                    self.processed_ids.add(gid_key)
                return None
        
        # Interval-based processing: only process every N frames
        if not self._should_process_now(primary_key):
            return None

        # Validate that person is fully visible before processing
        bbox_int = tuple(int(v) for v in bbox)
        is_visible, visibility_improved = self._is_person_fully_visible(frame, bbox_int, primary_key)
        
        if not is_visible:
            # Allow retry if visibility has improved and haven't exceeded max retries
            if visibility_improved and primary_key in self.failed_attempts:
                if self.failed_attempts[primary_key] < self.max_retries:
                    # Reset frame counter to allow immediate retry
                    self.frame_counters[primary_key] = self.process_interval - 1
            return None

        crop = self._crop_person(frame, bbox_int)
        if crop.size == 0:
            return None

        results = self.model.predict(crop, conf=self.conf, verbose=False)
        detections: List[str] = []
        confidences: List[str] = []

        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                detections.append(names.get(cls_id, f"class_{cls_id}"))
                confidences.append(f"{conf:.2f}")

        if not detections:
            # Don't mark as processed on failure - allow retry
            if primary_key not in self.failed_attempts:
                self.failed_attempts[primary_key] = 0
            self.failed_attempts[primary_key] += 1
            
            # Reset frame counter for retry
            self.frame_counters[primary_key] = 0
            return None

        # Successful detection - mark as processed and clear failed attempts
        timestamp = datetime.now().isoformat()
        if global_id is not None:
            image_name = f"{timestamp.replace(':', '').replace('-', '')}_gid{global_id}_tid{tracking_id}.jpg"
        else:
            image_name = f"{timestamp.replace(':', '').replace('-', '')}_tid{tracking_id}.jpg"
        image_path = os.path.join(self.output_dir, image_name)
        cv2.imwrite(image_path, crop)

        clothing_str = "|".join(detections)
        confidence_str = "|".join(confidences)

        log_line = (
            f"{timestamp},{tracking_id},{global_id if global_id is not None else ''},"
            f"{clothing_str},{confidence_str},{image_path}\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

        # Mark both GID and TID as processed to avoid conflicts
        self.processed_ids.add(primary_key)
        if gid_key and gid_key != primary_key:
            self.processed_ids.add(gid_key)
        if tid_key != primary_key:
            self.processed_ids.add(tid_key)
        
        # Clear failed attempts and reset frame counter
        if primary_key in self.failed_attempts:
            del self.failed_attempts[primary_key]
        self.frame_counters[primary_key] = 0
        
        return log_line

