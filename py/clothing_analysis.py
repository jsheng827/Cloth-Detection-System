"""
Shared clothing analysis module.
Provides unified clothing detection and violation analysis across CLI and dashboard.
"""
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    SHOE_CONFIDENCE_THRESHOLD,
    CLOTH_CONFIDENCE_THRESHOLD,
    CLOTHING_KEYWORDS,
    SHOE_KEYWORDS,
    MAX_CLOTHING_ITEMS,
    MAX_SHOE_ITEMS,
)
from violation_settings import load_violation_settings
from db import (
    save_evaluation,
    save_violation,
    get_next_evaluation_and_cloth_ids,
    get_next_violation_id,
)


def is_banned(label_lower: str, banned_keywords: Optional[List[str]] = None) -> bool:
    """
    Check if label contains any banned keyword as a whole word (not substring).
    
    Args:
        label_lower: Lowercase label to check
        banned_keywords: Optional list of banned keywords (loads from settings if not provided)
        
    Returns:
        True if label contains banned keyword, False otherwise
    """
    # Load violation settings if not provided
    if banned_keywords is None:
        banned_keywords = load_violation_settings()
    
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


def analyze_clothing_and_log(
    person_crop: np.ndarray,
    cam_idx: int,
    clothing_model: YOLO,
    shoe_model: Optional[YOLO],
    global_id: Optional[int] = None,
    tracking_id: Optional[int] = None,
    shoe_conf: Optional[float] = None,
) -> Tuple[str, str, str]:
    """
    Run cloth and shoe models on a cropped person image, decide status, save into MongoDB,
    and return (status, description, violation_type).
    
    Args:
        person_crop: Cropped image of person (numpy array)
        cam_idx: Camera index (1-based)
        clothing_model: YOLO model for clothing detection
        shoe_model: Optional YOLO model for shoe detection
        global_id: Optional global ID for person re-identification
        tracking_id: Optional tracking ID
        shoe_conf: Optional shoe confidence threshold (defaults to config value)
        
    Returns:
        Tuple of (status, violation_description, violation_type)
    """
    # Use provided shoe_conf or default from config
    if shoe_conf is None:
        shoe_conf = SHOE_CONFIDENCE_THRESHOLD
    
    # Load current violation settings (allows dynamic updates from Settings page)
    banned_keywords = load_violation_settings()
    
    # Run clothing detection model
    cloth_results = clothing_model(person_crop, conf=CLOTH_CONFIDENCE_THRESHOLD, verbose=False)[0]

    # Collect all classes detected on this person (clothing) with confidence scores
    labels = []
    label_confidences: Dict[str, float] = {}  # Store label -> confidence mapping
    for det in cloth_results.boxes:
        cls_id = int(det.cls)
        label = cloth_results.names.get(cls_id, "cloth")
        confidence = float(det.conf)  # Get confidence score
        labels.append(label)
        # Store highest confidence if label appears multiple times
        if label not in label_confidences or confidence > label_confidences[label]:
            label_confidences[label] = confidence

    # Run shoe detection model on the lower half of the person crop
    if shoe_model is not None:
        h, w = person_crop.shape[:2]
        # Use bottom 80% of person (keep more context for small people)
        y_start = int(h * 0.2)
        shoe_roi = person_crop[y_start:h, :]

        # If ROI is too small, skip
        if shoe_roi.shape[0] >= 20 and shoe_roi.shape[1] >= 20:
            # Resize ROI to larger fixed size so shoes aren't tiny
            shoe_roi_resized = cv2.resize(shoe_roi, (640, 640))

            # Run shoe model
            shoe_results = shoe_model(
                shoe_roi_resized,
                conf=shoe_conf,
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

    # Find which specific label(s) triggered the violation
    banned_labels = [label for label in labels if is_banned(label.lower(), banned_keywords)]
    is_violation = len(banned_labels) > 0

    status = "Not Appropriate" if is_violation else "Appropriate"

    # Separate clothing vs shoe labels for violation reporting
    unique_banned_labels = []
    seen = set()
    for label in banned_labels:
        if label not in seen:
            unique_banned_labels.append(label)
            seen.add(label)

    clothing_candidates = []
    shoe_candidates = []

    for label in unique_banned_labels:
        ll = label.lower()
        if any(k in ll for k in SHOE_KEYWORDS):
            shoe_candidates.append(label)
        elif any(k in ll for k in CLOTHING_KEYWORDS):
            clothing_candidates.append(label)
        else:
            clothing_candidates.append(label)

    clothing_candidates.sort(key=lambda l: label_confidences.get(l, 0.0), reverse=True)
    shoe_candidates.sort(key=lambda l: label_confidences.get(l, 0.0), reverse=True)

    selected_clothing = clothing_candidates[:MAX_CLOTHING_ITEMS]
    selected_shoes = shoe_candidates[:MAX_SHOE_ITEMS]
    selected_labels = selected_clothing + selected_shoes

    # Build evaluation labels (limit to max items for both clothing and shoes)
    all_clothing_candidates = []
    all_shoe_candidates = []
    for label in sorted(set(labels)):
        ll = label.lower()
        if any(k in ll for k in SHOE_KEYWORDS):
            all_shoe_candidates.append(label)
        elif any(k in ll for k in CLOTHING_KEYWORDS):
            all_clothing_candidates.append(label)
        else:
            all_clothing_candidates.append(label)

    all_clothing_candidates.sort(key=lambda l: label_confidences.get(l, 0.0), reverse=True)
    all_shoe_candidates.sort(key=lambda l: label_confidences.get(l, 0.0), reverse=True)

    eval_clothing = all_clothing_candidates[:MAX_CLOTHING_ITEMS]
    eval_shoes = all_shoe_candidates[:MAX_SHOE_ITEMS]
    eval_labels = eval_clothing + eval_shoes

    if not eval_labels and labels:
        eval_labels = sorted(set(labels))

    if not eval_labels:
        clothing_category = "Unknown"
    else:
        clothing_category = ", ".join(eval_labels)

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
        if not selected_labels:
            selected_labels = unique_banned_labels

        violation_type = ", ".join(selected_labels) if selected_labels else "unknown"

        # Build description with all violation types and their confidence scores
        violation_parts = []
        for label in selected_labels:
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

