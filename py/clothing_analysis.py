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
    SHOE_KEYWORDS,
    TOP_KEYWORDS,
    BOTTOM_KEYWORDS,
    DRESS_KEYWORDS,
)
from violation_settings import load_violation_settings
from db import (
    save_evaluation,
    save_violation,
    get_next_evaluation_and_cloth_ids,
    get_next_violation_id,
)


def categorize_clothing(label: str) -> Optional[str]:
    """
    Categorize a clothing label into TOP, BOTTOM, or DRESS.
    
    IMPORTANT:
    - Ensure that plain tops like "vest" are NOT treated as dresses like "vest_dress".
    - A dress must match the full dress keyword, not just share a substring.
    
    Args:
        label: Clothing label to categorize
        
    Returns:
        "TOP", "BOTTOM", "DRESS", or None if not a clothing item
    """
    label_lower = label.lower().strip()
    
    # Check for dress first (dress keywords are more specific)
    # Only consider it a dress when the full dress keyword appears in the label
    # or the label exactly matches a dress keyword.
    # Example:
    #   - "vest_dress"  -> DRESS  (matches "vest_dress")
    #   - "red_vest_dress" -> DRESS  (contains "vest_dress")
    #   - "vest"       -> TOP (should NOT be DRESS)
    for dress_kw in DRESS_KEYWORDS:
        dress_kw_lower = dress_kw.lower()
        if label_lower == dress_kw_lower or dress_kw_lower in label_lower:
            return "DRESS"
    
    # Check for top (but exclude labels that clearly reference a dress)
    # This avoids classifying "vest_dress" as TOP.
    if "dress" not in label_lower:
        for top_kw in TOP_KEYWORDS:
            top_kw_lower = top_kw.lower()
            if label_lower == top_kw_lower or top_kw_lower in label_lower:
                return "TOP"
    
    # Check for bottom (also exclude obvious dress labels)
    if "dress" not in label_lower:
        for bottom_kw in BOTTOM_KEYWORDS:
            bottom_kw_lower = bottom_kw.lower()
            if label_lower == bottom_kw_lower or bottom_kw_lower in label_lower:
                return "BOTTOM"
    
    return None


def has_complete_coverage(
    clothing_labels: List[str], label_confidences: Dict[str, float]
) -> Tuple[bool, List[str]]:
    """
    Check if clothing detection has complete body coverage.
    Complete coverage means:
    - One unique TOP + one unique BOTTOM, OR
    - One unique DRESS
    
    Also removes duplicates and returns the filtered labels.
    If a dress is detected, exclude any tops/bottoms that are part of the dress.
    
    Args:
        clothing_labels: List of detected clothing labels
        label_confidences: Dictionary mapping labels to confidence scores
        
    Returns:
        Tuple of (has_complete_coverage, filtered_unique_labels)
    """
    # Categorize all clothing labels
    tops = []
    bottoms = []
    dresses = []
    shoes = []
    other = []
    
    for label in clothing_labels:
        category = categorize_clothing(label)
        if category == "TOP":
            tops.append(label)
        elif category == "BOTTOM":
            bottoms.append(label)
        elif category == "DRESS":
            dresses.append(label)
        elif any(kw in label.lower() for kw in SHOE_KEYWORDS):
            shoes.append(label)
        else:
            other.append(label)
    
    # Remove duplicates within each category (keep highest confidence)
    def deduplicate_category(items: List[str]) -> List[str]:
        """Remove duplicates, keeping the one with highest confidence."""
        seen = {}
        for item in items:
            item_lower = item.lower()
            if item_lower not in seen:
                seen[item_lower] = item
            else:
                # Keep the one with higher confidence
                current_conf = label_confidences.get(item, 0.0)
                existing_conf = label_confidences.get(seen[item_lower], 0.0)
                if current_conf > existing_conf:
                    seen[item_lower] = item
        return list(seen.values())
    
    unique_tops = deduplicate_category(tops)
    unique_bottoms = deduplicate_category(bottoms)
    unique_dresses = deduplicate_category(dresses)
    unique_shoes = deduplicate_category(shoes)
    
    # If a dress is detected, filter out tops/bottoms that might be part of the dress
    # (e.g., if "vest_dress" is detected, don't count "vest" as a separate top)
    if unique_dresses:
        filtered_tops = []
        filtered_bottoms = []
        
        # Collect all dress names (lowercase) for comparison
        dress_names_lower = [d.lower() for d in unique_dresses]
        
        # Filter tops: exclude any top that is a substring of any dress
        for top in unique_tops:
            top_lower = top.lower()
            # Keep top only if it's not part of any dress name
            is_part_of_dress = any(top_lower in dress_name for dress_name in dress_names_lower)
            if not is_part_of_dress:
                filtered_tops.append(top)
        
        # Filter bottoms: exclude any bottom that is a substring of any dress
        for bottom in unique_bottoms:
            bottom_lower = bottom.lower()
            is_part_of_dress = any(bottom_lower in dress_name for dress_name in dress_names_lower)
            if not is_part_of_dress:
                filtered_bottoms.append(bottom)
        
        # Update the lists
        unique_tops = filtered_tops
        unique_bottoms = filtered_bottoms
    
    # Check for complete coverage
    has_complete = False
    filtered_labels = []
    
    # Case 1: One unique dress (complete coverage)
    if len(unique_dresses) == 1:
        has_complete = True
        filtered_labels = unique_dresses + unique_shoes
    
    # Case 2: One unique top + one unique bottom (complete coverage)
    elif len(unique_tops) == 1 and len(unique_bottoms) == 1:
        has_complete = True
        filtered_labels = unique_tops + unique_bottoms + unique_shoes
    
    # Case 3: Incomplete coverage - don't report yet
    else:
        has_complete = False
        # Still include shoes if any, but don't save to DB
        filtered_labels = unique_shoes
    
    return has_complete, filtered_labels


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

    # Apply complete coverage check and deduplication FIRST
    # Separate clothing from shoes for categorization
    clothing_only_labels = [
        label for label in labels
        if not any(kw in label.lower() for kw in SHOE_KEYWORDS)
    ]
    
    # Check for complete coverage (top+bottom OR dress) and get filtered labels
    has_complete, filtered_labels = has_complete_coverage(
        clothing_only_labels, label_confidences
    )
    
    # If no complete coverage, return early without saving to database
    if not has_complete:
        # Return status without saving to database (wait for complete coverage)
        violation_desc = ""
        violation_type = ""
        return "Pending", violation_desc, violation_type
    
    # Now check for violations using the filtered labels (complete coverage confirmed)
    banned_labels = [label for label in filtered_labels if is_banned(label.lower(), banned_keywords)]
    is_violation = len(banned_labels) > 0

    status = "Not Appropriate" if is_violation else "Appropriate"

    # Build evaluation labels from filtered labels (already deduplicated and complete)
    eval_labels = filtered_labels
    
    # Sort by confidence for consistent ordering
    eval_labels.sort(key=lambda l: label_confidences.get(l, 0.0), reverse=True)
    
    if not eval_labels:
        clothing_category = "Unknown"
    else:
        clothing_category = ", ".join(eval_labels)

    # Skip saving to database if clothing_category is "Unknown"
    if clothing_category == "Unknown":
        # Return status without saving to database
        violation_desc = ""
        violation_type = ""
        return status, violation_desc, violation_type
    
    # Prepare violation reporting labels (from banned items in filtered labels)
    selected_labels = banned_labels if banned_labels else []

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

