"""
Shared clothing analysis module.
Provides unified clothing detection and violation analysis across CLI and dashboard.
"""
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    SHOE_CONFIDENCE_THRESHOLD,
    SHOE_ROI_START_RATIO,
    SHOE_ROI_MIN_HEIGHT,
    SHOE_ROI_MIN_WIDTH,
    SHOE_RESIZE_TARGET,
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

# Module-level state tracking for shoe detection retries per person
# Key: (global_id, tracking_id) tuple to uniquely identify each person
# Value: dict with 'retry_count' and 'shoes_detected' flag
_shoe_retry_state: Dict[Tuple[Optional[int], Optional[int]], Dict[str, Any]] = {}


def categorize_clothing(label: str) -> Optional[str]:
    """
    Categorize a clothing label into TOP, BOTTOM, or DRESS.
    
    Args:
        label: Clothing label to categorize
        
    Returns:
        "TOP", "BOTTOM", "DRESS", or None if not a clothing item
    """
    label_lower = label.lower().strip()
    
    # Check for dress first (dress keywords are more specific)
    # Use exact match or if label contains the full dress keyword
    for dress_kw in DRESS_KEYWORDS:
        dress_kw_lower = dress_kw.lower()
        # Exact match or label contains the full dress keyword
        if label_lower == dress_kw_lower or dress_kw_lower in label_lower:
            return "DRESS"
    
    # Check for top (but exclude if it's part of a dress)
    # Only check if label doesn't contain "dress"
    if "dress" not in label_lower:
        for top_kw in TOP_KEYWORDS:
            top_kw_lower = top_kw.lower()
            # Exact match or label contains the full top keyword
            if label_lower == top_kw_lower or top_kw_lower in label_lower:
                return "TOP"
    
    # Check for bottom (but exclude if it's part of a dress)
    if "dress" not in label_lower:
        for bottom_kw in BOTTOM_KEYWORDS:
            bottom_kw_lower = bottom_kw.lower()
            # Exact match or label contains the full bottom keyword
            if label_lower == bottom_kw_lower or bottom_kw_lower in label_lower:
                return "BOTTOM"
    
    return None


def has_complete_coverage(
    clothing_labels: List[str], label_confidences: Dict[str, float], all_labels: Optional[List[str]] = None, require_shoes: bool = False
) -> Tuple[bool, List[str]]:
    """
    Check if clothing detection has complete body coverage.
    Complete coverage means:
    - One unique TOP + one unique BOTTOM + (optionally) at least one SHOE, OR
    - One unique DRESS + (optionally) at least one SHOE
    
    Also removes duplicates and returns the filtered labels.
    If a dress is detected, exclude any tops/bottoms that are part of the dress.
    
    Args:
        clothing_labels: List of detected clothing labels (without shoes)
        label_confidences: Dictionary mapping labels to confidence scores
        all_labels: Optional list of all labels including shoes (if None, extracts from clothing_labels)
        require_shoes: If True, shoes are required for complete coverage. If False, shoes are optional.
        
    Returns:
        Tuple of (has_complete_coverage, filtered_unique_labels)
    """
    # Use all_labels if provided, otherwise use clothing_labels
    labels_to_check = all_labels if all_labels is not None else clothing_labels
    
    # Categorize all labels (including shoes)
    tops = []
    bottoms = []
    dresses = []
    shoes = []
    
    for label in labels_to_check:
        category = categorize_clothing(label)
        if category == "TOP":
            tops.append(label)
        elif category == "BOTTOM":
            bottoms.append(label)
        elif category == "DRESS":
            dresses.append(label)
        elif any(kw in label.lower() for kw in SHOE_KEYWORDS):
            shoes.append(label)
        # Unrecognized labels are ignored (not categorized)
    
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
    
    # Limit shoes to only one with highest confidence
    if unique_shoes:
        # Sort by confidence (highest first) and take only the first one
        unique_shoes.sort(key=lambda s: label_confidences.get(s, 0.0), reverse=True)
        unique_shoes = unique_shoes[:1]  # Keep only the highest confidence shoe
    
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
    
    # Check if we have at least one shoe
    has_shoes = len(unique_shoes) >= 1
    
    # Case 1: One unique dress (with optional shoes)
    # A dress provides complete body coverage
    if len(unique_dresses) == 1 and len(unique_tops) == 0 and len(unique_bottoms) == 0:
        if require_shoes:
            # Shoes required: must have shoes for complete coverage
            if has_shoes:
                has_complete = True
                filtered_labels = unique_dresses + unique_shoes
        else:
            # Shoes optional: dress alone is sufficient
            has_complete = True
            filtered_labels = unique_dresses + unique_shoes if has_shoes else unique_dresses
    
    # Case 2: One unique top + one unique bottom (with optional shoes)
    # Must have exactly one top AND exactly one bottom AND no dress
    elif len(unique_tops) == 1 and len(unique_bottoms) == 1 and len(unique_dresses) == 0:
        if require_shoes:
            # Shoes required: must have shoes for complete coverage
            if has_shoes:
                has_complete = True
                filtered_labels = unique_tops + unique_bottoms + unique_shoes
        else:
            # Shoes optional: top + bottom alone is sufficient
            has_complete = True
            filtered_labels = unique_tops + unique_bottoms + (unique_shoes if has_shoes else [])
    
    # Case 3: Incomplete coverage - don't report yet
    # This includes cases like:
    # - Only top(s) without bottom(s) (e.g., only "vest")
    # - Only bottom(s) without top(s)
    # - Multiple tops or bottoms
    # - Mix of dress and top/bottom (ambiguous)
    # - No clothing detected
    # - If require_shoes=True: Top + bottom but no shoes, or Dress but no shoes
    else:
        has_complete = False
        # Return empty list - no complete coverage
        filtered_labels = []
    
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


def preprocess_shoe_roi(roi: np.ndarray) -> np.ndarray:
    """
    Preprocess shoe ROI for better detection.
    
    Args:
        roi: Region of interest containing potential shoes
        
    Returns:
        Preprocessed ROI
    """
    # Strategy 1: Enhance contrast using CLAHE
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Strategy 2: Sharpen image
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Strategy 3: Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    return denoised


def detect_shoes_enhanced(
    person_crop: np.ndarray,
    shoe_model: YOLO,
    shoe_conf: float = 0.25,
    retry_attempt: int = 0,
    max_retries: int = 5,
) -> List[Tuple[str, float]]:
    """
    Enhanced shoe detection with multiple strategies.
    Tries once per call with confidence based on retry_attempt.
    Each person should call this multiple times (once per analyze_clothing_and_log call)
    with incrementing retry_attempt until shoes are detected or max_retries is reached.
    
    Args:
        person_crop: Cropped image of person
        shoe_model: YOLO model for shoe detection
        shoe_conf: Initial confidence threshold
        retry_attempt: Current retry attempt number (0 = first attempt, 1 = first retry, etc.)
        max_retries: Maximum number of retry attempts with lower confidence
        
    Returns:
        List of (label, confidence) tuples for detected shoes
    """
    h, w = person_crop.shape[:2]
    all_detections = []
    
    # Calculate confidence level based on retry attempt
    # retry_attempt 0: 100% confidence
    # retry_attempt 1: 80% confidence
    # retry_attempt 2: 60% confidence
    # retry_attempt 3: 40% confidence
    if retry_attempt == 0:
        attempt_conf = shoe_conf  # 100%
    elif retry_attempt <= max_retries:
        # Progressively lower confidence: 80%, 60%, 40% of original
        attempt_conf = shoe_conf * (1.0 - retry_attempt * 0.2)
    else:
        # Beyond max retries, use minimum confidence (20%)
        attempt_conf = shoe_conf * 0.2
    
    # Try with current confidence level
    # Strategy 1: Multi-level ROI detection (bottom 60%, 70%, 80%)
    roi_ratios = [0.4, 0.3, 0.2]  # Start from 40%, 30%, 20% (bottom 60%, 70%, 80%)
    
    for ratio in roi_ratios:
        y_start = int(h * ratio)
        shoe_roi = person_crop[y_start:h, :]
        roi_h, roi_w = shoe_roi.shape[:2]
        
        # Skip if ROI too small
        if roi_h < 40 or roi_w < 40:
            continue
        
        # Preprocess ROI for better detection
        processed_roi = preprocess_shoe_roi(shoe_roi)
        
        # Strategy 2: Multi-scale resizing
        scales = [640, 800, 1024]
        
        for target_size in scales:
            aspect_ratio = roi_w / roi_h
            
            if aspect_ratio > 1:
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            else:
                new_h = target_size
                new_w = int(target_size * aspect_ratio)
            
            # Ensure minimum size
            new_h = max(new_h, 60)
            new_w = max(new_w, 60)
            
            # Resize with high-quality interpolation
            resized_roi = cv2.resize(
                processed_roi,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Run detection with current confidence level
            results = shoe_model(
                resized_roi,
                conf=attempt_conf,
                iou=0.5,  # Add IOU threshold for better NMS
                verbose=False,
            )[0]
            
            if results and results.boxes is not None:
                for det in results.boxes:
                    cls_id = int(det.cls)
                    label = results.names.get(cls_id, "shoe")
                    confidence = float(det.conf)
                    
                    # Filter out very small detections (likely noise)
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                    box_area = (x2 - x1) * (y2 - y1)
                    roi_area = new_w * new_h
                    
                    # Shoe should occupy reasonable area (not too small, not entire image)
                    if 0.02 < (box_area / roi_area) < 0.8:
                        all_detections.append((label, confidence))
    
    # Strategy 3: Full image detection with lower confidence (backup)
    if not all_detections:
        full_results = shoe_model(
            person_crop,
            conf=attempt_conf * 0.7,  # Lower confidence for full image
            verbose=False,
        )[0]
        
        if full_results and full_results.boxes is not None:
            for det in full_results.boxes:
                # Only consider detections in bottom 70% of image
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                center_y = (y1 + y2) / 2
                
                if center_y > h * 0.3:  # Bottom 70%
                    cls_id = int(det.cls)
                    label = full_results.names.get(cls_id, "shoe")
                    confidence = float(det.conf)
                    all_detections.append((label, confidence))
    
    # Remove duplicates and return best detection
    if all_detections:
        # Sort by confidence and remove near-duplicates
        all_detections.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only highest confidence unique shoe
        seen_labels = set()
        unique_detections = []
        for label, conf in all_detections:
            label_lower = label.lower()
            if label_lower not in seen_labels:
                seen_labels.add(label_lower)
                unique_detections.append((label, conf))
        
        return unique_detections[:1]  # Return only best detection
    
    return []


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

    # Run ENHANCED shoe detection model with per-person retry mechanism
    # Each person gets independent retry attempts that reset when a new person appears
    if shoe_model is not None:
        # Create identity key for tracking retry state
        # Use global_id if available, otherwise use tracking_id
        identity_key = (global_id, tracking_id)
        max_retries = 8  # Maximum number of retry attempts
        
        # Check if this is a new person (not in retry state)
        if identity_key not in _shoe_retry_state:
            # New person detected - initialize retry state
            _shoe_retry_state[identity_key] = {
                'retry_count': 0,
                'shoes_detected': False,
            }
        
        # Get current retry state
        retry_state = _shoe_retry_state[identity_key]
        
        # Only attempt detection if shoes haven't been detected yet and retry count is within limit
        if not retry_state['shoes_detected'] and retry_state['retry_count'] <= max_retries:
            # Try detection with current retry attempt
            shoe_detections = detect_shoes_enhanced(
                person_crop,
                shoe_model,
                shoe_conf=shoe_conf,
                retry_attempt=retry_state['retry_count'],
                max_retries=max_retries,
            )
            
            # Check if shoes were detected
            if shoe_detections:
                # Shoes detected - mark as detected and stop retrying
                retry_state['shoes_detected'] = True
            else:
                # No shoes detected - increment retry count for next call
                retry_state['retry_count'] += 1
        else:
            # Shoes already detected or max retries reached - skip detection
            shoe_detections = []
        
        # Add detected shoes to labels
        for shoe_label, shoe_confidence in shoe_detections:
            labels.append(shoe_label)
            if shoe_label not in label_confidences or shoe_confidence > label_confidences[shoe_label]:
                label_confidences[shoe_label] = shoe_confidence

    # Apply complete coverage check and deduplication FIRST
    # Separate clothing from shoes for categorization
    clothing_only_labels = [
        label for label in labels
        if not any(kw in label.lower() for kw in SHOE_KEYWORDS)
    ]
    
    # Check for complete coverage (top+bottom OR dress, with optional shoes)
    # Pass all labels (including shoes) to the function
    # Set require_shoes=False so we proceed even if no shoes detected after retries
    has_complete, filtered_labels = has_complete_coverage(
        clothing_only_labels, label_confidences, all_labels=labels, require_shoes=False
    )
    
    # If no complete coverage, return early without saving to database
    if not has_complete:
        # Return status without saving to database (wait for complete coverage)
        violation_desc = ""
        violation_type = ""
        return "Pending", violation_desc, violation_type
    
    # Additional safety check: filtered_labels should contain at least one clothing item
    # (not just shoes). Filter out shoes to check if we have actual clothing.
    clothing_in_filtered = [
        label for label in filtered_labels
        if not any(kw in label.lower() for kw in SHOE_KEYWORDS)
    ]
    
    # If no clothing items in filtered_labels, return early
    if not clothing_in_filtered:
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