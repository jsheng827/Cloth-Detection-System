"""
Violation settings management.
Allows users to configure which clothing items are considered violations.
"""
import json
import os
from pathlib import Path
from typing import List

# Get the model directory path (same location as model metadata)
BASE_DIR = Path(__file__).parent.parent
SETTINGS_DIR = BASE_DIR / "model"
VIOLATION_SETTINGS_FILE = SETTINGS_DIR / "violation_settings.json"

# Ensure settings directory exists
SETTINGS_DIR.mkdir(exist_ok=True)

# Default violation types (common clothing items that might be violations)
DEFAULT_VIOLATION_TYPES = [
    "shorts",
    "skirt",
    "flipflops",
    "sandals",
    "vest",
    "sling_dress",
    "sling",
]

# All available clothing items that can be detected (common YOLO clothing classes)
# This list includes common clothing items that might be detected by clothing detection models
ALL_AVAILABLE_ITEMS = [
    # Tops
    "short_sleeve_top",
    "long_sleeve_top",
    "short_sleeve_outwear",
    "long_sleeve_outwear",
    "vest",
    "sling",
    # Bottoms
    "shorts",
    "trousers",
    "skirt",
    # Dresses
    "short_sleeve_dress",
    "long_sleeve_dress",
    "vest_dress",
    "sling_dress",
    # Footwear
    "sandals",
    "flipflops",
    "sneakers",
    "boots",
    "loafers",
    "heels",
]


def load_violation_settings() -> List[str]:
    """
    Load user-selected violation types from settings file.
    
    Returns:
        List of violation type keywords
    """
    if VIOLATION_SETTINGS_FILE.exists():
        try:
            with open(VIOLATION_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("violation_types", DEFAULT_VIOLATION_TYPES)
        except Exception:
            # If file is corrupted, return defaults
            return DEFAULT_VIOLATION_TYPES
    return DEFAULT_VIOLATION_TYPES


def save_violation_settings(violation_types: List[str]) -> None:
    """
    Save user-selected violation types to settings file.
    
    Args:
        violation_types: List of violation type keywords to save
    """
    data = {
        "violation_types": violation_types,
    }
    with open(VIOLATION_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_all_available_items() -> List[str]:
    """
    Get list of all available clothing items that can be detected.
    
    Returns:
        List of all available clothing item keywords
    """
    # Remove duplicates and sort
    return sorted(list(set(ALL_AVAILABLE_ITEMS)))


def reset_to_defaults() -> List[str]:
    """
    Reset violation settings to default values.
    
    Returns:
        Default violation types list
    """
    save_violation_settings(DEFAULT_VIOLATION_TYPES)
    return DEFAULT_VIOLATION_TYPES

