"""
Configuration file for Cloth Detection System.
Centralizes all thresholds and parameters for consistency across modules.
"""

# Shoe Detection Configuration
SHOE_CONFIDENCE_THRESHOLD = 0.25  # Confidence required to consider a shoe detection

# Clothing Detection Configuration
CLOTH_CONFIDENCE_THRESHOLD = 0.50  # Confidence threshold for clothing detection model
CLOTH_MIN_SIZE = 70  # Minimum bounding box size (pixels) for cloth detection
CLOTH_EDGE_MARGIN = 0.03  # Edge margin as fraction of frame dimension (3%)
CLOTH_INTERVAL = 30  # Process cloth detection every N frames per person
CLOTH_MAX_RETRIES = 3  # Maximum retry attempts for failed detections

# Re-Identification Configuration
REID_THRESHOLD = 0.7  # Cosine similarity threshold for matching across cameras
REID_INTERVAL = 10  # Run Re-ID every N frames per camera

# Tracking Configuration
TRACK_MAX_AGE = 30  # Maximum frames to keep lost tracks alive
TRACK_MIN_HITS = 3  # Minimum consecutive hits before reporting a track
TRACK_IOU_THRESHOLD = 0.3  # IoU threshold for association
SIMILARITY_LAMBDA = 0.5  # Blend factor between IoU and Re-ID similarity (0-1)

# Detection Configuration
DETECTION_CONFIDENCE = 0.75  # Default confidence threshold for person detection
DETECTION_IMAGE_SIZE = 640  # Default inference image size (square)

# Banned Clothing Items (violation keywords)
BANNED_KEYWORDS = ["shorts", "skirt", "flipflops", "sandals", "vest", "sling_dress", "sling"]

# Clothing and Shoe Keywords for categorization
CLOTHING_KEYWORDS = ["short", "skirt", "crop", "vest"]
SHOE_KEYWORDS = ["flipflops", "sandals"]

# Clothing Category Keywords
TOP_KEYWORDS = [
    "short_sleeve_top",
    "long_sleeve_top",
    "short_sleeve_outwear",
    "long_sleeve_outwear",
    "vest",
    "sling",
    "crop_top",
    "tank_top",
    "sleeveless",
]
BOTTOM_KEYWORDS = [
    "shorts",
    "trousers",
    "skirt",
]
DRESS_KEYWORDS = [
    "short_sleeve_dress",
    "long_sleeve_dress",
    "vest_dress",
    "sling_dress",
]

# Maximum items to report in evaluation/violation
MAX_CLOTHING_ITEMS = 2  # Maximum clothing items to report
MAX_SHOE_ITEMS = 1  # Maximum shoe items to report

