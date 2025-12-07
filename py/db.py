from pymongo import MongoClient
from pymongo.errors import PyMongoError
import base64
import cv2
from datetime import datetime
from typing import Optional
import os

# Load the Atlas URI from environment variables first, fall back to the literal
# connection string only if the env vars are not set. This lets you keep secrets
# out of the repo but still test quickly.
DEFAULT_ATLAS_URI = (
    "mongodb+srv://jiesheng:abc123456"
    "@real-timeclothingsystem.avf7zh8.mongodb.net/?appName=Real-timeClothingSystem"
)

ATLAS_URI = os.getenv("ATLAS_URI") or os.getenv("MONGO_URI") or DEFAULT_ATLAS_URI

if not ATLAS_URI:
    try:
        import streamlit as st  # type: ignore

        ATLAS_URI = st.secrets["atlas_uri"]
    except Exception:
        raise RuntimeError(
            "Mongo URI missing. Set ATLAS_URI env var or add atlas_uri to .streamlit/secrets.toml"
        )

try:
    client = MongoClient(ATLAS_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")  # fail fast when auth/network is wrong
except PyMongoError as err:
    raise RuntimeError(f"Could not connect to MongoDB Atlas: {err}") from err

# Choose your database
db = client["clothing_detection_system"]

# Collections (match the names you created in Atlas)
evaluations = db["evaluation"]
violations = db["violation"]


def save_evaluation(
    evaluation_id,
    clothing_detection_id,
    clothing_category,
    status,
    details="-",
    global_id=None,
    tracking_id=None,
):
    doc = {
        "evaluation_id": evaluation_id,
        "clothing_detection_id": clothing_detection_id,
        "clothing_category": clothing_category,
        "status": status,
        "datetime": datetime.now().isoformat(),
        "details": details or "-",
    }
    
    # Add GID and TID if provided
    if global_id is not None:
        doc["global_id"] = global_id
    if tracking_id is not None:
        doc["tracking_id"] = tracking_id

    result = evaluations.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return doc


def save_violation(
    violation_id,
    evaluation_id,
    evaluation_status,
    violation_type,
    description,
    image,
):
    # Encode cropped violation image as Base64 for MongoDB
    _, buffer = cv2.imencode(".jpg", image)
    img_b64 = base64.b64encode(buffer).decode()

    doc = {
        "violation_id": violation_id,
        "evaluation_id": evaluation_id,
        "evaluation_status": evaluation_status,
        "violation_type": violation_type,
        "description": description,
        "image": img_b64,
    }

    violations.insert_one(doc)


def get_next_evaluation_and_cloth_ids():
    """
    Return the next incremental evaluation_id and clothing_detection_id
    based on the existing documents in the evaluation collection.

    This looks at the current document count so that new IDs always
    follow the records that are already stored.
    Uses 5-digit padding (up to 99999) for consistent string sorting.
    """
    count = evaluations.count_documents({})
    idx = count + 1
    # Keep the EV / CD prefix pattern used in the seed data.
    # Use 5-digit padding for consistency with violation IDs
    eval_id = f"EV{idx:05d}"
    cloth_id = f"CD{idx:05d}"
    return eval_id, cloth_id


def get_next_violation_id():
    """
    Return the next incremental violation_id based on the existing
    documents in the violation collection.
    Uses 5-digit padding (up to 99999) for consistent string sorting.
    """
    count = violations.count_documents({})
    idx = count + 1
    return f"V{idx:05d}"


def get_total_evaluations():
    """Return the total count of evaluations in the database."""
    return evaluations.count_documents({})


def get_total_violations():
    """Return the total count of violations in the database."""
    return violations.count_documents({})


def get_evaluation_status_by_gid(global_id: Optional[int]) -> Optional[str]:
    """
    Get the evaluation status (Appropriate/Not Appropriate) for a given Global ID.
    Returns the most recent evaluation status if multiple exist.
    Returns None if no evaluation found.
    """
    if global_id is None:
        return None
    
    # Find the most recent evaluation for this GID, sorted by datetime descending
    evaluation = evaluations.find_one(
        {"global_id": global_id},
        sort=[("datetime", -1)]  # Most recent first
    )
    
    if evaluation:
        return evaluation.get("status")
    return None


def get_latest_global_id() -> int:
    """
    Get the latest (maximum) global_id from the evaluations collection.
    Returns 0 if no evaluations exist with a global_id.
    This ensures GID persistence across system restarts.
    """
    # Find the document with the maximum global_id
    result = evaluations.find_one(
        {"global_id": {"$exists": True, "$ne": None}},
        sort=[("global_id", -1)]  # Sort by global_id descending
    )
    
    if result and "global_id" in result:
        return int(result["global_id"])
    return 0


def migrate_violation_ids_to_padded_format():
    """
    Migrate existing violation IDs to 5-digit padded format (V00001, V00002, etc.).
    Handles old formats: V1, V01, V100, V146, etc.
    
    This function should be run once to update existing records.
    Returns the number of violations updated.
    """
    import re
    
    updated_count = 0
    
    # Get all violations
    all_violations = list(violations.find({}))
    
    for violation in all_violations:
        old_id = violation.get("violation_id", "")
        
        # Skip if already in correct format (V followed by 5 digits)
        if re.match(r"^V\d{5}$", old_id):
            continue
        
        # Extract numeric part from old ID (handles V1, V01, V100, etc.)
        match = re.search(r"V(\d+)", old_id)
        if match:
            numeric_part = int(match.group(1))
            new_id = f"V{numeric_part:05d}"
            
            # Update the violation document
            violations.update_one(
                {"_id": violation["_id"]},
                {"$set": {"violation_id": new_id}}
            )
            updated_count += 1
    
    return updated_count


def migrate_evaluation_ids_to_padded_format():
    """
    Migrate existing evaluation and clothing detection IDs to 5-digit padded format.
    Handles old formats: EV1, EV01, EV100, etc. and CD1, CD01, CD100, etc.
    
    This function should be run once to update existing records.
    Returns the number of evaluations updated.
    """
    import re
    
    updated_count = 0
    
    # Get all evaluations
    all_evaluations = list(evaluations.find({}))
    
    for evaluation in all_evaluations:
        old_eval_id = evaluation.get("evaluation_id", "")
        old_cloth_id = evaluation.get("clothing_detection_id", "")
        update_fields = {}
        
        # Update evaluation_id if needed
        if old_eval_id and not re.match(r"^EV\d{5}$", old_eval_id):
            match = re.search(r"EV(\d+)", old_eval_id)
            if match:
                numeric_part = int(match.group(1))
                new_eval_id = f"EV{numeric_part:05d}"
                update_fields["evaluation_id"] = new_eval_id
        
        # Update clothing_detection_id if needed
        if old_cloth_id and not re.match(r"^CD\d{5}$", old_cloth_id):
            match = re.search(r"CD(\d+)", old_cloth_id)
            if match:
                numeric_part = int(match.group(1))
                new_cloth_id = f"CD{numeric_part:05d}"
                update_fields["clothing_detection_id"] = new_cloth_id
        
        if update_fields:
            # Update the evaluation document
            evaluations.update_one(
                {"_id": evaluation["_id"]},
                {"$set": update_fields}
            )
            updated_count += 1
    
    return updated_count
