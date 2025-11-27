from pymongo import MongoClient
from pymongo.errors import PyMongoError
import base64
import cv2
from datetime import datetime
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
):
    doc = {
        "evaluation_id": evaluation_id,
        "clothing_detection_id": clothing_detection_id,
        "clothing_category": clothing_category,
        "status": status,
        "datetime": datetime.now().isoformat(),
        "details": details or "-",
    }

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
    """
    count = evaluations.count_documents({})
    idx = count + 1
    # Keep the EV / CD prefix pattern used in the seed data.
    eval_id = f"EV{idx:02d}"
    cloth_id = f"CD{idx:02d}"
    return eval_id, cloth_id


def get_next_violation_id():
    """
    Return the next incremental violation_id based on the existing
    documents in the violation collection.
    """
    count = violations.count_documents({})
    idx = count + 1
    return f"V{idx:02d}"


def get_total_evaluations():
    """Return the total count of evaluations in the database."""
    return evaluations.count_documents({})


def get_total_violations():
    """Return the total count of violations in the database."""
    return violations.count_documents({})
