"""
Model management utilities for storing and retrieving uploaded models.
"""
import json
import os
from typing import Dict, List, Optional
from pathlib import Path


# Get the model directory path
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)


def load_metadata() -> Dict:
    """Load model metadata from JSON file."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_metadata(metadata: Dict) -> None:
    """Save model metadata to JSON file."""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_models_by_type(model_type: str) -> List[Dict]:
    """
    Get all models of a specific type.
    
    Args:
        model_type: One of "detection", "reid", "cloth"
    
    Returns:
        List of dicts with keys: name, path, type, uploaded_at
    """
    metadata = load_metadata()
    models = metadata.get("models", [])
    return [m for m in models if m.get("type") == model_type]


def get_all_models() -> List[Dict]:
    """Get all uploaded models."""
    metadata = load_metadata()
    return metadata.get("models", [])


def add_model(
    model_name: str,
    model_type: str,
    file_path: str,
    original_filename: Optional[str] = None,
) -> Dict:
    """
    Add a model to the metadata.
    
    Args:
        model_name: User-friendly name for the model
        model_type: One of "detection", "reid", "cloth"
        file_path: Relative path to the model file (e.g., "./model/my_model.pt")
        original_filename: Original filename before rename (optional)
    
    Returns:
        Model dict with metadata
    """
    metadata = load_metadata()
    if "models" not in metadata:
        metadata["models"] = []
    
    from datetime import datetime
    
    model_info = {
        "name": model_name,
        "type": model_type,
        "path": file_path,
        "original_filename": original_filename or os.path.basename(file_path),
        "uploaded_at": datetime.now().isoformat(),
    }
    
    metadata["models"].append(model_info)
    save_metadata(metadata)
    return model_info


def delete_model(model_name: str, model_type: str) -> bool:
    """
    Delete a model from metadata and optionally remove the file.
    
    Args:
        model_name: Name of the model to delete
        model_type: Type of the model
    
    Returns:
        True if deleted, False if not found
    """
    metadata = load_metadata()
    models = metadata.get("models", [])
    
    for i, model in enumerate(models):
        if model.get("name") == model_name and model.get("type") == model_type:
            # Optionally delete the file
            model_path = BASE_DIR / model["path"].lstrip("./")
            if model_path.exists():
                try:
                    model_path.unlink()
                except Exception:
                    pass  # File deletion failed, but continue with metadata removal
            
            models.pop(i)
            metadata["models"] = models
            save_metadata(metadata)
            return True
    
    return False


def get_model_path(model_name: str, model_type: str) -> Optional[str]:
    """
    Get the file path for a model by name and type.
    
    Args:
        model_name: Name of the model
        model_type: Type of the model
    
    Returns:
        File path (relative to project root) or None if not found
    """
    models = get_models_by_type(model_type)
    for model in models:
        if model.get("name") == model_name:
            path = model.get("path")
            # Ensure path exists
            if path:
                full_path = BASE_DIR / path.lstrip("./")
                if full_path.exists():
                    return path
                else:
                    # File doesn't exist, return None
                    return None
    return None


def model_exists(model_name: str, model_type: str) -> bool:
    """Check if a model with given name and type exists."""
    models = get_models_by_type(model_type)
    return any(m.get("name") == model_name for m in models)

