import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

try:
    from torchreid import models as torchreid_models  # type: ignore
except Exception:
    torchreid_models = None

from config import REID_THRESHOLD

__all__ = ["OsNetReID", "init_reid", "PersonReIDManager"]


class OsNetReID:
    """OSNet feature extractor wrapper used by Deep OC-SORT."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if torchreid_models is None:
            raise RuntimeError(
                "torchreid is required for Re-ID. Install with: pip install torchreid"
            )

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = torchreid_models.osnet_x1_0(num_classes=1, pretrained=True)
        if hasattr(self.model, "classifier"):
            self.model.classifier = nn.Identity()
        self.model.to(self.device)

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing = self.model.load_state_dict(state, strict=False)
            if missing.missing_keys:
                print(
                    f"[ReID] Loaded checkpoint with missing keys: {missing.missing_keys}",
                    file=sys.stderr,
                )
            print(f"[ReID] Loaded weights from {model_path}")
        else:
            if model_path:
                print(f"[ReID] Checkpoint not found at {model_path}. Using pretrained OSNet.", file=sys.stderr)

        self.model.eval()
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # OSNet x1_0 output dimension
        self.feature_dim = getattr(self.model, "feature_dim", 512)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        return tensor

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract an L2-normalized feature vector from an image crop."""
        if image is None or image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        with torch.no_grad():
            tensor = self._preprocess(image)
            features = self.model(tensor).squeeze().cpu().numpy()

        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.astype(np.float32)

    def extract_from_frame(self, frame: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], max(x1 + 1, x2))
        y2 = min(frame.shape[0], max(y1 + 1, y2))
        crop = frame[y1:y2, x1:x2]
        return self.extract(crop)


def init_reid(model_path: Optional[str] = None, device: Optional[str] = None) -> OsNetReID:
    """
    Initialize and return an OSNet Re-ID extractor.

    Args:
        model_path: Path to an OSNet checkpoint (.pth). Defaults to pretrained weights.
        device: Torch device string.
    """
    return OsNetReID(model_path=model_path, device=device)


class PersonReIDManager:
    """
    Maintains a global gallery of embeddings and assigns consistent Global IDs (GID)
    across multiple cameras.
    """

    def __init__(
        self, 
        similarity_threshold: float = REID_THRESHOLD, 
        max_features_per_id: int = 50,
        initial_global_id: Optional[int] = None
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_features_per_id = max_features_per_id
        self.gallery: Dict[int, List[np.ndarray]] = {}
        self.track_to_global: Dict[str, Dict[int, int]] = {}
        # Start from initial_global_id + 1 if provided, otherwise start from 1
        self.next_global_id = (initial_global_id + 1) if initial_global_id is not None else 1

    def _register_feature(self, global_id: int, feature: np.ndarray) -> None:
        feats = self.gallery.setdefault(global_id, [])
        feats.append(feature)
        if len(feats) > self.max_features_per_id:
            feats.pop(0)

    def assign_global_id(
        self,
        camera_name: str,
        local_track_id: int,
        feature: np.ndarray,
    ) -> int:
        if feature is None or np.linalg.norm(feature) == 0:
            global_id = self.track_to_global.get(camera_name, {}).get(local_track_id)
            if global_id is not None:
                return global_id
            global_id = self.next_global_id
            self.next_global_id += 1
            return global_id

        best_gid = None
        best_sim = self.similarity_threshold

        norm_feat = feature / (np.linalg.norm(feature) + 1e-8)

        for gid, feats in self.gallery.items():
            for stored in feats:
                sim = float(np.dot(norm_feat, stored))
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid

        if best_gid is None:
            best_gid = self.next_global_id
            self.next_global_id += 1

        self._register_feature(best_gid, norm_feat)
        self.track_to_global.setdefault(camera_name, {})[local_track_id] = best_gid
        return best_gid

    def get_existing_global_id(self, camera_name: str, local_track_id: int) -> Optional[int]:
        return self.track_to_global.get(camera_name, {}).get(local_track_id)

    def reset_camera(self, camera_name: str) -> None:
        if camera_name in self.track_to_global:
            del self.track_to_global[camera_name]

