from __future__ import annotations

import itertools
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from re_id import OsNetReID
from config import (
    TRACK_MAX_AGE,
    TRACK_MIN_HITS,
    TRACK_IOU_THRESHOLD,
    SIMILARITY_LAMBDA,
)


def _xyxy_to_xyah(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    x_c = x1 + w / 2.0
    y_c = y1 + h / 2.0
    a = w / h
    return np.array([x_c, y_c, a, h], dtype=np.float32)


def _xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    x, y, a, h = state[:4]
    w = a * h
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter_area + 1e-6
    return inter_area / union


def _cosine_similarity_matrix(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    if features_a.size == 0 or features_b.size == 0:
        return np.zeros((features_a.shape[0], features_b.shape[0]), dtype=np.float32)
    return np.clip(features_a @ features_b.T, -1.0, 1.0)


class KalmanTrack:
    _id_iter = itertools.count(1)

    def __init__(self, bbox: np.ndarray, score: float, feature: Optional[np.ndarray], max_feature_history: int = 30):
        self.kf = self._init_kalman_filter(_xyxy_to_xyah(bbox))
        self.time_since_update = 0
        self.id = next(self._id_iter)
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.score = score
        self.features: Deque[np.ndarray] = deque(maxlen=max_feature_history)
        if feature is not None:
            self.features.append(feature)

    @staticmethod
    def _init_kalman_filter(measurement: np.ndarray) -> KalmanFilter:
        kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        kf.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.F[i, i + 4] = dt
        kf.H = np.zeros((4, 8), dtype=np.float32)
        kf.H[:4, :4] = np.eye(4, dtype=np.float32)
        kf.R *= 0.01
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q = np.eye(8, dtype=np.float32)
        kf.x[:4] = measurement.reshape(-1, 1)
        return kf

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _xyah_to_xyxy(self.kf.x.flatten())

    def update(self, bbox: np.ndarray, score: float, feature: Optional[np.ndarray]) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        self.kf.update(_xyxy_to_xyah(bbox))
        if feature is not None:
            self.features.append(feature)

    def get_state(self) -> np.ndarray:
        return _xyah_to_xyxy(self.kf.x.flatten())

    def get_feature(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        stacked = np.stack(self.features, axis=0)
        mean_feat = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_feat)
        if norm > 0:
            mean_feat /= norm
        return mean_feat


class DeepOCSort:
    def __init__(
        self,
        reid_extractor: Optional[OsNetReID] = None,
        max_age: int = 10,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        similarity_lambda: float = 0.5,
    ) -> None:
        self.reid_extractor = reid_extractor
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.similarity_lambda = np.clip(similarity_lambda, 0.0, 1.0)
        self.trackers: List[KalmanTrack] = []

    def _prepare_embeddings(
        self,
        detections: List[Dict],
        frame: Optional[np.ndarray],
    ) -> None:
        if self.reid_extractor is None or frame is None:
            return
        for det in detections:
            if det.get("embedding") is None:
                det["embedding"] = self.reid_extractor.extract_from_frame(frame, det["bbox"])

    def _associate(
        self,
        detections: List[Dict],
        predicted_boxes: np.ndarray,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        if len(self.trackers) == 0 or len(detections) == 0:
            return (
                np.empty((0, 2), dtype=int),
                list(range(len(detections))),
                list(range(len(self.trackers))),
            )

        det_boxes = np.stack([d["bbox"] for d in detections], axis=0)
        iou_matrix = 1.0 - _iou_matrix(det_boxes, predicted_boxes)

        track_features = []
        for trk in self.trackers:
            feat = trk.get_feature()
            track_features.append(feat if feat is not None else np.zeros(1, dtype=np.float32))

        det_features = []
        for det in detections:
            feat = det.get("embedding")
            det_features.append(feat if feat is not None else np.zeros(1, dtype=np.float32))

        features_available = (
            len(track_features) > 0
            and len(det_features) > 0
            and track_features[0].ndim == 1
            and det_features[0].ndim == 1
            and track_features[0].shape[0] == det_features[0].shape[0]
        )

        if features_available:
            track_mat = np.stack(track_features, axis=0)
            det_mat = np.stack(det_features, axis=0)
            sim_matrix = 1.0 - _cosine_similarity_matrix(det_mat, track_mat)
            cost_matrix = (
                self.similarity_lambda * iou_matrix + (1.0 - self.similarity_lambda) * sim_matrix
            )
        else:
            cost_matrix = iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.trackers)))

        for r, c in zip(row_ind, col_ind):
            iou = 1.0 - iou_matrix[r, c]
            if iou < self.iou_threshold:
                continue
            matches.append([r, c])
            unmatched_dets.remove(r)
            unmatched_trks.remove(c)

        return np.array(matches), unmatched_dets, unmatched_trks

    def update(
        self,
        detections: List[Dict[str, np.ndarray]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Update tracker state.

        Args:
            detections: list of dicts with keys {"bbox": np.ndarray, "score": float, "embedding": Optional[np.ndarray]}
            frame: current RGB/BGR frame used for extracting embeddings if not provided.
        """
        self._prepare_embeddings(detections, frame)

        predicted_boxes = np.zeros((len(self.trackers), 4), dtype=np.float32)
        to_delete = []
        for i, trk in enumerate(self.trackers):
            bbox = trk.predict()
            predicted_boxes[i, :] = bbox
            if np.any(np.isnan(bbox)):
                to_delete.append(i)
        for idx in reversed(to_delete):
            self.trackers.pop(idx)
            predicted_boxes = np.delete(predicted_boxes, idx, axis=0)

        matches, unmatched_dets, unmatched_trks = self._associate(detections, predicted_boxes)

        for det_idx, trk_idx in matches:
            det = detections[det_idx]
            self.trackers[trk_idx].update(det["bbox"], det["score"], det.get("embedding"))

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.trackers.append(
                KalmanTrack(det["bbox"], det["score"], det.get("embedding"))
            )

        active_tracks: List[Dict] = []
        for trk_idx in reversed(unmatched_trks):
            trk = self.trackers[trk_idx]
            if trk.time_since_update > self.max_age:
                self.trackers.pop(trk_idx)

        for trk in self.trackers:
            # Only require min_hits for newly created tracks
            # Allow tracks with time_since_update > 0 to be shown (handles short occlusions)
            # but still require they meet min_hits threshold
            if trk.hits < self.min_hits:
                continue
            # Only show tracks that haven't exceeded max_age
            if trk.time_since_update > self.max_age:
                continue
            bbox = trk.get_state()
            active_tracks.append(
                {
                    "track_id": trk.id,
                    "bbox": bbox,
                    "score": trk.score,
                }
            )

        return active_tracks


def init_tracker(
    reid_extractor: Optional[OsNetReID] = None,
    max_age: int = 10,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    similarity_lambda: float = 0.5,
) -> DeepOCSort:
    """Factory helper for Deep OC-SORT tracker."""
    return DeepOCSort(
        reid_extractor=reid_extractor,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        similarity_lambda=similarity_lambda,
    )

