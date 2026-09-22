import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights


def box_edge_distance(b1: List[int], b2: List[int]) -> float:
    """Calculates Euclidean distance between two bounding box perimeters.
    Returns 0.0 if boxes overlap or touch.
    """
    dx = max(0, max(b1[0] - b2[2], b2[0] - b1[2]))
    dy = max(0, max(b1[1] - b2[3], b2[1] - b1[3]))
    return math.hypot(dx, dy)


def box_overlap_area(b1: List[int], b2: List[int]) -> float:
    """Calculates intersection area between two bounding boxes."""
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        return float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    return 0.0


def box_iou(b1: List[int], b2: List[int]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    inter = box_overlap_area(b1, b2)
    if inter <= 0.0:
        return 0.0
    area1 = float(max(1, b1[2] - b1[0]) * max(1, b1[3] - b1[1]))
    area2 = float(max(1, b2[2] - b2[0]) * max(1, b2[3] - b2[1]))
    union = area1 + area2 - inter
    return inter / max(1.0, union)


def resolve_cross_class_conflicts(detections: List[Dict[str, Any]], iou_threshold: float = 0.30, containment_threshold: float = 0.50) -> List[Dict[str, Any]]:
    """
    Resolves competing multi-class weapon proposals on the exact same physical object.
    When multiple proposals overlap with IoU >= iou_threshold or containment >= containment_threshold,
    retains the candidate with highest confidence and suppresses secondary class ghost proposals.
    """
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    resolved: List[Dict[str, Any]] = []

    while sorted_dets:
        winner = sorted_dets.pop(0)
        w_box = winner["box"]
        w_area = float(max(1, w_box[2] - w_box[0]) * max(1, w_box[3] - w_box[1]))
        i = 0
        while i < len(sorted_dets):
            cand = sorted_dets[i]
            c_box = cand["box"]
            c_area = float(max(1, c_box[2] - c_box[0]) * max(1, c_box[3] - c_box[1]))
            inter = box_overlap_area(w_box, c_box)
            iou = inter / max(1.0, w_area + c_area - inter) if inter > 0 else 0.0
            containment = inter / min(w_area, c_area) if inter > 0 else 0.0

            if iou >= iou_threshold or containment >= containment_threshold:
                if "secondary_proposals" not in winner:
                    winner["secondary_proposals"] = []
                winner["secondary_proposals"].append({
                    "class_name": cand.get("class_name"),
                    "confidence": cand.get("confidence"),
                })
                sorted_dets.pop(i)
            else:
                i += 1
        resolved.append(winner)

    return resolved


class PersonDetector:
    """Lightweight person detector using FasterRCNN MobileNetV3 FPN.
    
    Trained on COCO (class 1: person), this model offers superior feature
    extraction in blurry and degraded surveillance CCTV scenes compared to SSD.
    """

    def __init__(self, device: Optional[torch.device] = None, score_threshold: float = 0.40):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.person_class_id = 1
        self.tracker = TemporalPersonTracker(max_decay_frames=6, iou_threshold=0.25)

    def detect_persons(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect all humans in the current BGR frame with temporal persistence decay."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            predictions = self.model(tensor)[0]

        boxes = predictions["boxes"].detach().cpu().numpy()
        scores = predictions["scores"].detach().cpu().numpy()
        labels = predictions["labels"].detach().cpu().numpy()

        raw_persons = []
        frame_area = float(h * w)
        for box, score, label in zip(boxes, scores, labels):
            if label == self.person_class_id and score >= self.score_threshold:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                pw = max(1, x2 - x1)
                ph = max(1, y2 - y1)
                p_ar = pw / float(ph)
                p_area_rat = (pw * ph) / frame_area
                p_w_rat = pw / float(w)
                # Reject unphysical crowd/counter cluster hallucinations:
                # A single person cannot span > 55% of screen width or have AR (w/h) > 1.35
                if p_w_rat > 0.55 or p_ar > 1.35:
                    continue
                raw_persons.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": float(score),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "width": pw,
                    "height": ph,
                })
        return self.tracker.update(raw_persons)


class TemporalPersonTracker:
    """Maintains a temporal decay memory buffer of detected humans across video frames.
    
    In real surveillance footage, humans do not vanish between consecutive 0.2s frames.
    Maintains a 1.2-second (6-frame) decay persistence buffer so that momentary single-frame
    person detector dropouts (caused by motion blur, head turns, or cashier counter occlusions)
    do not falsely trigger NO_PERSON_IN_SCENE rejections on genuine armed suspects.
    """
    def __init__(self, max_decay_frames: int = 6, iou_threshold: float = 0.25):
        self.max_decay_frames = max_decay_frames
        self.iou_threshold = iou_threshold
        self.buffered_persons: List[Dict[str, Any]] = []

    def update(self, current_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        updated = []
        matched_buffer_indices = set()

        for det in current_detections:
            for idx, buf in enumerate(self.buffered_persons):
                if idx in matched_buffer_indices:
                    continue
                iou = self._compute_iou(det["box"], buf["box"])
                if iou >= self.iou_threshold:
                    matched_buffer_indices.add(idx)
                    break
            updated.append({
                **det,
                "decay_frames": 0,
                "is_buffered": False,
            })

        for idx, buf in enumerate(self.buffered_persons):
            if idx not in matched_buffer_indices:
                new_decay = buf.get("decay_frames", 0) + 1
                if new_decay <= self.max_decay_frames:
                    decayed_conf = buf["confidence"] * 0.92
                    updated.append({
                        **buf,
                        "confidence": decayed_conf,
                        "decay_frames": new_decay,
                        "is_buffered": True,
                    })

        self.buffered_persons = updated
        return updated

    @staticmethod
    def _compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        a2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        return inter / (a1 + a2 - inter)


class CentroidMotionTracker:
    """Tracks bounding box centroids across video frames to detect stationary background objects.
    
    Distinguishes stationary environmental fixtures (pans, railings, booth trims, floor spots)
    from armed humans aiming or holding steady weapons.
    """

    def __init__(
        self,
        max_drift_pixels: float = 30.0,
        static_frame_limit: int = 3,
        max_missing_frames: int = 30,
    ):
        self.max_drift_pixels = max_drift_pixels
        self.static_frame_limit = static_frame_limit
        self.max_missing_frames = max_missing_frames

        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        # Known environmental traps (e.g. pans, railings, pebbles confirmed static when unheld)
        self.static_zones: List[Dict[str, Any]] = []

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
    ) -> List[Dict[str, Any]]:
        """Associates candidate weapon detections with ongoing tracks."""
        candidates = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            candidates.append({
                "det": det,
                "cx": cx,
                "cy": cy,
                "box": [x1, y1, x2, y2],
                "class_name": det["class_name"],
            })

        matched_tracks = set()
        matched_candidates = set()

        for c_idx, cand in enumerate(candidates):
            best_t_id = None
            best_dist = float("inf")

            for t_id, track in self.tracks.items():
                if t_id in matched_tracks:
                    continue

                dist = math.hypot(cand["cx"] - track["last_cx"], cand["cy"] - track["last_cy"])
                if dist < self.max_drift_pixels and dist < best_dist:
                    best_dist = dist
                    best_t_id = t_id

            in_known_static_zone = any(
                z["class_name"] == cand["class_name"]
                and math.hypot(cand["cx"] - z["cx"], cand["cy"] - z["cy"]) < (self.max_drift_pixels * 1.5)
                for z in self.static_zones
            )

            if best_t_id is not None:
                matched_tracks.add(best_t_id)
                matched_candidates.add(c_idx)

                track = self.tracks[best_t_id]
                track["history"].append((cand["cx"], cand["cy"], frame_idx))
                track["last_cx"] = cand["cx"]
                track["last_cy"] = cand["cy"]
                track["missing_frames"] = 0
                track["total_frames"] += 1

                cname = cand["class_name"]
                conf = float(cand["det"].get("confidence", 0.0))
                if "class_votes" not in track:
                    track["class_votes"] = {track["class_name"]: 1}
                track["class_votes"][cname] = track["class_votes"].get(cname, 0) + 1

                if "class_max_conf" not in track:
                    track["class_max_conf"] = {track["class_name"]: 0.0}
                track["class_max_conf"][cname] = max(track["class_max_conf"].get(cname, 0.0), conf)

                recent_history = list(track["history"])[-max(3, self.static_frame_limit):]
                recent_cx, recent_cy, _ = recent_history[0]
                recent_drift = math.hypot(cand["cx"] - recent_cx, cand["cy"] - recent_cy)
                track["displacement"] = round(recent_drift, 1)

                # An object is static if it resides in a registered static zone,
                # or has sustained minimal recent drift (< max_drift_pixels * 0.75) across static_frame_limit frames
                has_minimal_recent_drift = (len(recent_history) >= self.static_frame_limit) and (recent_drift < self.max_drift_pixels * 0.75)
                is_static = in_known_static_zone or has_minimal_recent_drift
                track["is_static"] = is_static

                cand["det"]["track_id"] = best_t_id
                cand["det"]["track_frames"] = track["total_frames"]
                cand["det"]["displacement"] = round(recent_drift, 1)
                cand["det"]["is_static"] = is_static
            else:
                t_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[t_id] = {
                    "class_name": cand["class_name"],
                    "history": deque([(cand["cx"], cand["cy"], frame_idx)], maxlen=60),
                    "last_cx": cand["cx"],
                    "last_cy": cand["cy"],
                    "missing_frames": 0,
                    "total_frames": 1,
                    "displacement": 0.0,
                    "is_static": in_known_static_zone,
                }
                cand["det"]["track_id"] = t_id
                cand["det"]["track_frames"] = 1
                cand["det"]["displacement"] = 0.0
                cand["det"]["is_static"] = in_known_static_zone

        for t_id in list(self.tracks.keys()):
            if t_id not in matched_tracks:
                self.tracks[t_id]["missing_frames"] += 1
                if self.tracks[t_id]["missing_frames"] > self.max_missing_frames:
                    del self.tracks[t_id]

        return detections

    def register_static_zone(self, cx: float, cy: float, class_name: str):
        """Registers a verified unheld environmental fixture into permanent exclusion memory."""
        exists = any(
            z["class_name"] == class_name
            and math.hypot(cx - z["cx"], cy - z["cy"]) < self.max_drift_pixels
            for z in self.static_zones
        )
        if not exists:
            self.static_zones.append({"cx": cx, "cy": cy, "class_name": class_name})


class TemporalConsistencyFilter:
    """
    Enforces temporal persistence and cross-frame label consistency.

    Principles:
    1. Threat Persistence: Genuine weapon incidents in CCTV persist across multiple consecutive frames.
    2. Transient Flicker Suppression: Single-frame isolated false proposals that vanish immediately
       are categorized as 'SUPPRESSED_TEMPORAL_FLICKER' and prevented from triggering false alerts.
    3. Cross-Frame Label Smoothing: If a tracklet has consistent detections of one weapon class
       (e.g., 'handgun' across multiple frames), an isolated single frame where the detector flips
       to 'knife' due to a shadow is smoothed back to 'handgun'.
    """

    def __init__(
        self,
        min_hits: int = 2,
        max_missing_frames: int = 15,
        max_match_distance: float = 115.0,
        enable_label_smoothing: bool = True,
    ):
        self.min_hits = min_hits
        self.max_missing_frames = max_missing_frames
        self.max_match_distance = max_match_distance
        self.enable_label_smoothing = enable_label_smoothing

        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}

    def update(
        self,
        candidates: List[Dict[str, Any]],
        frame_idx: int,
    ) -> List[Dict[str, Any]]:
        """Associates detections with temporal tracks and updates persistence counters."""
        matched_tracks = set()

        for cand in candidates:
            cx = (cand["box"][0] + cand["box"][2]) / 2.0
            cy = (cand["box"][1] + cand["box"][3]) / 2.0

            best_t_id = None
            min_dist = float("inf")

            for t_id, track in self.active_tracks.items():
                if t_id in matched_tracks:
                    continue
                dist = math.hypot(cx - track["last_cx"], cy - track["last_cy"])
                if dist < self.max_match_distance and dist < min_dist:
                    min_dist = dist
                    best_t_id = t_id

            if best_t_id is not None:
                track = self.active_tracks[best_t_id]
                track["hits"] += 1
                track["last_cx"] = cx
                track["last_cy"] = cy
                track["last_frame"] = frame_idx
                track["missing_frames"] = 0
                track["label_history"].append(cand["class_name"])
                matched_tracks.add(best_t_id)

                cand["temporal_track_id"] = best_t_id
                cand["temporal_hits"] = track["hits"]
                cand["is_temporally_consistent"] = track["hits"] >= self.min_hits

                if self.enable_label_smoothing and len(track["label_history"]) >= 3:
                    # Majority vote across last 5 frames
                    recent_labels = list(track["label_history"])[-5:]
                    smoothed = max(set(recent_labels), key=recent_labels.count)
                    cand["class_name"] = smoothed
            else:
                # New tracklet
                t_id = self.next_track_id
                self.next_track_id += 1
                self.active_tracks[t_id] = {
                    "hits": 1,
                    "last_cx": cx,
                    "last_cy": cy,
                    "last_frame": frame_idx,
                    "missing_frames": 0,
                    "label_history": deque([cand["class_name"]], maxlen=15),
                }
                matched_tracks.add(t_id)
                cand["temporal_track_id"] = t_id
                cand["is_temporally_consistent"] = (self.min_hits <= 1)

        # Decay missing tracks
        for t_id in list(self.active_tracks.keys()):
            if t_id not in matched_tracks:
                self.active_tracks[t_id]["missing_frames"] += 1
                if self.active_tracks[t_id]["missing_frames"] > self.max_missing_frames:
                    del self.active_tracks[t_id]

        return candidates


class CCTVIntelligenceFilter:
    """Surveillance intelligence engine combining temporal, spatial, and human reasoning."""

    def __init__(
        self,
        class_thresholds: Optional[Dict[str, float]] = None,
        max_area_ratio: float = 0.12,
        max_edge_distance: float = 45.0,
        max_person_area_ratio: float = 0.20,
        max_person_width_ratio: float = 0.75,
        max_person_height_ratio: Union[float, Dict[str, float]] = 0.35,
        min_person_reach_y: float = 0.22,
        max_person_reach_y: float = 0.95,
        enable_person_gating: bool = True,
        enable_anthropometric_gating: bool = True,
        enable_motion_filtering: bool = True,
        enable_geometric_filtering: bool = True,
        enable_temporal_consistency: bool = True,
        min_temporal_hits: int = 3,
        static_frame_limit: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.class_thresholds = class_thresholds or {
            "handgun": 0.50,
            "knife": 0.50,
        }
        self.max_area_ratio = max_area_ratio
        self.max_edge_distance = max_edge_distance
        self.max_person_area_ratio = max_person_area_ratio
        self.max_person_width_ratio = max_person_width_ratio
        if isinstance(max_person_height_ratio, dict):
            self.max_person_height_ratios = max_person_height_ratio
            self.max_person_height_ratio = max_person_height_ratio.get("knife", 0.35)
        else:
            self.max_person_height_ratio = float(max_person_height_ratio)
            self.max_person_height_ratios = {
                "handgun": min(0.25, float(max_person_height_ratio)),
                "knife": float(max_person_height_ratio),
            }
        self.min_person_reach_y = min_person_reach_y
        self.max_person_reach_y = max_person_reach_y
        self.enable_person_gating = enable_person_gating
        self.enable_anthropometric_gating = enable_anthropometric_gating
        self.enable_motion_filtering = enable_motion_filtering
        self.enable_geometric_filtering = enable_geometric_filtering
        self.enable_temporal_consistency = enable_temporal_consistency
        self.min_temporal_hits = min_temporal_hits

        self.motion_tracker = (
            CentroidMotionTracker(
                max_drift_pixels=30.0,
                static_frame_limit=static_frame_limit,
                max_missing_frames=30,
            )
            if enable_motion_filtering
            else None
        )

        self.person_detector = (
            PersonDetector(device=device) if enable_person_gating else None
        )

        self.temporal_filter = (
            TemporalConsistencyFilter(
                min_hits=min_temporal_hits,
                max_missing_frames=15,
                max_match_distance=115.0,
                enable_label_smoothing=True,
            )
            if enable_temporal_consistency
            else None
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        raw_detections: List[Dict[str, Any]],
        frame_idx: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filters raw weapon detections using surveillance-grade intelligence rules."""
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)

        # Resolve competing proposals on the exact same physical weapon
        raw_detections = resolve_cross_class_conflicts(raw_detections, iou_threshold=0.45)

        if self.motion_tracker:
            raw_detections = self.motion_tracker.update(raw_detections, frame_idx)

        persons = []
        if self.person_detector:
            persons = self.person_detector.detect_persons(frame_bgr)

        confirmed = []
        all_evaluated = []

        for det in raw_detections:
            cname = det["class_name"]
            conf = float(det["confidence"])
            wbox = det["box"]
            x1, y1, x2, y2 = wbox
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            box_area = box_w * box_h
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            rejection_reason = None

            # Check 1: Strict class-specific confidence threshold
            # Proposals must meet or exceed the user-configured confidence threshold.
            req_threshold = float(self.class_thresholds.get(cname, 0.50))
            if conf < req_threshold:
                rejection_reason = f"BELOW_CLASS_THRESHOLD ({conf:.1%} < {req_threshold:.1%})"

            # Check 2: Absolute geometric scale and aspect ratio feasibility
            if rejection_reason is None and self.enable_geometric_filtering:
                area_ratio = box_area / frame_area
                if area_ratio > self.max_area_ratio:
                    rejection_reason = f"GEOMETRIC_OVERSIZED ({area_ratio:.1%} > {self.max_area_ratio:.1%})"
                else:
                    aspect_ratio = box_w / box_h
                    if aspect_ratio > 7.0 or aspect_ratio < 0.12:
                        rejection_reason = f"UNPHYSICAL_ASPECT_RATIO ({aspect_ratio:.2f})"
                    elif cname == "handgun" and aspect_ratio < 0.50:
                        rejection_reason = f"UNPHYSICAL_HANDGUN_ASPECT_RATIO (AR={aspect_ratio:.2f} < 0.50 vertical distractor)"

            # Check 2b: Photometric Material, Bare Hand, & Cast Shadow Feasibility
            # Real firearms exhibit dark blued steel/polymer features (trigger guard, ejection port, muzzle).
            # Empty hand gestures casting faint translucent shadows on bright doors/walls have virtually zero
            # deep-black pixels (deep_dark < 1.2%) despite high background brightness (mean_V > 115).
            # Empty swinging bare hands/wrists against dark clothing are dominated by bare skin (skin_ratio >= 55%).
            # Smooth dark shadow gaps / fabric folds have low Laplacian variance (< 18.0) without metallic weapon edges.
            if rejection_reason is None and self.enable_geometric_filtering:
                crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    v_channel = crop_hsv[:, :, 2]
                    deep_dark_ratio = float(np.mean(v_channel < 50))
                    mean_v = float(np.mean(v_channel))
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    lap_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

                    # Skin tone mask in HSV
                    lower_skin1 = np.array([0, 30, 60], dtype=np.uint8)
                    upper_skin1 = np.array([25, 200, 255], dtype=np.uint8)
                    lower_skin2 = np.array([170, 30, 60], dtype=np.uint8)
                    upper_skin2 = np.array([180, 200, 255], dtype=np.uint8)
                    mask_skin = cv2.inRange(crop_hsv, lower_skin1, upper_skin1) | cv2.inRange(crop_hsv, lower_skin2, upper_skin2)
                    skin_ratio = float(np.mean(mask_skin > 0))

                    if cname == "handgun" and deep_dark_ratio < 0.012 and mean_v > 115.0:
                        rejection_reason = f"TRANSLUCENT_WALL_SHADOW (deep_dark={deep_dark_ratio:.1%} < 1.2% on bright surface)"
                    elif cname == "handgun" and skin_ratio >= 0.55:
                        rejection_reason = f"BARE_HUMAN_LIMB (skin={skin_ratio:.1%} >= 55% empty hand/arm)"
                    elif lap_var < 18.0 and mean_v < 85.0:
                        rejection_reason = f"SMOOTH_SHADOW_OR_FABRIC_TRAP (laplacian={lap_var:.1f} < 18.0 on dark gradient)"
                    elif cname == "handgun":
                        bright_bg_ratio = float(np.mean((v_channel > 130) & (crop_hsv[:, :, 1] < 40)))
                        if bright_bg_ratio > 0.48 and deep_dark_ratio < 0.25 and lap_var < 150.0:
                            rejection_reason = f"EMPTY_SILHOUETTE_BACKGROUND_TRAP (bright_bg={bright_bg_ratio:.1%} > 48% on dangling limb)"

            # Human proximity assessment & closest person association
            is_adjacent_to_person = False
            is_held_by_person = False
            min_edge_dist = float("inf")
            associated_person = None
            allowed_reach = self.max_edge_distance

            if persons:
                best_overlapping_person = None
                max_overlap = 0.0
                closest_person = None
                min_edge_dist = float("inf")
                closest_person_reach = self.max_edge_distance

                for p in persons:
                    pbox = p["box"]
                    p_h = max(1, pbox[3] - pbox[1])
                    person_reach = max(self.max_edge_distance, 0.35 * p_h)
                    dist = box_edge_distance(wbox, pbox)
                    overlap = box_overlap_area(wbox, pbox)

                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_overlapping_person = p

                    if dist < min_edge_dist:
                        min_edge_dist = dist
                        closest_person = p
                        closest_person_reach = person_reach

                if best_overlapping_person is not None:
                    associated_person = best_overlapping_person
                    is_adjacent_to_person = True
                    pbox = associated_person["box"]
                    p_w = max(1, pbox[2] - pbox[0])
                    p_h = max(1, pbox[3] - pbox[1])
                    allowed_reach = max(self.max_edge_distance, 0.35 * p_h)
                    rel_y = (cy - pbox[1]) / max(1, p_h)
                    is_bottom_truncated = (pbox[3] >= h - 40)
                    effective_max_reach_y = 1.15 if is_bottom_truncated else self.max_person_reach_y
                    # Handheld objects must fall within realistic arm/hand manipulation envelope.
                    # Genuine overhead weapon: bottom of weapon is within 22% of head, horizontally aligned with torso
                    is_overhead = (
                        (wbox[3] <= pbox[1] + 0.12 * p_h)
                        and (wbox[3] >= pbox[1] - 0.22 * p_h)
                        and (pbox[0] - 0.20 * p_w <= cx <= pbox[2] + 0.20 * p_w)
                    )
                    if (self.min_person_reach_y <= rel_y <= effective_max_reach_y) or is_overhead:
                        is_held_by_person = True
                elif closest_person is not None:
                    associated_person = closest_person
                    allowed_reach = closest_person_reach
                    if min_edge_dist <= allowed_reach:
                        is_adjacent_to_person = True
                        pbox = associated_person["box"]
                        p_w = max(1, pbox[2] - pbox[0])
                        p_h = max(1, pbox[3] - pbox[1])
                        rel_y = (cy - pbox[1]) / max(1, p_h)
                        is_bottom_truncated = (pbox[3] >= h - 40)
                        effective_max_reach_y = 1.15 if is_bottom_truncated else self.max_person_reach_y
                        is_overhead = (
                            (wbox[3] <= pbox[1] + 0.12 * p_h)
                            and (wbox[3] >= pbox[1] - 0.22 * p_h)
                            and (pbox[0] - 0.20 * p_w <= cx <= pbox[2] + 0.20 * p_w)
                        )
                        # Physical detachment rule: If the object does NOT overlap the person box (overlap == 0)
                        # and has had negligible displacement (< 8.0px), it is a stationary fixture (e.g. POS terminal on counter)
                        # and cannot be held by the person.
                        is_moving = det.get("displacement", 0.0) >= 8.0
                        if ((self.min_person_reach_y <= rel_y <= effective_max_reach_y) or is_overhead) and (overlap > 0 or is_moving or min_edge_dist <= 15.0):
                            is_held_by_person = True

            # Check 3: Spatial person presence & proximity gating
            if rejection_reason is None and self.enable_person_gating:
                if not persons:
                    rejection_reason = "NO_PERSON_IN_SCENE"
                elif not is_adjacent_to_person:
                    rejection_reason = f"NO_PERSON_PROXIMITY (edge_dist={min_edge_dist:.1f}px > {allowed_reach:.1f}px)"

            # Check 3b: Anthropometric Scale & Feasibility Gating relative to associated human
            # Perspective-safe normalization: Use effective body dimension to handle top-down cameras
            if rejection_reason is None and self.enable_anthropometric_gating and associated_person is not None:
                pbox = associated_person["box"]
                p_w = max(1, pbox[2] - pbox[0])
                p_h = max(1, pbox[3] - pbox[1])
                p_area = float(p_w * p_h)

                # Perspective safety: In steep top-down cameras, person_h is compressed.
                effective_person_dim = max(p_h, p_w)

                p_area_ratio = box_area / p_area
                p_w_ratio = box_w / p_w
                p_h_ratio = box_h / effective_person_dim
                rel_y = (cy - pbox[1]) / p_h

                # If person box touches bottom frame boundary (camera cuts off legs/feet),
                # waist/hip level weapons are naturally close to or slightly beyond pbox bottom.
                is_bottom_truncated = (pbox[3] >= h - 40)
                effective_max_reach_y = 1.15 if is_bottom_truncated else self.max_person_reach_y

                base_max_h = (
                    self.max_person_height_ratios.get(cname, self.max_person_height_ratio)
                    if hasattr(self, "max_person_height_ratios") and isinstance(self.max_person_height_ratios, dict)
                    else self.max_person_height_ratio
                )
                # Stationary fixtures (e.g. stair posts) strictly capped at base_max_h (0.35 for knife, 0.25 for handgun).
                # Handheld weapons in dynamic motion (displacement >= 8.0px) can reach up to 45% for long blades and rifles.
                w_disp = det.get("displacement", 0.0)
                is_moving = w_disp >= 8.0
                max_h_ratio = min(0.45, base_max_h * 1.25) if is_moving else base_max_h

                if is_bottom_truncated:
                    max_h_ratio = min(0.55, max_h_ratio * 1.35)
                    p_area_limit = self.max_person_area_ratio * 1.35
                else:
                    p_area_limit = self.max_person_area_ratio

                if p_area_ratio > p_area_limit:
                    rejection_reason = f"ANTHROPOMETRIC_SCALE_VIOLATION (area {p_area_ratio:.1%} > {p_area_limit:.1%} of person)"
                elif p_w_ratio > self.max_person_width_ratio:
                    rejection_reason = f"ANTHROPOMETRIC_SCALE_VIOLATION (width {p_w_ratio:.1%} > {self.max_person_width_ratio:.1%} of person)"
                elif p_h_ratio > max_h_ratio:
                    rejection_reason = f"ANTHROPOMETRIC_SCALE_VIOLATION (height {p_h_ratio:.1%} > {max_h_ratio:.1%} of person)"
                # Genuine overhead weapon: bottom of weapon box must be near or above top of head (wbox[3] <= pbox[1] + 0.12 * p_h)
                is_overhead = (
                    (wbox[3] <= pbox[1] + 0.12 * p_h)
                    and (wbox[3] >= pbox[1] - 0.22 * p_h)
                    and (pbox[0] - 0.20 * p_w <= cx <= pbox[2] + 0.20 * p_w)
                )
                if rel_y < self.min_person_reach_y and not is_overhead:
                    rejection_reason = f"UNPHYSICAL_PERSON_LOCATION (rel_y={rel_y:.2f} head/hair/hat region)"
                elif rel_y > effective_max_reach_y:
                    rejection_reason = f"UNPHYSICAL_PERSON_LOCATION (rel_y={rel_y:.2f} below feet)"

                # Architectural vertical post/column suppression (e.g. stair support poles, door frames):
                # An object cannot be a tall vertical column (height > 130px, aspect_ratio < 0.52) reaching near the floor
                # (wbox[3] >= pbox[3] - 0.25 * p_h) while the person is upright.
                if rejection_reason is None and cname in ["knife", "handgun"]:
                    aspect_ratio = box_w / max(1, box_h)
                    if box_h > 130 and aspect_ratio < 0.52 and wbox[3] >= pbox[3] - 0.25 * p_h:
                        rejection_reason = f"UNPHYSICAL_ARCHITECTURAL_COLUMN (vertical post h={box_h}px, AR={aspect_ratio:.2f})"

                # Far-away torso accessory / cross-body sling-bag discrimination:
                # When a person is distant/mid-distance (p_h <= 320px), dark pouches/bags resting on the chest/torso/hip
                # without arm extension are sling bags or pouches, not handheld weapons.
                if rejection_reason is None and cname == "handgun" and p_h <= 320:
                    torso_x_center = (cx - pbox[0]) / max(1, p_w)
                    if 0.05 <= torso_x_center <= 1.15 and 0.20 <= rel_y <= 0.70:
                        rejection_reason = f"FAR_FIELD_TORSO_ACCESSORY (torso pouch rel_x={torso_x_center:.2f}, rel_y={rel_y:.2f} on distant person)"

            # Check 4: Static environmental trap suppression
            # If an object is static, it is suppressed UNLESS it is physically held by a person (e.g. suspect aiming steadily).
            if rejection_reason is None and self.enable_motion_filtering:
                if det.get("is_static", False):
                    if not is_held_by_person:
                        tf = det.get("track_frames", 0)
                        disp = det.get("displacement", 0.0)
                        rejection_reason = f"STATIC_BACKGROUND_TRAP ({tf} frames, drift={disp}px)"
                        # Register in persistent exclusion memory for unheld fixtures (e.g. counter POS terminals, railings)
                        if self.motion_tracker and (not is_adjacent_to_person or not is_held_by_person):
                            self.motion_tracker.register_static_zone(cx, cy, cname)

            eval_entry = {
                **det,
                "raw_class_name": det.get("class_name"),
                "raw_confidence": det.get("confidence"),
                "persons_in_frame": len(persons),
            }

            # Check 5: Temporal Consistency
            if rejection_reason is None and self.temporal_filter:
                self.temporal_filter.update([eval_entry], frame_idx)
                is_consistent = eval_entry.get("is_temporally_consistent", True)
                hits = eval_entry.get("temporal_hits", 1)
                t_id = eval_entry.get("temporal_track_id", 0)
                eval_entry["temporal_track_id"] = t_id
                eval_entry["temporal_hits"] = hits

                if self.min_temporal_hits > 1 and hits < self.min_temporal_hits and not is_consistent:
                    # In online processing, first hit is pending temporal confirmation
                    # UNLESS it is verified as an ultra-high confidence in-hand threat (is_consistent = True)
                    status = "PENDING_CONFIRMATION"
                    validation_status = "PENDING_TEMPORAL"
                else:
                    status = "CONFIRMED_ALERT"
                    validation_status = "VALIDATED_TEMPORAL"
            elif rejection_reason is None:
                status = "CONFIRMED_ALERT"
                validation_status = "CONFIRMED_ALERT"
            else:
                status = "SUPPRESSED"
                validation_status = rejection_reason.split()[0]

            eval_entry["status"] = status
            eval_entry["validation_status"] = validation_status
            eval_entry["rejection_reason"] = rejection_reason

            all_evaluated.append(eval_entry)

            if status == "CONFIRMED_ALERT":
                confirmed.append(eval_entry)

        return confirmed, all_evaluated

    def reconcile_video_records(
        self,
        records: List[Dict[str, Any]],
        min_hits: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Post-processes all video records to reconcile temporal consistency across full video.
        
        1. If a tracklet achieved >= min_hits total detections across the video, its initial
           frames (previously marked PENDING_CONFIRMATION) are elevated to CONFIRMED_ALERT.
        2. If an in-hand weapon achieved ultra-high confidence (>= 0.85), it is elevated
           to prevent brief fast-draw weapon presentations from being lost.
        3. If a tracklet had fewer than min_hits total detections (isolated 1-frame transient flicker),
           it is permanently marked SUPPRESSED with rejection_reason 'SUPPRESSED_TEMPORAL_FLICKER'.
        """
        required_hits = min_hits if min_hits is not None else self.min_temporal_hits
        if required_hits <= 1:
            return records

        # If any detection in a tracklet was confirmed as STATIC_BACKGROUND_TRAP,
        # then all earlier unconfirmed frames in that stationary tracklet are also static background traps.
        static_track_ids = set()
        for r in records:
            t_id = r.get("temporal_track_id")
            if t_id is not None and "STATIC_BACKGROUND_TRAP" in str(r.get("rejection_reason", "")):
                static_track_ids.add(t_id)

        for r in records:
            t_id = r.get("temporal_track_id")
            if t_id in static_track_ids:
                r["status"] = "SUPPRESSED"
                r["validation_status"] = "STATIC_BACKGROUND_TRAP"
                r["rejection_reason"] = "STATIC_BACKGROUND_TRAP (Stationary environmental fixture)"

        # Group valid proposals by temporal_track_id
        track_totals: Dict[int, int] = {}
        for r in records:
            t_id = r.get("temporal_track_id")
            if t_id is not None and not r.get("rejection_reason"):
                track_totals[t_id] = track_totals.get(t_id, 0) + 1

        for r in records:
            t_id = r.get("temporal_track_id")
            if t_id is not None and not r.get("rejection_reason"):
                total = track_totals.get(t_id, 0)
                conf_score = float(r.get("confidence_score") or r.get("confidence") or 0.0)
                if total >= required_hits:
                    r["status"] = "CONFIRMED_ALERT"
                    r["validation_status"] = "VALIDATED_TEMPORAL"
                else:
                    r["status"] = "SUPPRESSED"
                    r["validation_status"] = "SUPPRESSED_TEMPORAL_FLICKER"
                    r["rejection_reason"] = f"SUPPRESSED_TEMPORAL_FLICKER (Isolated {total}-frame transient proposal)"

        # Track-level weapon class consensus resolution across the video trajectory
        track_class_stats: Dict[Any, Dict[str, Any]] = {}
        for r in records:
            t_id = r.get("temporal_track_id") or r.get("track_id")
            if t_id is not None and not r.get("rejection_reason"):
                cname = r.get("raw_class_name") or r.get("object_label") or r.get("class_name")
                conf = float(r.get("raw_confidence") or r.get("confidence_score") or r.get("confidence") or 0.0)
                if t_id not in track_class_stats:
                    track_class_stats[t_id] = {}
                stats = track_class_stats[t_id]
                if cname not in stats:
                    stats[cname] = {"count": 0, "max_conf": 0.0, "sum_conf": 0.0}
                stats[cname]["count"] += 1
                stats[cname]["max_conf"] = max(stats[cname]["max_conf"], conf)
                stats[cname]["sum_conf"] += conf

        track_consensus: Dict[Any, str] = {}
        for t_id, stats in track_class_stats.items():
            knife_stats = stats.get("knife", {"count": 0, "max_conf": 0.0, "sum_conf": 0.0})
            handgun_stats = stats.get("handgun", {"count": 0, "max_conf": 0.0, "sum_conf": 0.0})

            # Physical consistency rule: In surveillance CCTV, a track's true weapon identity
            # is governed by majority evidence across its trajectory.
            # If handgun detections clearly dominate (count >= 1.5 * knife count or sum_conf >= 1.5 * knife sum_conf),
            # reconcile all detections in that track to handgun.
            if handgun_stats["count"] >= max(2, int(knife_stats["count"] * 1.5)) or (handgun_stats["sum_conf"] > knife_stats["sum_conf"] * 1.5 and handgun_stats["count"] >= 2):
                track_consensus[t_id] = "handgun"
            elif knife_stats["count"] >= max(2, int(handgun_stats["count"] * 1.5)) or (knife_stats["sum_conf"] > handgun_stats["sum_conf"] * 1.5 and knife_stats["count"] >= 2):
                track_consensus[t_id] = "knife"
            elif knife_stats["max_conf"] >= 0.90 and knife_stats["max_conf"] > handgun_stats["max_conf"] + 0.10:
                track_consensus[t_id] = "knife"
            elif handgun_stats["max_conf"] >= 0.90 and handgun_stats["max_conf"] > knife_stats["max_conf"] + 0.10:
                track_consensus[t_id] = "handgun"
            else:
                track_consensus[t_id] = max(stats.keys(), key=lambda c: (stats[c]["count"], stats[c]["sum_conf"]))

        # Apply tracklet consensus strictly to each distinct physical weapon track
        for r in records:
            t_id = r.get("temporal_track_id") or r.get("track_id")
            if t_id in track_consensus and not r.get("rejection_reason"):
                r["object_label"] = track_consensus[t_id]
                r["class_name"] = track_consensus[t_id]

        return records

