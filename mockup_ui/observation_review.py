"""Persistent analyst decisions, kept separate from immutable detection records."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

REVIEW_DIR = Path(__file__).resolve().parent / "reviews"
DECISIONS = ("Accept", "Reject", "Uncertain")
NOT_PERFORMED = "not_performed"
DETECTION_KEYS = ("frame_number", "timestamp_seconds", "class_name", "confidence", "box")


def validate_review(decision, notes):
    if decision not in DECISIONS:
        raise ValueError("Please select Accept, Reject, or Uncertain before completing the review.")
    notes = notes.strip()
    if decision == "Reject" and not notes:
        raise ValueError("A reason is required when rejecting an observation.")
    return notes


def status_label(status):
    return "Not performed" if status == NOT_PERFORMED else str(status)


def make_observations(summary, run_id):
    source = summary["source_video"]
    return [{
        "observationId": f"{run_id}-OBS-{index:05d}",
        "automatedValidationStatus": row.get("automated_validation_status", NOT_PERFORMED),
        "analystReview": None,
        "sourceReferences": [{"videoPath": source, "modelPath": summary["model_path"],
                              "frameNumber": row["frame_number"], "videoOffsetSeconds": row["timestamp_seconds"],
                              "cameraId": None, "sourceTimestamp": None}],
        "detectionDetails": deepcopy(row),
    } for index, row in enumerate(summary["detections"], 1)]


def validate_observations(observations, summary):
    if len(observations) != len(summary["detections"]):
        raise ValueError("Review observations do not match this analysis.")
    identifiers = set()
    for observation, detection in zip(observations, summary["detections"], strict=True):
        identifier = observation["observationId"]
        if not identifier or identifier in identifiers:
            raise ValueError("Observation IDs must be unique.")
        identifiers.add(identifier)
        if observation["detectionDetails"] != detection:
            raise ValueError("Review detection details disagree with the original analysis.")
        references = observation["sourceReferences"]
        if not references or references[0]["videoPath"] != summary["source_video"] or references[0]["modelPath"] != summary["model_path"]:
            raise ValueError("Review source references disagree with the original analysis.")
        if (references[0]["frameNumber"], references[0]["videoOffsetSeconds"]) != (detection["frame_number"], detection["timestamp_seconds"]):
            raise ValueError("Review frame references disagree with the original analysis.")
        if observation["automatedValidationStatus"] != detection.get("automated_validation_status", NOT_PERFORMED):
            raise ValueError("The automated validation result was changed.")
        review = observation["analystReview"]
        if review is not None:
            validate_review(review["decision"], review["notes"])
            datetime.fromisoformat(review["reviewedAt"])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".review-", suffix=".tmp", delete=False) as stream:
            name = stream.name
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


class ReviewStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.reload()

    @classmethod
    def for_result(cls, result):
        from mockup_ui.model_bridge import result_summary
        path = result.review_path or REVIEW_DIR / f"{result.run_id}.json"
        path = Path(path)
        summary = result_summary(result)
        if not path.exists():
            atomic_json(path, {"schemaVersion": 1, "analysis": summary,
                               "artifacts": {"annotatedVideo": str(result.output_path), "detectionsCsv": str(result.csv_path)},
                               "observations": make_observations(summary, result.run_id)})
        store = cls(path)
        if store.data["analysis"] != summary:
            raise ValueError("Saved reviews belong to a different analysis. Reopen the matching review file.")
        result.review_path = path
        return store

    def reload(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schemaVersion") != 1:
            raise ValueError("Unsupported observation review file.")
        validate_observations(data["observations"], data["analysis"])
        self.data = data
        return self.data

    @property
    def observations(self):
        return deepcopy(self.data["observations"])

    def save_review(self, observation_id, decision, notes):
        notes = validate_review(decision, notes)
        self.reload()
        updated = deepcopy(self.data)
        observation = next((row for row in updated["observations"] if row["observationId"] == observation_id), None)
        if observation is None:
            raise ValueError("The selected observation is not part of this analysis.")
        review = {"decision": decision, "notes": notes,
                  "reviewedAt": datetime.now(timezone.utc).isoformat(timespec="microseconds")}
        observation["analystReview"] = review
        observation.setdefault("reviewHistory", []).append(deepcopy(review))
        atomic_json(self.path, updated)
        self.data = updated
        return deepcopy(review)

    def restore_result(self):
        from mockup_ui.model_bridge import VideoAnalysisResult, VideoInfo
        summary = self.data["analysis"]
        video = VideoInfo(Path(summary["source_video"]), summary["width"], summary["height"],
                          summary["fps"], summary["frame_count"], summary["duration_seconds"], None,
                          summary.get("source_size_bytes", 0), summary.get("source_modified_ns", 0))
        return VideoAnalysisResult(video, Path(self.data["artifacts"]["annotatedVideo"]),
                                   Path(self.data["artifacts"]["detectionsCsv"]), deepcopy(summary["detections"]),
                                   summary["model_path"], summary["device"], summary["elapsed_seconds"],
                                   summary["confidence_threshold"], summary["analyzed_frames"],
                                   summary["completed_at_utc"], summary["run_id"], self.path)
