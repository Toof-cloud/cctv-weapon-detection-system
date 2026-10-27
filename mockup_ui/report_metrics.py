"""Report adapter for the project's benchmark calculations, using unfiltered run records."""
from collections import defaultdict
import csv
import math
from pathlib import Path
import tempfile

from benchmarks.calculate_tcr_and_mccr import compute_mccr_for_multicam, compute_tcr_for_csv

ENGINE_PATH = Path(__file__).resolve().parents[1] / "benchmarks/calculate_tcr_and_mccr.py"
METRIC_CSV_NAME = "metric_input.csv"
REPORT_DISCLAIMER = (
    "This is a system-generated report. All detections and validation results are subject to "
    "human analyst review and should be treated as reviewable observations, not conclusive findings."
)


def unavailable(reason):
    return {"status": "unavailable", "value_percent": None, "reason": reason}


def calculate_report_metrics(csv_path: Path) -> dict:
    """Never substitute analyst decisions or previously filtered UI observations."""
    missing = "Unfiltered pipeline records were not saved for this run. Process the video again to calculate this metric."
    result = {"tcr": unavailable(missing), "mccr": unavailable(missing)}
    if not csv_path.is_file():
        return result
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return {key: unavailable("No detection observations are available; no percentage can be calculated.") for key in result}
    required = {"frame_number", "object_label", "validation_status"}
    if not required <= columns:
        return {key: unavailable("Pipeline CSV lacks required automated validation fields.") for key in result}
    if any(not r["frame_number"].isdigit() or not r["object_label"] or not r["validation_status"] for r in rows):
        return {key: unavailable("Pipeline records contain missing labels, statuses, or invalid frame numbers.") for key in result}

    # Each video's frame numbers have their own origin; never combine them for TCR.
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("camera_id", ""), row.get("source_video", ""))].append(row)
    per_video = []
    with tempfile.TemporaryDirectory(prefix="forensikada-metrics-") as folder:
        for index, ((camera, source), records) in enumerate(sorted(groups.items())):
            path = Path(folder) / f"video-{index}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(records)
            counts = compute_tcr_for_csv(path)
            counts.pop("csv_name", None)
            per_video.append({"camera_id": camera, "source_video": source, **counts})
    totals = {key: sum(video[key] for video in per_video) for key in
              ("n_ts", "n_isolated", "n_interrupted", "n_not_evaluable", "n_te")}
    if len(per_video) == 1:
        result["tcr"] = {"status": "computed" if totals["n_te"] else "unavailable",
                         "value_percent": per_video[0]["tcr"] if totals["n_te"] else None,
                         "reason": "Calculated from saved pipeline statuses." if totals["n_te"] else "No temporally eligible observations (N_TE = 0).",
                         **totals, "per_video": per_video}
    else:
        result["tcr"] = {"status": "per_video", "value_percent": None,
                         "reason": "TCR is calculated separately for each source video and camera.", "per_video": per_video}

    cameras = {r.get("camera_id") for r in rows if r.get("camera_id")}
    if len(cameras) < 2:
        result["mccr"] = unavailable("Requires two camera feeds with camera identifiers and comparable timestamps. This run has fewer than two recorded cameras.")
        return result
    if len(cameras) != 2 or len(groups) != 2:
        result["mccr"] = unavailable("This benchmark requires a two-camera comparison with one source video per camera.")
        return result
    if not {"timestamp_seconds", "confidence_score", "camera_id", "source_video"} <= columns:
        result["mccr"] = unavailable("Camera identifiers, source references, timestamps, and confidence scores are required.")
        return result
    try:
        valid = all(r["camera_id"] and r["source_video"] and
                    math.isfinite(float(r["timestamp_seconds"])) and float(r["timestamp_seconds"]) >= 0 and
                    math.isfinite(float(r["confidence_score"])) and 0 <= float(r["confidence_score"]) <= 1 for r in rows)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        result["mccr"] = unavailable("Missing camera/source references or invalid timestamps/confidence scores.")
        return result
    counts = compute_mccr_for_multicam(csv_path, delta_t=1.5)
    result["mccr"] = {"status": "computed" if counts["n_mc"] else "unavailable",
                      "value_percent": counts["mccr"] if counts["n_mc"] else None,
                      "reason": "Calculated under the benchmark's aligned-start assumption; synchronization is not independently verified." if counts["n_mc"] else "No eligible cross-camera observations (N_MC = 0).",
                      **counts}
    return result


def metric_fields(metrics: dict, name: str) -> list[tuple[str, object]]:
    metric = metrics.get(name, unavailable("Metric was not calculated for this report."))
    value = metric["value_percent"]
    rows = [("Metric", "Temporal Consistency Rate (TCR)" if name == "tcr" else "Multi-Camera Corroboration Rate (MCCR)"),
            ("Result", f"{value:.2f}%" if value is not None else ("See per-video results" if metric["status"] == "per_video" else "N/A")),
            ("Interpretation", metric["reason"])]
    if name == "tcr":
        count_labels = [("n_ts", "Supported observations (N_TS)"), ("n_isolated", "Isolated observations"),
                        ("n_interrupted", "Interruptions"), ("n_not_evaluable", "Boundary observations excluded"),
                        ("n_te", "Eligible total (N_TE)")]
        rows.extend((label, metric[key]) for key, label in count_labels if key in metric)
        if metric["status"] == "per_video":
            for item in metric["per_video"]:
                score = f"{item['tcr']:.2f}%" if item["n_te"] else "N/A"
                rows.append((f"{item['camera_id']} / {item['source_video']}",
                             f"{score}; supported {item['n_ts']}; isolated {item['n_isolated']}; "
                             f"interruptions {item['n_interrupted']}; boundary excluded {item['n_not_evaluable']}; N_TE {item['n_te']}"))
    else:
        if "camera_a" in metric:
            rows.append(("Compared cameras", f"{metric['camera_a']} and {metric['camera_b']}"))
        rows.extend((label, metric[key]) for key, label in (
            ("n_cc", "Corroborated observations (N_CC)"), ("not_corroborated", "Not corroborated"),
            ("uncertain", "Uncertain / conflicting labels"), ("not_applicable", "Outside concurrent window"),
            ("n_mc", "Eligible observations (N_MC)"), ("concurrent_interval", "Concurrent comparison window")) if key in metric)
    if metrics.get("temporal_consistency_enabled") is False and name == "tcr":
        rows.append(("Run configuration", "Temporal filtering was disabled. This score reflects the saved alert statuses and does not demonstrate temporal validation."))
    return rows
