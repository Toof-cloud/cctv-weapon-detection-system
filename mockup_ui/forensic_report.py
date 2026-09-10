"""Readable and structured reports from completed, saved detection results."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
from uuid import uuid4
from mockup_ui.observation_review import DETECTION_KEYS, make_observations, validate_observations
from mockup_ui.report_metrics import (
    ENGINE_PATH, METRIC_CSV_NAME, REPORT_DISCLAIMER, calculate_report_metrics, metric_fields,
)

MOCKUP_DIR = Path(__file__).resolve().parent
VALIDATION_STATUS = "pending_human_validation"
TIME_BASIS = "Video-relative offset: frame number / FPS, rounded by the detection pipeline. Not a recording date/time."
CSV_FIELDS = (
    "report_id", "record_id", "source_video_reference", "source_video_name",
    "frame_number", "video_relative_timestamp_seconds", "source_timestamp",
    "source_timestamp_status", "camera_id", "camera_id_status", "object_label",
    "x1", "y1", "x2", "y2", "confidence_score", "validation_status",
    "reviewer", "reviewed_at", "model_reference",
    "observation_id", "analyst_decision", "analyst_notes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_reference(path: Path) -> dict:
    """A current fingerprint is not presented as an ingestion-time fingerprint."""
    reference = {"path": str(path), "sha256": None, "size_bytes": None,
                 "checked_at_utc": utc_now(), "status": "unavailable"}
    try:
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            reference["status"] = "changed_during_report_generation"
        else:
            reference.update(sha256=digest, size_bytes=after.st_size,
                             status="fingerprinted_at_report_generation")
    except OSError:
        pass
    return reference


def _read_verified_summary(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with (summary_path.parent / "detections.csv").open(newline="", encoding="utf-8") as stream:
        original_rows = [{
            "frame_number": int(row["frame_number"]),
            "timestamp_seconds": float(row["timestamp_seconds"]),
            "class_name": row["class_name"], "confidence": float(row["confidence"]),
            "box": [int(row[key]) for key in ("x1", "y1", "x2", "y2")],
        } for row in csv.DictReader(stream)]
    if original_rows != [{key: row[key] for key in DETECTION_KEYS} for row in summary["detections"]]:
        raise ValueError("The saved detection CSV and summary disagree. A report was not generated.")
    return summary


def build_report(summary_path: Path) -> dict:
    summary_path = Path(summary_path).resolve()
    summary = _read_verified_summary(summary_path)
    report_id = f"FR-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    observations = summary.get("observations")
    if observations is None:
        # Stable identifiers for historical exports without a run ID.
        run_id = summary.get("run_id") or hashlib.sha256(summary_path.read_bytes()).hexdigest()[:32]
        observations = make_observations(summary, run_id)
    validate_observations(observations, summary)
    source = Path(summary["source_video"])
    rows = [{
        "report_id": report_id, "record_id": f"{report_id}-D{index:05d}",
        "source_video_reference": str(source), "source_video_name": source.name,
        "frame_number": row["frame_number"],
        "video_relative_timestamp_seconds": row["timestamp_seconds"],
        "source_timestamp": None, "source_timestamp_status": "not_recorded_by_current_pipeline",
        "camera_id": None, "camera_id_status": "not_recorded_by_current_pipeline",
        "object_label": row["class_name"],
        **dict(zip(("x1", "y1", "x2", "y2"), row["box"], strict=True)),
        "confidence_score": row["confidence"], "validation_status": VALIDATION_STATUS,
        "reviewer": None, "reviewed_at": None, "model_reference": summary["model_path"],
    } for index, row in enumerate(summary["detections"], 1)]
    for row, observation in zip(rows, observations, strict=True):
        review = observation["analystReview"]
        row.update(observation_id=observation["observationId"],
                   analyst_decision=review["decision"] if review else None,
                   analyst_notes=review["notes"] if review else "",
                   reviewed_at=review["reviewedAt"] if review else None,
                   validation_status="analyst_reviewed" if review else VALIDATION_STATUS)
    reviewed = sum(row["analystReview"] is not None for row in observations)
    review_summary = f"{reviewed} of {len(observations)} observations reviewed"
    counts_by_decision = dict(Counter(row["analyst_decision"] or "Not reviewed" for row in rows))
    counts = dict(Counter(row["object_label"] for row in rows))
    metric_path = summary_path.parent / METRIC_CSV_NAME
    metrics = calculate_report_metrics(metric_path)
    metrics["temporal_consistency_enabled"] = summary.get("temporal_consistency_enabled")
    metric_videos = metrics["tcr"].get("per_video", [])
    camera_id = metric_videos[0]["camera_id"] if len(metric_videos) == 1 else None
    camera_id = camera_id or None
    camera_status = "recorded_by_pipeline" if camera_id else "not_recorded_by_current_pipeline"
    for row in rows:
        row.update(camera_id=camera_id, camera_id_status=camera_status)
    return {
        "schema_version": "2.0", "report_id": report_id, "report_type": "forensic_detection_report",
        "generated_at_utc": utc_now(), "validation_status": "analyst_reviewed" if observations and reviewed == len(observations) else VALIDATION_STATUS,
        "review_summary": review_summary, "counts_by_analyst_decision": counts_by_decision,
        "metrics": metrics,
        "source": {
            "reference": str(source), "name": source.name,
            "fps": summary["fps"], "duration_seconds": summary["duration_seconds"],
            "width": summary.get("width"), "height": summary.get("height"),
            "frame_count": summary.get("frame_count"),
            "camera_id": camera_id, "camera_id_status": camera_status,
            "recording_start_timestamp": None,
            "source_timestamp_status": "not_recorded_by_current_pipeline",
            "file_at_report_generation": file_reference(source),
        },
        "processing": {
            "status": "completed_saved_run", "model_reference": summary["model_path"],
            "device": summary["device"], "confidence_threshold": summary["confidence_threshold"],
            "analyzed_frames": summary["analyzed_frames"], "elapsed_seconds": summary["elapsed_seconds"],
            "completed_at_utc": summary.get("completed_at_utc"),
        },
        "summary": {"total_frame_detections": len(rows), "counts_per_class": counts,
                    "frames_with_detections": len({row["frame_number"] for row in rows})},
        "definitions": {
            "frame_numbering": "Zero-based: the first frame is frame 0.",
            "timestamp_basis": TIME_BASIS,
            "box_format": "[x1, y1, x2, y2] in source-frame pixels, origin at the upper-left corner.",
            "count_definition": "Per-frame observations; the same weapon can appear in multiple frames. These are not unique-object counts.",
            "validation_status": "Analyst Review Decision records the human assessment. Confidence and processing success never select an analyst decision.",
            "missing_metadata": "Camera identifiers are reported when supplied by the pipeline. Original recording timestamps are not collected and are not inferred from filenames or filesystem dates.",
            "fingerprints": "SHA-256 values identify files read when this report was generated; no ingestion-time hash or chain-of-custody history is asserted.",
        },
        "artifacts": {"saved_summary": file_reference(summary_path),
                      "metric_input": file_reference(metric_path),
                      "metric_engine": file_reference(ENGINE_PATH),
                      "pipeline_detections": file_reference(summary_path.parent / "detections.csv"),
                      "annotated_video": str(summary_path.parent / "annotated.mp4")},
        "detections": rows,
    }


def render_html(report: dict) -> str:
    def e(value):
        return escape("Not recorded" if value is None else str(value), quote=True)

    def fields(values):
        return "<dl>" + "".join(f"<dt>{e(label)}</dt><dd>{e(value)}</dd>" for label, value in values) + "</dl>"

    source, processing, summary = report["source"], report["processing"], report["summary"]
    counts = summary["counts_per_class"]
    dimensions = f"{source['width']} x {source['height']} pixels" if source["width"] and source["height"] else None
    class_summary = "; ".join(f"{name.title()}: {count}" for name, count in sorted(counts.items())) or "None"
    decision_summary = "; ".join(
        f"{decision}: {count}" for decision, count in sorted(report["counts_by_analyst_decision"].items())
    ) or "No observations"
    table_rows = "".join(
        f"<tr><td>{e(row['observation_id'])}</td><td>{e(row['frame_number'])}</td>"
        f"<td>{row['video_relative_timestamp_seconds']:.2f}</td><td>{e(row['object_label'].title())}</td>"
        f"<td>[{row['x1']}, {row['y1']}, {row['x2']}, {row['y2']}]</td>"
        f"<td>{row['confidence_score']:.2%}</td>"
        f"<td>{e(row['analyst_decision'] or 'Not reviewed')}</td></tr>"
        for row in report["detections"]
    ) or '<tr><td colspan="7">No handgun or knife observations met the configured confidence threshold.</td></tr>'
    review_details = "".join(
        f"<h3>Observation {e(row['observation_id'])}</h3>" + fields([
            ("Analyst Review Decision", row["analyst_decision"] or "Not reviewed"),
            ("Analyst notes", row["analyst_notes"] or "No notes provided"),
            ("Review timestamp (UTC)", row["reviewed_at"]),
        ]) for row in report["detections"]
    ) or "<p>No observations are available for analyst review.</p>"
    artifacts = report["artifacts"]
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forensic detection report — {e(source['name'])}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf0f5;color:#222a39;font:14px/1.55 Arial,sans-serif}}
main{{max-width:1080px;margin:30px auto;background:white;padding:40px 44px;border-top:7px solid #2f318e}}
header{{border-bottom:1px solid #dce0e8;padding-bottom:22px}}.brand{{color:#2f318e;letter-spacing:2px;font-size:12px;font-weight:bold}}
h1{{font-size:30px;line-height:1.2;margin:9px 0}}h2{{font-size:17px;margin:28px 0 12px}}p{{margin:8px 0}}
.muted{{color:#596577}}.badge{{display:inline-block;background:#fff2d8;color:#775014;padding:5px 10px;border-radius:5px}}
.disclaimer{{margin:20px 0;background:#fff4da;border:1px solid #d8b45c;color:#493614;padding:12px 14px;font-weight:bold}}
.metrics{{display:flex;gap:12px;margin-top:24px}}.metric{{flex:1;background:#f4f5fa;padding:16px;border-radius:6px}}
.metric strong{{display:block;font-size:27px;color:#2f318e}}.metric span{{font-size:12px;color:#596577}}
dl{{display:grid;grid-template-columns:185px minmax(0,1fr);margin:0;border-top:1px solid #e0e4ed}}
dt,dd{{margin:0;padding:8px 10px;border-bottom:1px solid #e0e4ed;overflow-wrap:anywhere}}dt{{font-weight:bold;background:#f7f8fb}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th{{background:#303841;color:white;text-align:left}}
th,td{{padding:9px 8px;border:1px solid #e0e4ed;vertical-align:top}}tr:nth-child(even){{background:#f7f8fb}}
li{{margin:5px 0}}footer{{margin-top:28px;border-top:1px solid #dce0e8;padding-top:14px;font-size:12px;color:#596577}}
@media(max-width:700px){{main{{margin:0;padding:24px 18px}}dl{{grid-template-columns:140px minmax(0,1fr)}}.metrics{{flex-wrap:wrap}}.metric{{min-width:40%}}}}
@media print{{body{{background:white}}main{{margin:0;padding:0;max-width:none;border:0}}thead{{display:table-header-group}}tr,dl,.metrics{{break-inside:avoid}}h2{{break-after:avoid}}.table-wrap{{overflow:visible}}@page{{size:A4 landscape;margin:15mm}}}}
</style></head><body><main>
<header><div class="brand">FORENSIKADA / ANALYSIS RECORD</div><h1>Forensic detection report</h1>
<p class="muted">{e(source['name'])}</p><span class="badge">{e(report['review_summary'])}</span>
<p class="muted">Report {e(report['report_id'])}<br>Generated (UTC): {e(report['generated_at_utc'])}</p></header>
<div class="disclaimer">{e(REPORT_DISCLAIMER)}</div>
<h2>Video Metadata</h2>
{fields([('Video filename', source['name']), ('Resolution', dimensions), ('Frame rate', f"{source['fps']:g} FPS"), ('Duration', f"{source['duration_seconds']:.3f} seconds"), ('Source frame count', source['frame_count']), ('Camera identifier', source['camera_id']), ('Recording date/time', source['recording_start_timestamp'])])}
<h2>Video and Detection Information</h2>
<div class="metrics"><div class="metric"><strong>{summary['total_frame_detections']}</strong><span>Frame detections</span></div>
<div class="metric"><strong>{counts.get('knife', 0)}</strong><span>Knife observations</span></div>
<div class="metric"><strong>{counts.get('handgun', 0)}</strong><span>Handgun observations</span></div>
<div class="metric"><strong>{processing['analyzed_frames']}</strong><span>Frames analyzed</span></div></div>
{fields([('Model used', processing['model_reference']), ('Run status', 'Completed saved run'), ('Confidence threshold', f"{processing['confidence_threshold']:.0%}"), ('Frames analyzed', processing['analyzed_frames']), ('Processing device / time', f"{processing['device'].upper()} / {processing['elapsed_seconds']:.2f} seconds"), ('Analysis completed (UTC)', processing['completed_at_utc']), ('Detection summary', f"{summary['total_frame_detections']} observations; {summary['frames_with_detections']} positive frames; {class_summary}"), ('Analyst review coverage', report['review_summary'])])}
<h2>Interpretation</h2>
<p>The model generated the listed handgun and knife observations at the configured confidence threshold. A confidence score describes model certainty and is not an analyst decision. Analyst decisions document a later human assessment and do not replace or delete the original model observation.</p>
<p>Frame numbers are zero-based. Times are video-relative offsets and are not recording dates. Bounding boxes use [x1, y1, x2, y2] source-frame pixel coordinates. Counts represent frame observations, so the same physical object may appear more than once.</p>
<h2>Object Detection Observations and Reviews</h2>
<div class="table-wrap"><table><thead><tr><th>Observation</th><th>Frame (0-based)</th><th>Video offset (s)</th><th>Object label</th><th>Box [x1, y1, x2, y2]</th><th>Confidence</th><th>Analyst Review Decision</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<h2>TCR Information</h2>
{fields(metric_fields(report.get("metrics", {}), "tcr"))}
<h2>MCCR Information</h2>
{fields(metric_fields(report.get("metrics", {}), "mccr"))}
<h2>Analyst Review Information</h2>
{fields([('Review coverage', report['review_summary']), ('Decision totals', decision_summary)])}
{review_details}
<h2>Source References</h2>
{fields([('Source video', source['reference']), ('Model', processing['model_reference']), ('Annotated video', artifacts['annotated_video']), ('Saved summary', artifacts['saved_summary']['path']), ('Pipeline detections', artifacts['pipeline_detections']['path']), ('Metric input records', artifacts['metric_input']['path']), ('Metric calculation script', artifacts['metric_engine']['path'])])}
<h2>Traceability Report</h2>
{fields([('Report identifier', report['report_id']), ('Generated (UTC)', report['generated_at_utc']), ('Source video fingerprint', source['file_at_report_generation']['sha256']), ('Source video check', source['file_at_report_generation']['status']), ('Saved summary fingerprint', artifacts['saved_summary']['sha256']), ('Saved summary check', artifacts['saved_summary']['status']), ('Pipeline detections fingerprint', artifacts['pipeline_detections']['sha256']), ('Pipeline detections check', artifacts['pipeline_detections']['status']), ('Metric input fingerprint', artifacts['metric_input']['sha256']), ('Metric input check', artifacts['metric_input']['status']), ('Metric script fingerprint', artifacts['metric_engine']['sha256'])])}
<p>{e(report['definitions']['fingerprints'])}</p>
<footer>Generated from saved detection records without rerunning or relabeling the model output.
Companion files: forensic_report.json and forensic_records.csv. Use your browser's Print command to print or save this report as PDF.</footer>
</main></body></html>'''


def write_forensic_report(summary_path: Path, destination: Path | None = None) -> Path:
    """Write once per destination; original video, CSV and summary stay untouched."""
    summary_path = Path(summary_path).resolve()
    report = build_report(summary_path)
    destination = Path(destination).resolve() if destination else summary_path.parent / report["report_id"]
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / name for name in ("forensic_report.html", "forensic_report.json", "forensic_records.csv", "forensic_report.pdf")]
    if any(path.exists() for path in paths):
        raise FileExistsError("A forensic report already exists in this folder. Choose a new destination.")
    paths[0].write_text(render_html(report), encoding="utf-8")
    paths[1].write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with paths[2].open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(report["detections"])
    from mockup_ui.pdf_report import write_pdf
    write_pdf(report, paths[3])
    return paths[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a forensic report from a saved mockup summary.json.")
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    print(write_forensic_report(args.summary))
