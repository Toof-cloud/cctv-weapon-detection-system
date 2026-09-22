"""Multi-camera session forensic report generation and dialog.

Computes both Temporal Consistency Rate (TCR) per camera and Multi-Camera Corroboration Rate (MCCR),
renders comprehensive HTML and ReportLab PDF documents, and provides an in-app viewer.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)

from benchmarks.calculate_tcr_and_mccr import compute_tcr_for_csv
from mockup_ui.observation_review import ReviewStore
from mockup_ui.model_bridge import timecode

REPORT_DISCLAIMER = (
    "This is a system-generated report. All detections and validation results are subject to "
    "human analyst review and should be treated as reviewable observations, not conclusive findings."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_digest(path: Path) -> str:
    try:
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except Exception:
        return "Unavailable"


def compute_tcr_from_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fallback TCR calculation when metric_input.csv is not present (e.g. synthetic fixtures)."""
    if not detections:
        return {"n_ts": 0, "n_isolated": 0, "n_interrupted": 0, "n_not_evaluable": 0, "n_te": 0, "tcr": None}
    
    frame_nums = [int(d.get("frame_number", 0)) for d in detections]
    min_frame = min(frame_nums) if frame_nums else 0
    max_frame = max(frame_nums) if frame_nums else 0
    
    supported = 0
    isolated = 0
    not_eval = 0
    interrupted = 0
    
    track_frames: Dict[str, List[int]] = {}
    for d in detections:
        fn = int(d.get("frame_number", 0))
        status = d.get("automated_validation_status", "")
        label = d.get("class_name", d.get("object_label", "weapon"))
        
        if fn == min_frame or fn == max_frame:
            not_eval += 1
            continue
        if status in ("VALIDATED_TEMPORAL", "CONFIRMED_ALERT"):
            supported += 1
            track_frames.setdefault(label, []).append(fn)
        elif status == "SUPPRESSED_TEMPORAL_FLICKER":
            isolated += 1

    for lbl, f_list in track_frames.items():
        f_sorted = sorted(f_list)
        for i in range(len(f_sorted) - 1):
            if f_sorted[i + 1] - f_sorted[i] > 15:
                interrupted += 1

    n_te = supported + isolated + interrupted
    tcr = round(100.0 * supported / n_te, 2) if n_te > 0 else None
    return {
        "n_ts": supported,
        "n_isolated": isolated,
        "n_interrupted": interrupted,
        "n_not_evaluable": not_eval,
        "n_te": n_te,
        "tcr": tcr,
    }


def compute_session_metrics(sources, results, alignment, rows, raw_mccr_metrics) -> Dict[str, Any]:
    """Aggregates both TCR (per camera & overall) and MCCR for the multi-camera session."""
    per_camera_tcr = []
    tot_ts = tot_iso = tot_int = tot_ne = tot_te = 0
    
    for source, result in zip(sources, results):
        metric_path = getattr(result, "metric_input_path", None)
        tcr_info = None
        if metric_path and Path(metric_path).exists():
            try:
                tcr_info = compute_tcr_for_csv(Path(metric_path))
            except Exception:
                pass
        if not tcr_info:
            tcr_info = compute_tcr_from_detections(getattr(result, "detections", []))
            
        n_ts = tcr_info.get("n_ts", 0)
        n_iso = tcr_info.get("n_isolated", 0)
        n_int = tcr_info.get("n_interrupted", 0)
        n_ne = tcr_info.get("n_not_evaluable", 0)
        n_te = tcr_info.get("n_te", 0)
        rate = tcr_info.get("tcr")
        
        tot_ts += n_ts
        tot_iso += n_iso
        tot_int += n_int
        tot_ne += n_ne
        tot_te += n_te
        
        per_camera_tcr.append({
            "camera_id": source.camera_id,
            "source_video": source.video.path.name,
            "n_ts": n_ts,
            "n_isolated": n_iso,
            "n_interrupted": n_int,
            "n_not_evaluable": n_ne,
            "n_te": n_te,
            "tcr_percent": rate,
        })

    overall_tcr = round(100.0 * tot_ts / tot_te, 2) if tot_te > 0 else None
    
    counts = raw_mccr_metrics.get("counts", {})
    mccr_val = raw_mccr_metrics.get("mccr_percent")
    eligible = raw_mccr_metrics.get("eligible", 0)
    
    return {
        "mccr": {
            "value_percent": mccr_val,
            "n_cc": counts.get("Corroborated", 0),
            "n_mc": eligible,
            "n_not_corroborated": counts.get("Not Corroborated", 0),
            "n_uncertain": counts.get("Uncertain", 0),
            "n_not_applicable": counts.get("Not Applicable", 0),
            "window_seconds": alignment.window_seconds,
            "confirmed": alignment.confirmed,
            "method": alignment.method or "Unspecified",
        },
        "tcr": {
            "overall_percent": overall_tcr,
            "n_ts": tot_ts,
            "n_isolated": tot_iso,
            "n_interrupted": tot_int,
            "n_not_evaluable": tot_ne,
            "n_te": tot_te,
            "per_camera": per_camera_tcr,
        }
    }


def build_session_report(sources, results, alignment, rows, raw_mccr_metrics) -> Dict[str, Any]:
    """Builds a structured dictionary containing all data required for HTML/PDF rendering."""
    metrics = compute_session_metrics(sources, results, alignment, rows, raw_mccr_metrics)
    report_id = f"MSFR-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    
    reviewed = sum(bool(r.get("analyst_decision") and r.get("analyst_decision") != "Not reviewed") for r in rows)
    decisions = Counter(r.get("analyst_decision") or "Not reviewed" for r in rows)
    objects = Counter(r.get("object_label") for r in rows)
    
    return {
        "schema_version": "2.0",
        "report_type": "multi_camera_session_forensic_report",
        "report_id": report_id,
        "generated_at_utc": utc_now(),
        "scene_id": alignment.scene_id or "Not specified",
        "alignment_confirmed": alignment.confirmed,
        "alignment_method": alignment.method or "Unspecified",
        "window_seconds": alignment.window_seconds,
        "review_summary": f"{reviewed} of {len(rows)} observations reviewed",
        "counts_by_analyst_decision": dict(decisions),
        "counts_by_object": dict(objects),
        "metrics": metrics,
        "cameras": [
            {
                "camera_id": s.camera_id,
                "location": s.location or "Not specified",
                "filename": s.video.path.name,
                "path": str(s.video.path),
                "width": s.video.width,
                "height": s.video.height,
                "fps": s.video.fps,
                "duration": s.video.duration,
                "offset_seconds": s.offset_seconds,
                "observation_count": sum(1 for r in rows if r["camera_id"] == s.camera_id),
            }
            for s in sources
        ],
        "observations": rows,
    }


def render_session_html(report: Dict[str, Any]) -> str:
    """Renders a self-contained, responsive forensic HTML report for the multi-camera session."""
    def e(value):
        return escape("Not recorded" if value is None or value == "" else str(value), quote=True)

    def fields(pairs):
        return "<dl>" + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in pairs) + "</dl>"

    mccr = report["metrics"]["mccr"]
    tcr = report["metrics"]["tcr"]
    mccr_val = f"{mccr['value_percent']:.2f}%" if mccr["value_percent"] is not None else "N/A"
    tcr_val = f"{tcr['overall_percent']:.2f}%" if tcr["overall_percent"] is not None else "N/A"
    
    cam_cards = "".join(
        f'<div class="metric"><strong>{c["camera_id"]}</strong>'
        f'<span>{e(c["filename"])}<br/>{c["width"]}x{c["height"]} @ {c["fps"]:.1f} FPS<br/>'
        f'Offset: {c["offset_seconds"]:+.3f}s &bull; {c["observation_count"]} obs</span></div>'
        for c in report["cameras"]
    )

    tcr_rows = "".join(
        f"<tr><td>{e(c['camera_id'])}</td><td>{e(c['source_video'])}</td>"
        f"<td>{c['n_ts']}</td><td>{c['n_isolated']}</td><td>{c['n_interrupted']}</td>"
        f"<td>{c['n_te']}</td><td><strong>{f'{c[\"tcr_percent\"]:.2f}%' if c['tcr_percent'] is not None else 'N/A'}</strong></td></tr>"
        for c in tcr["per_camera"]
    )

    obs_rows = "".join(
        f"<tr><td>{e(r['session_seconds']):.2f}s</td>"
        f"<td>{e(r['camera_id'])}</td>"
        f"<td>{e(r['video_seconds']):.2f}s</td>"
        f"<td>{r['frame_number']}</td>"
        f"<td>{e(r['object_label'].title())}</td>"
        f"<td>{r['confidence']:.1%}</td>"
        f"<td><span class=\"status-pill status-{r['corroboration_status'].lower().replace(' ', '-')}\">{e(r['corroboration_status'])}</span></td>"
        f"<td>{e(r.get('analyst_decision', 'Not reviewed'))}</td>"
        f"<td>{e(r.get('analyst_notes', ''))}</td></tr>"
        for r in report["observations"]
    ) or '<tr><td colspan="9">No weapon observations recorded across the session cameras.</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Camera Forensic Report &mdash; {e(report['scene_id'])}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #edf0f5; color: #222a39; font: 14px/1.55 Arial, sans-serif; }}
main {{ max-width: 1180px; margin: 30px auto; background: white; padding: 40px 44px; border-top: 7px solid #2f318e; }}
header {{ border-bottom: 1px solid #dce0e8; padding-bottom: 22px; }}
.brand {{ color: #2f318e; letter-spacing: 2px; font-size: 12px; font-weight: bold; }}
h1 {{ font-size: 28px; line-height: 1.2; margin: 9px 0; }}
h2 {{ font-size: 16px; margin: 28px 0 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #2f318e; border-bottom: 1px solid #e0e4ed; padding-bottom: 5px; }}
p {{ margin: 8px 0; }}
.muted {{ color: #596577; }}
.badge {{ display: inline-block; background: #fff2d8; color: #775014; padding: 5px 10px; border-radius: 5px; font-weight: 600; font-size: 12px; }}
.disclaimer {{ margin: 20px 0; background: #fff4da; border: 1px solid #d8b45c; color: #493614; padding: 12px 14px; font-weight: bold; border-radius: 4px; }}
.metrics {{ display: flex; gap: 12px; margin: 18px 0; flex-wrap: wrap; }}
.metric {{ flex: 1; min-width: 180px; background: #f4f5fa; padding: 16px; border-radius: 6px; border-left: 4px solid #2f318e; }}
.metric strong {{ display: block; font-size: 24px; color: #2f318e; }}
.metric span {{ font-size: 11.5px; color: #596577; }}
dl {{ display: grid; grid-template-columns: 200px minmax(0, 1fr); margin: 0; border-top: 1px solid #e0e4ed; }}
dt, dd {{ margin: 0; padding: 8px 10px; border-bottom: 1px solid #e0e4ed; overflow-wrap: anywhere; }}
dt {{ font-weight: bold; background: #f7f8fb; }}
.table-wrap {{ overflow-x: auto; margin: 12px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th {{ background: #303841; color: white; text-align: left; padding: 9px 8px; }}
th, td {{ padding: 8px 8px; border: 1px solid #e0e4ed; vertical-align: top; }}
tr:nth-child(even) {{ background: #f7f8fb; }}
.status-pill {{ padding: 3px 7px; border-radius: 4px; font-weight: 600; font-size: 11px; display: inline-block; }}
.status-corroborated {{ background: #dff4e2; color: #1c6d2c; }}
.status-not-corroborated {{ background: #fde8e8; color: #a51d24; }}
.status-uncertain {{ background: #fff3cd; color: #856404; }}
.status-not-applicable {{ background: #e9ecef; color: #495057; }}
footer {{ margin-top: 32px; border-top: 1px solid #dce0e8; padding-top: 14px; font-size: 12px; color: #596577; }}
</style>
</head>
<body>
<main>
<header>
  <div class="brand">FORENSIKADA / MULTI-CAMERA FORENSIC ANALYSIS</div>
  <h1>Multi-Camera Session Forensic Report</h1>
  <p class="muted">Incident / Scene: <strong>{e(report['scene_id'])}</strong> &bull; Alignment: {e(report['alignment_method'])} &bull; Window &Delta;t = {report['window_seconds']:.2f}s</p>
  <span class="badge">{e(report['review_summary'])}</span>
  <p class="muted">Report ID: {e(report['report_id'])} &bull; Generated (UTC): {e(report['generated_at_utc'])}</p>
</header>

<div class="disclaimer">{e(REPORT_DISCLAIMER)}</div>

<h2>Key Performance Metrics</h2>
<div class="metrics">
  <div class="metric">
    <strong>{mccr_val}</strong>
    <span>Multi-Camera Corroboration (MCCR)<br/>{mccr['n_cc']} / {mccr['n_mc']} eligible</span>
  </div>
  <div class="metric">
    <strong>{tcr_val}</strong>
    <span>Session Aggregate TCR<br/>{tcr['n_ts']} / {tcr['n_te']} supported</span>
  </div>
  <div class="metric">
    <strong>{len(report['observations'])}</strong>
    <span>Total Observations<br/>Handgun: {report['counts_by_object'].get('handgun', 0)} &bull; Knife: {report['counts_by_object'].get('knife', 0)}</span>
  </div>
  <div class="metric">
    <strong>{len(report['cameras'])}</strong>
    <span>Recording Feeds<br/>Synced time range</span>
  </div>
</div>

<h2>Camera Feeds & Timing Setup</h2>
<div class="metrics">
  {cam_cards}
</div>

<h2>Temporal Consistency Rate (TCR) Breakdown</h2>
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Camera ID</th>
        <th>Source Video</th>
        <th>Supported (N_TS)</th>
        <th>Isolated</th>
        <th>Interrupted</th>
        <th>Eligible (N_TE)</th>
        <th>TCR %</th>
      </tr>
    </thead>
    <tbody>
      {tcr_rows}
    </tbody>
  </table>
</div>
<p class="muted">TCR measures the stability of detection tracklets across time. Observations on frame boundaries are excluded as not evaluable.</p>

<h2>Multi-Camera Corroboration Rate (MCCR) Details</h2>
{fields([
    ("Overall MCCR Rate", mccr_val),
    ("Corroborated Observations (N_CC)", f"{mccr['n_cc']} observation(s)"),
    ("Total Eligible Observations (N_MC)", f"{mccr['n_mc']} observation(s) (N_CC + Not Corroborated)"),
    ("Not Corroborated", f"{mccr['n_not_corroborated']} observation(s)"),
    ("Uncertain Observations", f"{mccr['n_uncertain']} observation(s) (conflicting class in peer view; excluded from denominator)"),
    ("Not Applicable", f"{mccr['n_not_applicable']} observation(s) (no peer footage available at session time; excluded from denominator)"),
    ("Alignment Reference", report['alignment_method']),
    ("Matching Concurrency Window (\u0394t)", f"{report['window_seconds']:.2f} seconds"),
])}

<h2>Combined Observation & Corroboration Log</h2>
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>Session Time</th>
        <th>Camera</th>
        <th>Video Time</th>
        <th>Frame</th>
        <th>Object</th>
        <th>Confidence</th>
        <th>Cross-View Status</th>
        <th>Analyst Decision</th>
        <th>Analyst Notes</th>
      </tr>
    </thead>
    <tbody>
      {obs_rows}
    </tbody>
  </table>
</div>

<footer>
  Generated by Forensikada Multi-Camera Forensic Suite. Companion artifacts: session.json, combined_observations.csv.
  Observations are machine-generated propositions subject to human review.
</footer>
</main>
</body>
</html>"""


def write_session_pdf(report: Dict[str, Any], path: Path) -> Path:
    """Generates an executive-grade, multi-page ReportLab PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    regular, bold = "Helvetica", "Helvetica-Bold"
    font_dir = Path("C:/Windows/Fonts")
    if (font_dir / "arial.ttf").exists() and (font_dir / "arialbd.ttf").exists():
        if "Forensikada" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("Forensikada", str(font_dir / "arial.ttf")))
            pdfmetrics.registerFont(TTFont("ForensikadaBold", str(font_dir / "arialbd.ttf")))
            pdfmetrics.registerFontFamily("Forensikada", normal="Forensikada", bold="ForensikadaBold")
        regular, bold = "Forensikada", "ForensikadaBold"

    styles = getSampleStyleSheet()
    for s in styles.byName.values():
        s.fontName = regular

    styles.add(ParagraphStyle(
        "ReportBody", fontName=regular, fontSize=8.5, leading=12, spaceAfter=4,
        splitLongWords=True, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "ReportTitle", fontName=bold, fontSize=20, leading=24,
        textColor=colors.HexColor("#2f318e"), spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "Section", fontName=bold, fontSize=11, leading=15,
        textColor=colors.HexColor("#2f318e"), spaceBefore=12, spaceAfter=6, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        "Disclaimer", fontName=bold, fontSize=8.5, leading=12,
        textColor=colors.HexColor("#493614"),
    ))
    styles.add(ParagraphStyle(
        "TableHead", fontName=bold, fontSize=8, leading=10,
        textColor=colors.white, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "TableCell", fontName=regular, fontSize=8, leading=10,
        alignment=TA_LEFT,
    ))

    body = styles["ReportBody"]

    def p(text, style=body):
        val = "Not recorded" if text is None or text == "" else str(text)
        return Paragraph(escape(val).replace("\n", "<br/>"), style)

    width = A4[0] - 72

    def field_table(rows_data):
        table = Table([[p(k), p(v)] for k, v in rows_data], colWidths=[160, width - 160], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f3f8")),
            ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor("#dce0e8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    mccr = report["metrics"]["mccr"]
    tcr = report["metrics"]["tcr"]
    mccr_str = f"{mccr['value_percent']:.2f}%" if mccr["value_percent"] is not None else "N/A"
    tcr_str = f"{tcr['overall_percent']:.2f}%" if tcr["overall_percent"] is not None else "N/A"

    disclaimer_table = Table([[p(REPORT_DISCLAIMER, styles["Disclaimer"])]], colWidths=[width], hAlign="LEFT")
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4da")),
        ("BOX", (0, 0), (-1, -1), .8, colors.HexColor("#d8b45c")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    # Key metric summary box
    summary_matrix = [
        [
            Paragraph(f"<b>MCCR: {mccr_str}</b><br/><font size=7 color='#596577'>Corroborated: {mccr['n_cc']} / {mccr['n_mc']} eligible</font>", styles["ReportBody"]),
            Paragraph(f"<b>TCR: {tcr_str}</b><br/><font size=7 color='#596577'>Session aggregate: {tcr['n_ts']} / {tcr['n_te']} supported</font>", styles["ReportBody"]),
            Paragraph(f"<b>Observations: {len(report['observations'])}</b><br/><font size=7 color='#596577'>Handguns: {report['counts_by_object'].get('handgun', 0)} &bull; Knives: {report['counts_by_object'].get('knife', 0)}</font>", styles["ReportBody"]),
            Paragraph(f"<b>Cameras: {len(report['cameras'])}</b><br/><font size=7 color='#596577'>Incident: {report['scene_id']}</font>", styles["ReportBody"]),
        ]
    ]
    summary_box = Table(summary_matrix, colWidths=[width / 4] * 4, hAlign="LEFT")
    summary_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f5fa")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cfd1ea")),
        ("INNERGRID", (0, 0), (-1, -1), .5, colors.HexColor("#dce0e8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    story = [
        p("FORENSIKADA / MULTI-CAMERA FORENSIC AUDIT", styles["Section"]),
        p("Multi-Camera Session Forensic Report", styles["ReportTitle"]),
        p(f"Incident Scene: {report['scene_id']} &bull; Generated (UTC): {report['generated_at_utc']}<br/>"
          f"Report Identifier: {report['report_id']} &bull; Review Status: {report['review_summary']}"),
        Spacer(1, 4),
        disclaimer_table,
        Spacer(1, 6),
        summary_box,
        Spacer(1, 8),
        p("Incident & Synchronization Configuration", styles["Section"]),
        field_table([
            ("Incident Scene ID", report["scene_id"]),
            ("Alignment Confirmation", "Analyst Confirmed Synchronized" if report["alignment_confirmed"] else "Unconfirmed / Preliminary"),
            ("Alignment Reference Method", report["alignment_method"]),
            ("Matching Concurrency Window (\u0394t)", f"{report['window_seconds']:.2f} seconds"),
            ("Analyst Review Progress", report["review_summary"]),
        ]),
        Spacer(1, 8),
        p("Temporal Consistency Rate (TCR) per Camera", styles["Section"]),
    ]

    # TCR Table
    tcr_table_data = [[
        Paragraph("<b>Camera ID</b>", styles["TableHead"]),
        Paragraph("<b>Video Source</b>", styles["TableHead"]),
        Paragraph("<b>Supported (N_TS)</b>", styles["TableHead"]),
        Paragraph("<b>Isolated</b>", styles["TableHead"]),
        Paragraph("<b>Interrupted</b>", styles["TableHead"]),
        Paragraph("<b>Eligible (N_TE)</b>", styles["TableHead"]),
        Paragraph("<b>TCR %</b>", styles["TableHead"]),
    ]]
    for c in tcr["per_camera"]:
        tcr_rate_str = f"{c['tcr_percent']:.2f}%" if c["tcr_percent"] is not None else "N/A"
        tcr_table_data.append([
            Paragraph(c["camera_id"], styles["TableCell"]),
            Paragraph(c["source_video"], styles["TableCell"]),
            Paragraph(str(c["n_ts"]), styles["TableCell"]),
            Paragraph(str(c["n_isolated"]), styles["TableCell"]),
            Paragraph(str(c["n_interrupted"]), styles["TableCell"]),
            Paragraph(str(c["n_te"]), styles["TableCell"]),
            Paragraph(f"<b>{tcr_rate_str}</b>", styles["TableCell"]),
        ])
    tcr_tbl = Table(tcr_table_data, colWidths=[65, 140, 75, 60, 60, 65, width - 465], hAlign="LEFT")
    tcr_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#303841")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor("#dce0e8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tcr_tbl)

    story.extend([
        Spacer(1, 8),
        p("Multi-Camera Corroboration Rate (MCCR) Analysis", styles["Section"]),
        field_table([
            ("Overall MCCR Score", mccr_str),
            ("Corroborated Observations (N_CC)", f"{mccr['n_cc']} observations"),
            ("Total Eligible Observations (N_MC)", f"{mccr['n_mc']} observations"),
            ("Not Corroborated Observations", f"{mccr['n_not_corroborated']} observations"),
            ("Uncertain Observations (Conflicting Class)", f"{mccr['n_uncertain']} observations (excluded from MCCR denominator)"),
            ("Not Applicable (No Peer Video at Time)", f"{mccr['n_not_applicable']} observations (excluded from MCCR denominator)"),
        ]),
        Spacer(1, 8),
        p("Combined Multi-Camera Observations Log", styles["Section"]),
    ])

    # Observations Table
    obs_head = [
        Paragraph("<b>Session (s)</b>", styles["TableHead"]),
        Paragraph("<b>Camera</b>", styles["TableHead"]),
        Paragraph("<b>Video (s)</b>", styles["TableHead"]),
        Paragraph("<b>Frame</b>", styles["TableHead"]),
        Paragraph("<b>Object</b>", styles["TableHead"]),
        Paragraph("<b>Conf</b>", styles["TableHead"]),
        Paragraph("<b>Cross-View Status</b>", styles["TableHead"]),
        Paragraph("<b>Analyst Decision</b>", styles["TableHead"]),
    ]
    obs_table_data = [obs_head]
    for r in report["observations"]:
        obs_table_data.append([
            Paragraph(f"{r['session_seconds']:.2f}s", styles["TableCell"]),
            Paragraph(r["camera_id"], styles["TableCell"]),
            Paragraph(f"{r['video_seconds']:.2f}s", styles["TableCell"]),
            Paragraph(str(r["frame_number"]), styles["TableCell"]),
            Paragraph(r["object_label"].title(), styles["TableCell"]),
            Paragraph(f"{r['confidence']:.1%}", styles["TableCell"]),
            Paragraph(r["corroboration_status"], styles["TableCell"]),
            Paragraph(r.get("analyst_decision", "Not reviewed"), styles["TableCell"]),
        ])
    if len(obs_table_data) == 1:
        obs_table_data.append([Paragraph("No weapon observations detected in session.", styles["TableCell"])] * 8)

    obs_tbl = Table(obs_table_data, colWidths=[55, 55, 55, 45, 60, 45, 105, width - 420], hAlign="LEFT")
    obs_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#303841")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#dce0e8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(obs_tbl)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor("#66717d"))
        canvas.drawString(36, 20, "Forensikada | Multi-Camera Session Forensic Analysis Report")
        canvas.drawRightString(A4[0] - 36, 20, f"Page {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=36, leftMargin=36,
        topMargin=32, bottomMargin=36,
        title=f"Multi-Camera Forensic Report - {report['scene_id']}",
        author="Forensikada",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return Path(path)


class MultiCameraForensicReportDialog(QDialog):
    """Full-featured interactive in-app viewer for multi-camera forensic findings."""

    def __init__(self, sources, results, alignment, rows, parent=None, raw_mccr_metrics=None):
        super().__init__(parent)
        self.sources = sources
        self.results = results
        self.alignment = alignment
        self.rows = rows
        
        if raw_mccr_metrics is None:
            from mockup_ui.multi_camera import build_timeline
            _, raw_mccr_metrics = build_timeline(sources, results, alignment)
            
        self.report = build_session_report(sources, results, alignment, rows, raw_mccr_metrics)
        self.setObjectName("forensicReportDialog")
        self.setWindowTitle(f"Multi-Camera Forensic Analysis Report &bull; {self.report['scene_id']}")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(1120, 780)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(10)

        # Header
        top = QHBoxLayout()
        h_box = QVBoxLayout()
        heading = QLabel("Multi-Camera Forensic Analysis Report")
        heading.setObjectName("dialogHeading")
        sub = QLabel(f"Scene: {self.report['scene_id']} &bull; {len(sources)} Cameras &bull; Alignment: {self.report['alignment_method']}")
        sub.setObjectName("dialogDetail")
        h_box.addWidget(heading)
        h_box.addWidget(sub)
        top.addLayout(h_box, 1)

        reviewed_cnt = sum(bool(r.get("analyst_decision") and r.get("analyst_decision") != "Not reviewed") for r in rows)
        badge = QLabel(f"{reviewed_cnt} / {len(rows)} REVIEWED")
        badge.setObjectName("reviewBadge")
        top.addWidget(badge)
        layout.addLayout(top)

        disclaimer = QLabel(REPORT_DISCLAIMER)
        disclaimer.setObjectName("reportObservation")
        disclaimer.setWordWrap(True)
        layout.addWidget(disclaimer)

        # 6 Cards Grid
        mccr = self.report["metrics"]["mccr"]
        tcr = self.report["metrics"]["tcr"]
        mccr_str = f"{mccr['value_percent']:.2f}%" if mccr["value_percent"] is not None else "N/A"
        tcr_str = f"{tcr['overall_percent']:.2f}%" if tcr["overall_percent"] is not None else "N/A"

        cam_tcr_details = "\n".join(
            f"{c['camera_id']}: {f'{c[\"tcr_percent\"]:.2f}%' if c['tcr_percent'] is not None else 'N/A'} "
            f"({c['n_ts']} / {c['n_te']} eligible)"
            for c in tcr["per_camera"]
        )

        grid = QGridLayout()
        cards_info = [
            (
                "MULTI-CAMERA CORROBORATION (MCCR)",
                f"Rate: {mccr_str}\n"
                f"Corroborated (N_CC): {mccr['n_cc']} / {mccr['n_mc']} eligible\n"
                f"Not Corroborated: {mccr['n_not_corroborated']} \u00b7 Uncertain: {mccr['n_uncertain']}\n"
                f"Not Applicable: {mccr['n_not_applicable']}"
            ),
            (
                "TEMPORAL CONSISTENCY (TCR)",
                f"Overall Rate: {tcr_str} ({tcr['n_ts']} / {tcr['n_te']} supported)\n"
                f"{cam_tcr_details}"
            ),
            (
                "SESSION CAMERAS & FOOTAGE",
                "\n".join(f"{s.camera_id}: {s.video.path.name} (offset {s.offset_seconds:+.2f}s)" for s in sources)
            ),
            (
                "DETECTION SUMMARY",
                f"{len(rows):,} total observations\n"
                f"Handgun: {self.report['counts_by_object'].get('handgun', 0):,} \u00b7 "
                f"Knife: {self.report['counts_by_object'].get('knife', 0):,}"
            ),
            (
                "ALIGNMENT & CONCURRENCY",
                f"Scene: {self.report['scene_id']}\n"
                f"Window \u0394t: {self.report['window_seconds']:.2f} s\n"
                f"Confirmed: {'Yes' if self.report['alignment_confirmed'] else 'No'}"
            ),
            (
                "ANALYST REVIEW COVERAGE",
                f"Coverage: {self.report['review_summary']}\n" +
                ", ".join(f"{k}: {v}" for k, v in self.report["counts_by_analyst_decision"].items())
            ),
        ]

        for index, (title, content) in enumerate(cards_info):
            card = QFrame()
            card.setObjectName("reportCard")
            c_box = QVBoxLayout(card)
            t_lbl = QLabel(title)
            t_lbl.setObjectName("sectionTitle")
            v_lbl = QLabel(content)
            v_lbl.setObjectName("dialogDetail")
            v_lbl.setWordWrap(True)
            c_box.addWidget(t_lbl)
            c_box.addWidget(v_lbl)
            grid.addWidget(card, index // 3, index % 3)

        layout.addLayout(grid)

        # Observations Timeline Table
        layout.addWidget(QLabel("COMBINED OBSERVATIONS & CORROBORATION TIMELINE"))
        self.table = QTableWidget(len(rows), 9)
        self.table.setHorizontalHeaderLabels([
            "Session time", "Camera", "Video time", "Frame", "Object", "Confidence",
            "Cross-view status", "Analyst decision", "Analyst notes"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        for i, r in enumerate(rows):
            vals = [
                f"{r['session_seconds']:.2f}s",
                r["camera_id"],
                f"{r['video_seconds']:.2f}s",
                str(r["frame_number"]),
                r["object_label"].title(),
                f"{r['confidence']:.1%}",
                r["corroboration_status"],
                r.get("analyst_decision", "Not reviewed"),
                r.get("analyst_notes", ""),
            ]
            for col, val in enumerate(vals):
                it = QTableWidgetItem(val)
                it.setToolTip(f"{r.get('source_video', '')}\n{r.get('reason', '')}")
                self.table.setItem(i, col, it)

        layout.addWidget(self.table, 1)

        # Bottom buttons
        buttons_layout = QHBoxLayout()
        export_btn = QPushButton("Export PDF Report")
        export_btn.clicked.connect(self.export_pdf)
        buttons_layout.addWidget(export_btn)
        buttons_layout.addStretch()
        
        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        buttons_layout.addWidget(close_box)
        layout.addLayout(buttons_layout)

    def export_pdf(self):
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Multi-Camera Forensic Report PDF", "session_forensic_report.pdf", "PDF Files (*.pdf)"
        )
        if not dest:
            return
        try:
            write_session_pdf(self.report, Path(dest))
            QMessageBox.information(self, "Report Exported", f"Forensic PDF report saved successfully to:\n{dest}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", f"Could not export PDF: {exc}")
