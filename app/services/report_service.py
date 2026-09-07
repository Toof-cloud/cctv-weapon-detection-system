import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.utils.timestamps import format_timestamp


class ReportService:
    """
    Forensic-oriented detection report and audit log generator for FORENSIKADA.
    
    Generates structured, traceable detection records strictly containing:
    1. source-video references ('source_video')
    2. frame numbers ('frame_number')
    3. available source timestamps ('timestamp_seconds', 'timestamp_formatted')
    4. camera identifiers ('camera_id')
    5. object labels ('object_label')
    6. bounding-box coordinates ('bounding_box', 'x1', 'y1', 'x2', 'y2')
    7. confidence scores ('confidence_score')
    8. validation statuses ('validation_status', 'rejection_reason')
    """

    REPORT_COLUMNS = [
        "source_video",
        "frame_number",
        "timestamp_seconds",
        "timestamp_formatted",
        "camera_id",
        "object_label",
        "bounding_box",
        "confidence_score",
        "validation_status",
        "rejection_reason",
    ]

    def __init__(self, output_dir: Optional[Path | str] = None):
        self.output_dir = Path(output_dir) if output_dir else ROOT_DIR / "outputs" / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def normalize_record(
        self,
        record: Dict[str, Any],
        default_source_video: str = "Unknown_Video.mp4",
        default_camera_id: str = "CAM-01",
    ) -> Dict[str, Any]:
        """Ensures every detection record conforms strictly to the required 8 forensic fields."""
        source_video = str(record.get("source_video") or default_source_video)
        frame_number = int(record.get("frame_number", 0))
        
        ts_sec = record.get("timestamp_seconds")
        if ts_sec is None:
            ts_sec = float(record.get("timestamp", 0.0))
        ts_sec = round(float(ts_sec), 3)
        ts_formatted = str(record.get("timestamp_formatted") or format_timestamp(ts_sec))

        camera_id = str(record.get("camera_id") or default_camera_id)
        object_label = str(record.get("object_label") or record.get("class_name") or "unknown")

        # Bounding box coordinates
        if "box" in record and isinstance(record["box"], (list, tuple)) and len(record["box"]) == 4:
            x1, y1, x2, y2 = [int(v) for v in record["box"]]
        else:
            x1 = int(record.get("x1", 0))
            y1 = int(record.get("y1", 0))
            x2 = int(record.get("x2", 0))
            y2 = int(record.get("y2", 0))
        box_str = f"[{x1}, {y1}, {x2}, {y2}]"


        confidence = float(record.get("confidence_score") or record.get("confidence") or 0.0)
        confidence = round(confidence, 4)

        val_status = record.get("validation_status") or record.get("status") or "CONFIRMED_ALERT"
        rejection_reason = str(record.get("rejection_reason") or "")

        return {
            "source_video": source_video,
            "frame_number": frame_number,
            "timestamp_seconds": ts_sec,
            "timestamp_formatted": ts_formatted,
            "camera_id": camera_id,
            "object_label": object_label,
            "bounding_box": box_str,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence_score": confidence,
            "validation_status": val_status,
            "rejection_reason": rejection_reason,
        }

    def export_csv(
        self,
        records: List[Dict[str, Any]],
        output_path: Path | str,
        default_source_video: str = "Unknown_Video.mp4",
        default_camera_id: str = "CAM-01",
    ) -> Path:
        """Exports detection records to CSV strictly formatted with the 8 required forensic fields."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        normalized = [
            self.normalize_record(r, default_source_video, default_camera_id)
            for r in records
        ]

        with open(out_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.REPORT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(normalized)

        return out_file

    def export_json(
        self,
        records: List[Dict[str, Any]],
        case_metadata: Dict[str, Any],
        output_path: Path | str,
        default_source_video: str = "Unknown_Video.mp4",
        default_camera_id: str = "CAM-01",
    ) -> Path:
        """Exports a full forensic JSON log including case metadata and normalized records."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        normalized = [
            self.normalize_record(r, default_source_video, default_camera_id)
            for r in records
        ]

        confirmed = [r for r in normalized if r["validation_status"] in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")]
        suppressed = [r for r in normalized if r["validation_status"] not in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")]

        payload = {
            "forensic_report_header": {
                "system_name": "FORENSIKADA CCTV Weapon Detection System",
                "institution": "National University - Manila (CCIT)",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "case_id": case_metadata.get("case_id", f"CCTV-{datetime.now().strftime('%Y%m%d-%H%M')}"),
                "analyst_name": case_metadata.get("analyst_name", "Forensic Automated Pipeline"),
                "model_version": case_metadata.get("model_version", "Model 7 (Hard Negative Mining + CCTV Intelligence)"),
            },
            "source_cameras": case_metadata.get("cameras", []),
            "incident_summary": {
                "total_evaluated_proposals": len(normalized),
                "total_confirmed_alerts": len(confirmed),
                "total_suppressed_traps": len(suppressed),
                "peak_confidence": max([r["confidence_score"] for r in confirmed], default=0.0),
                "first_threat_timestamp": min([r["timestamp_seconds"] for r in confirmed], default=None),
            },
            "detection_records": normalized,
        }

        with open(out_file, mode="w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return out_file

    def export_html_report(
        self,
        records: List[Dict[str, Any]],
        case_metadata: Dict[str, Any],
        output_path: Path | str,
        keyframe_crops: Optional[List[Dict[str, Any]]] = None,
        default_source_video: str = "Unknown_Video.mp4",
        default_camera_id: str = "CAM-01",
    ) -> Path:
        """Generates an official, standalone forensic HTML incident report with styling and evidence."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        normalized = [
            self.normalize_record(r, default_source_video, default_camera_id)
            for r in records
        ]
        confirmed = [r for r in normalized if r["validation_status"] in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")]
        suppressed = [r for r in normalized if r["validation_status"] not in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")]

        case_id = case_metadata.get("case_id", f"CCTV-{datetime.now().strftime('%Y%m%d-%H%M')}")
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        analyst = case_metadata.get("analyst_name", "Forensic Investigation Unit")
        model_name = case_metadata.get("model_version", "Model 7 Faster R-CNN ResNet50-FPN-v2")

        # Threat counts by weapon & camera
        threat_breakdown = {}
        cam_breakdown = {}
        for r in confirmed:
            w = r["object_label"].capitalize()
            c = r["camera_id"]
            threat_breakdown[w] = threat_breakdown.get(w, 0) + 1
            cam_breakdown[c] = cam_breakdown.get(c, 0) + 1

        first_ts = min([r["timestamp_formatted"] for r in confirmed], default="N/A")
        peak_conf = max([r["confidence_score"] for r in confirmed], default=0.0)

        # Build Table Rows
        table_rows = []
        for r in normalized:
            is_conf = r["validation_status"] in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")
            badge_class = "badge-alert" if is_conf else "badge-suppressed"
            status_text = r["validation_status"]
            reason_text = f"<br><small class='text-muted'>{r['rejection_reason']}</small>" if r["rejection_reason"] else ""
            
            table_rows.append(f"""
            <tr>
                <td><strong>{r['camera_id']}</strong></td>
                <td>{r['timestamp_formatted']}<br><small class='text-muted'>{r['timestamp_seconds']}s</small></td>
                <td>Frame {r['frame_number']}</td>
                <td><span class='weapon-tag weapon-{r['object_label']}'>{r['object_label'].upper()}</span></td>
                <td><strong>{r['confidence_score']:.1%}</strong></td>
                <td><code>{r['bounding_box']}</code></td>
                <td><span class='{badge_class}'>{status_text}</span>{reason_text}</td>
                <td><small>{r['source_video']}</small></td>
            </tr>
            """)

        # Keyframe thumbnails section
        keyframes_html = ""
        if keyframe_crops:
            kf_items = []
            for kf in keyframe_crops[:8]:  # Top 8 key evidence frames
                img_path = kf.get("image_path", "")
                rel_img = Path(img_path).name
                caption = f"{kf.get('camera_id', 'CAM')} | Frame {kf.get('frame_number', 0)} ({kf.get('timestamp', '')}) | {kf.get('label', '').upper()} {kf.get('confidence', 0):.1%}"
                kf_items.append(f"""
                <div class="evidence-card">
                    <img src="{img_path}" alt="{caption}">
                    <div class="evidence-caption">{caption}</div>
                </div>
                """)
            keyframes_html = f"""
            <div class="section-title">Key Forensic Evidence Snapshots</div>
            <div class="evidence-grid">
                {''.join(kf_items)}
            </div>
            """

        cameras_info = ""
        if "cameras" in case_metadata and case_metadata["cameras"]:
            c_cards = []
            for c in case_metadata["cameras"]:
                c_cards.append(f"""
                <div class="meta-box">
                    <div class="meta-label">CAMERA: {c.get('camera_id', 'N/A')}</div>
                    <div><strong>File:</strong> {c.get('filename', 'N/A')}</div>
                    <div><strong>Resolution:</strong> {c.get('resolution', 'N/A')} @ {c.get('fps', 'N/A')} FPS</div>
                    <div><strong>Duration:</strong> {c.get('duration_seconds', 'N/A')}s ({c.get('total_frames', 'N/A')} frames)</div>
                </div>
                """)
            cameras_info = f"<div class='meta-grid'>{''.join(c_cards)}</div>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Forensic CCTV Detection Report - {case_id}</title>
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1e293b;
        background-color: #f8fafc;
        margin: 0;
        padding: 30px;
        line-height: 1.5;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        background: #ffffff;
        padding: 40px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }}
    .header {{
        border-bottom: 3px solid #0f172a;
        padding-bottom: 20px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }}
    .header h1 {{
        margin: 0;
        font-size: 26px;
        color: #0f172a;
        letter-spacing: -0.5px;
    }}
    .header .subtitle {{
        color: #64748b;
        font-size: 14px;
        margin-top: 5px;
    }}
    .case-badge {{
        background: #0f172a;
        color: #f8fafc;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 14px;
        text-align: right;
    }}
    .section-title {{
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
        margin: 30px 0 15px 0;
        border-left: 4px solid #0284c7;
        padding-left: 10px;
    }}
    .summary-cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 15px;
        margin-bottom: 25px;
    }}
    .card {{
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px;
    }}
    .card .val {{
        font-size: 26px;
        font-weight: bold;
        color: #0f172a;
    }}
    .card .lbl {{
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }}
    .meta-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 15px;
        margin-bottom: 20px;
    }}
    .meta-box {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 14px;
        font-size: 13px;
    }}
    .meta-label {{
        font-weight: bold;
        color: #0284c7;
        margin-bottom: 6px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12.5px;
        margin-top: 15px;
    }}
    th, td {{
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid #e2e8f0;
    }}
    th {{
        background-color: #f1f5f9;
        color: #334155;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.5px;
    }}
    tr:hover {{
        background-color: #f8fafc;
    }}
    .badge-alert {{
        background: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
    }}
    .badge-suppressed {{
        background: #f1f5f9;
        color: #475569;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
    }}
    .weapon-tag {{
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
    }}
    .weapon-handgun {{
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fca5a5;
    }}
    .weapon-knife {{
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fcd34d;
    }}
    .evidence-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }}
    .evidence-card {{
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        overflow: hidden;
        background: #ffffff;
    }}
    .evidence-card img {{
        width: 100%;
        height: auto;
        display: block;
    }}
    .evidence-caption {{
        padding: 8px 10px;
        font-size: 11.5px;
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
        font-weight: 600;
    }}
    .text-muted {{
        color: #64748b;
    }}
    .footer {{
        margin-top: 40px;
        border-top: 1px solid #e2e8f0;
        padding-top: 15px;
        display: flex;
        justify-content: space-between;
        font-size: 11.5px;
        color: #94a3b8;
    }}
    @media print {{
        body {{
            background: #ffffff;
            padding: 0;
        }}
        .container {{
            box-shadow: none;
            border: none;
            max-width: 100%;
        }}
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1>FORENSIC CCTV WEAPON DETECTION INCIDENT REPORT</h1>
            <div class="subtitle">FORENSIKADA Automated Surveillance Analysis Engine | National University - Manila</div>
        </div>
        <div class="case-badge">
            CASE REF: {case_id}<br>
            <span style="font-weight: normal; font-size: 12px;">Generated: {gen_time}</span>
        </div>
    </div>

    <div class="summary-cards">
        <div class="card">
            <div class="val">{len(confirmed)}</div>
            <div class="lbl">Confirmed Weapon Alerts</div>
        </div>
        <div class="card">
            <div class="val">{len(suppressed)}</div>
            <div class="lbl">Suppressed Traps (Audited)</div>
        </div>
        <div class="card">
            <div class="val">{peak_conf:.1%}</div>
            <div class="lbl">Peak Detection Confidence</div>
        </div>
        <div class="card">
            <div class="val">{first_ts}</div>
            <div class="lbl">First Verified Incident Time</div>
        </div>
    </div>

    <div class="section-title">Surveillance Hardware & Video Inputs</div>
    {cameras_info}

    {keyframes_html}

    <div class="section-title">Chronological Detection & Traceability Log</div>
    <table>
        <thead>
            <tr>
                <th>Camera</th>
                <th>Time (Source)</th>
                <th>Frame</th>
                <th>Weapon</th>
                <th>Confidence</th>
                <th>Bounding Box [x1, y1, x2, y2]</th>
                <th>Validation Status & Reason</th>
                <th>Source Video</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>

    <div class="footer">
        <div>Certified Forensic Audit Trail &bull; Conforms to Rule on Electronic Evidence (A.M. No. 01-7-01-SC)</div>
        <div>Engine: {model_name} &bull; Page 1 of 1</div>
    </div>
</div>
</body>
</html>
"""
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return out_file

    def export_pdf_report(
        self,
        records: List[Dict[str, Any]],
        case_metadata: Dict[str, Any],
        output_path: Path | str,
        keyframe_crops: Optional[List[Dict[str, Any]]] = None,
        default_source_video: str = "Unknown_Video.mp4",
        default_camera_id: str = "CAM-01",
    ) -> Path:
        """
        Renders the forensic report directly to PDF using PySide6's QTextDocument printToPdf.
        Also produces the HTML report in the same directory for universal browser preview.
        """
        out_pdf = Path(output_path)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        html_path = out_pdf.with_suffix(".html")

        # 1. Generate standalone HTML first
        self.export_html_report(
            records=records,
            case_metadata=case_metadata,
            output_path=html_path,
            keyframe_crops=keyframe_crops,
            default_source_video=default_source_video,
            default_camera_id=default_camera_id,
        )

        # 2. Render to PDF via PySide6 QPdfWriter
        try:
            from PySide6.QtGui import QTextDocument, QPdfWriter, QPageSize, QPageLayout
            from PySide6.QtCore import QMarginsF
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                app = QApplication(["--platform", "offscreen"])

            html_text = html_path.read_text(encoding="utf-8")
            doc = QTextDocument()
            doc.setHtml(html_text)

            pdf_writer = QPdfWriter(str(out_pdf))
            pdf_writer.setPageSize(QPageSize(QPageSize.A4))
            pdf_writer.setResolution(300)
            doc.print_(pdf_writer)

            if out_pdf.exists() and out_pdf.stat().st_size > 1000:
                print(f"[ReportService] Successfully generated Forensic PDF Report: {out_pdf.name} ({out_pdf.stat().st_size / 1024:.1f} KB)")
                return out_pdf
        except Exception as err:
            print(f"[ReportService] Warning: PySide6 PDF direct print raised: {err}. Standalone HTML report available at: {html_path}")

        return html_path