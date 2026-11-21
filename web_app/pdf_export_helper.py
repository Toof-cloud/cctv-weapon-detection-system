"""
Live Forensic PDF Report Generator for the FORENSIKADA Web Application (ReportLab).
Generates both Single-Video and Multi-Camera Forensic PDF Incident Reports in memory
so evaluators can download official PDFs directly from their web browser.
"""
from datetime import datetime, timezone
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def _header_style():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ForensikadaTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "ForensikadaSub",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "ForensikadaH2",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ForensikadaBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )
    return title_style, sub_style, h2_style, body_style


def build_single_video_pdf_bytes(
    entry: dict,
    filtered_records: list[dict],
    threshold: float,
    cctv_intel_enabled: bool,
    reviews: dict[str, dict],
    model_name: str = "Model 12 (Final Production SOTA)",
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    title_s, sub_s, h2_s, body_s = _header_style()

    story = []
    story.append(Paragraph("FORENSIKADA — FORENSIC VIDEO WEAPON DETECTION REPORT", title_s))
    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Generated: {gen_ts} | National University – Manila (CCIT) | Automated detections require human verification.", sub_s))

    # 1. Evidence & Chain of Custody Metadata
    story.append(Paragraph("1. Evidence Metadata & Integrity", h2_s))
    om = entry.get("orig_meta", {})
    meta_rows = [
        ["Evidence Title", entry.get("title", entry.get("source_filename", "Uploaded Video"))],
        ["Source Filename", entry.get("source_filename", "N/A")],
        ["SHA-256 Digital Hash", entry.get("sha256", "N/A")],
        ["Resolution / FPS / Duration", f"{om.get('width', 0)}x{om.get('height', 0)} | {om.get('fps', 0)} fps | {om.get('duration_seconds', 0)}s ({om.get('total_frames', 0)} frames)"],
        ["Detection Model", f"{model_name} (Faster R-CNN ResNet-50 FPN v2)"],
        ["Analysis Configuration", f"Confidence Threshold: {threshold*100:.0f}% | CCTV Intelligence Layer: {'ENABLED' if cctv_intel_enabled else 'RAW DETECTOR ONLY'}"],
    ]
    t_meta = Table(meta_rows, colWidths=[1.8 * inch, 5.5 * inch])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_meta)

    # 2. Detection Summary
    story.append(Paragraph("2. Detection & Observation Review Summary", h2_s))
    hg_cnt = sum(1 for r in filtered_records if r["object_label"] == "handgun")
    kn_cnt = sum(1 for r in filtered_records if r["object_label"] == "knife")
    mean_c = (sum(r["confidence_score"] for r in filtered_records) / len(filtered_records) * 100.0) if filtered_records else 0.0
    acc_cnt = sum(1 for v in reviews.values() if v.get("decision") == "Accept")
    rej_cnt = sum(1 for v in reviews.values() if v.get("decision") == "Reject")
    unc_cnt = sum(1 for v in reviews.values() if v.get("decision") == "Uncertain")

    sum_rows = [
        ["Total Detections", "Handguns", "Knives", "Mean Confidence", "Reviewed (Accept / Reject / Uncertain)"],
        [str(len(filtered_records)), str(hg_cnt), str(kn_cnt), f"{mean_c:.1f}%", f"{acc_cnt} / {rej_cnt} / {unc_cnt}"],
    ]
    t_sum = Table(sum_rows, colWidths=[1.2 * inch, 1.0 * inch, 1.0 * inch, 1.3 * inch, 2.8 * inch])
    t_sum.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_sum)

    # 3. Chronological Detection Log
    story.append(Paragraph("3. Chronological Frame-Level Detections", h2_s))
    if not filtered_records:
        story.append(Paragraph("No weapon detections exceeded the active confidence threshold and verification gates in this footage.", body_s))
    else:
        det_rows = [["#", "Time", "Frame", "Class", "Confidence", "Bounding Box [x1,y1,x2,y2]", "Reviewer Decision"]]
        for idx, r in enumerate(filtered_records[:80], 1):
            rkey = f"{r['frame_number']}_{r['object_label']}_{idx-1}"
            rev = reviews.get(rkey, {}).get("decision", "Pending")
            det_rows.append([
                str(idx),
                r["timestamp_formatted"],
                str(r["frame_number"]),
                r["object_label"].upper(),
                f"{r['confidence_score']*100:.1f}%",
                str(r["bounding_box"]),
                rev,
            ])
        t_det = Table(det_rows, colWidths=[0.4 * inch, 0.9 * inch, 0.65 * inch, 0.9 * inch, 0.9 * inch, 2.15 * inch, 1.4 * inch])
        t_det.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_det)
        if len(filtered_records) > 80:
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"Note: Showing first 80 of {len(filtered_records)} frame detections. Download the CSV file for the complete frame log.", body_s))

    doc.build(story)
    return buf.getvalue()


def build_multicam_pdf_bytes(
    scene_info: dict,
    cam1_records: list[dict],
    cam2_records: list[dict],
    threshold: float,
    cctv_intel_enabled: bool,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    title_s, sub_s, h2_s, body_s = _header_style()

    story = []
    story.append(Paragraph("FORENSIKADA — MULTI-CAMERA CORROBORATION FORENSIC REPORT", title_s))
    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Generated: {gen_ts} | Multi-Camera Session: {scene_info.get('title', 'N/A')}", sub_s))

    c1 = scene_info["cam1"]
    c2 = scene_info["cam2"]
    story.append(Paragraph("1. Synchronized Camera Feeds & Cryptographic Hashes", h2_s))
    meta_rows = [
        ["Camera ID", "Source Video", "SHA-256 Hash (First 32 hex)", "Detections"],
        ["CAM-01", c1.get("source_filename", ""), c1.get("sha256", "")[:32] + "...", str(len(cam1_records))],
        ["CAM-02", c2.get("source_filename", ""), c2.get("sha256", "")[:32] + "...", str(len(cam2_records))],
    ]
    t_meta = Table(meta_rows, colWidths=[0.9 * inch, 2.5 * inch, 2.9 * inch, 1.0 * inch])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_meta)

    combined = sorted(
        [{**r, "cam": "CAM-01"} for r in cam1_records] + [{**r, "cam": "CAM-02"} for r in cam2_records],
        key=lambda x: (x["timestamp_seconds"], x["cam"]),
    )

    story.append(Paragraph("2. Unified Chronological Multi-Camera Timeline", h2_s))
    if not combined:
        story.append(Paragraph(f"No confirmed weapon threats detected across CAM-01 or CAM-02 at threshold {threshold*100:.0f}% (CCTV Intelligence: {'ON' if cctv_intel_enabled else 'OFF'}).", body_s))
    else:
        rows = [["Time", "Camera", "Frame", "Weapon Class", "Confidence", "Bounding Box"]]
        for r in combined[:90]:
            rows.append([
                r["timestamp_formatted"],
                r["cam"],
                str(r["frame_number"]),
                r["object_label"].upper(),
                f"{r['confidence_score']*100:.1f}%",
                str(r["bounding_box"]),
            ])
        t_tl = Table(rows, colWidths=[1.0 * inch, 0.9 * inch, 0.75 * inch, 1.15 * inch, 1.0 * inch, 2.5 * inch])
        t_tl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_tl)

    doc.build(story)
    return buf.getvalue()

