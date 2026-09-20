"""Consolidated, paginated forensic report for a saved camera session."""
from pathlib import Path
from xml.sax.saxutils import escape

from mockup_ui.report_metrics import REPORT_DISCLAIMER


def write_session_pdf(path: Path, manifest: dict) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    page_size = landscape(A4)
    doc = SimpleDocTemplate(str(path), pagesize=page_size, leftMargin=38, rightMargin=38,
                            topMargin=28, bottomMargin=30)
    usable_width = page_size[0] - 76
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("SessionTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=19, leading=23, textColor=colors.HexColor("#2f318e"), spaceAfter=6))
    styles.add(ParagraphStyle("SessionHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=11, leading=15, textColor=colors.HexColor("#303841"),
                              spaceBefore=5, spaceAfter=3, keepWithNext=True))
    styles.add(ParagraphStyle("SessionBody", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=8, leading=11, spaceAfter=2, wordWrap="CJK"))
    styles.add(ParagraphStyle("SessionCell", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=7, leading=9, wordWrap="CJK"))
    styles.add(ParagraphStyle("SessionHeader", parent=styles["SessionCell"], fontName="Helvetica-Bold"))

    def paragraph(value, style="SessionBody"):
        value = "Not recorded" if value is None or value == "" else str(value)
        return Paragraph(escape(value).replace("\n", "<br/>"), styles[style])

    def heading(value):
        return paragraph(value, "SessionHeading")

    story = [paragraph("Multi-camera forensic report", "SessionTitle"),
             paragraph(REPORT_DISCLAIMER), Spacer(1, 5), heading("Session and detection information")]
    metrics = manifest["metrics"]
    counts = metrics["counts"]
    mccr = f"{metrics['mccr_percent']:.2f}%" if metrics["mccr_percent"] is not None else "N/A"
    overview = [
        ("Incident ID", manifest.get("scene_id") or "Not specified"),
        ("Generated (UTC)", manifest["generated_at_utc"]),
        ("Camera recordings", len(manifest["cameras"])),
        ("Total observations", len(manifest["observations"])),
        ("MCCR", mccr),
        ("Cross-view statuses", ", ".join(f"{key}: {counts.get(key, 0)}" for key in
                                      ("Corroborated", "Not Corroborated", "Uncertain", "Not Applicable"))),
        ("Alignment", f"{'Confirmed by analyst' if manifest['alignment_confirmed_by_analyst'] else 'Not confirmed'}; "
                       f"{manifest.get('alignment_method') or 'no reference recorded'}; "
                       f"matching window +/- {manifest['matching_window_seconds']:.2f} s"),
        ("Time basis", manifest["time_basis"]),
    ]
    pairs = Table([[paragraph(name, "SessionHeader"), paragraph(value)] for name, value in overview],
                  colWidths=[130, usable_width - 130], hAlign="LEFT")
    pairs.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f2f8")),
        ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#dce1eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.extend([pairs, heading("Camera sources and processing")])
    for camera in manifest["cameras"]:
        tcr = camera.get("tcr", {})
        tcr_value = tcr.get("value_percent")
        tcr_display = f"{tcr_value:.2f}%" if tcr_value is not None else "N/A"
        story.append(paragraph(f"{camera['camera_id']} - {Path(camera['source_video']).name}", "SessionHeader"))
        story.append(paragraph(f"Source: {camera['source_video']} | {camera['width']} x {camera['height']} | "
                               f"{camera['fps']:.2f} FPS | {camera['frame_count']} frames | "
                               f"{camera['observation_count']} observations | Location: {camera['location'] or 'Not recorded'} | "
                               f"Offset: {camera['offset_seconds']:+.3f} s"))
        if camera.get("original_source_video"):
            story.append(paragraph(f"Original recording before BasicVSR++ enhancement: {camera['original_source_video']}"))
        story.append(paragraph(f"Run: {camera['run_id']} | Model: {Path(camera['model_path']).name} | "
                               f"Confidence threshold: {camera['confidence_threshold']:.0%} | "
                               f"Device: {camera['device']} | Analyzed frames: {camera['analyzed_frames']} | "
                               f"Frames with detections: {camera['frames_with_detections']} | "
                               f"TCR: {tcr_display}"))
        story.append(paragraph(f"TCR interpretation: {tcr.get('reason') or 'Not available'} | "
                               f"Annotated video and camera records: {camera['export_folder']}"))

    story.append(heading("Object detection observations and reviews"))
    story.append(paragraph("Each row is a frame observation. Confidence and cross-view matching remain reviewable. "
                           "The CSV and JSON files contain all record identifiers and source references."))
    headers = ["Session time", "Camera", "Video time", "Frame", "Object", "Confidence", "Box [x1,y1,x2,y2]", "Cross-view", "Analyst"]
    widths = [69, 56, 69, 44, 60, 64, 126, 118, usable_width - 606]
    records = [[paragraph(name, "SessionHeader") for name in headers]]
    for row in manifest["observations"]:
        records.append([paragraph(value, "SessionCell") for value in (
            f"{row['session_seconds']:.3f} s", row["camera_id"], f"{row['video_seconds']:.3f} s",
            row["frame_number"], row["object_label"].title(), f"{row['confidence']:.1%}",
            str(row["box"]), row["corroboration_status"], row.get("analyst_decision", "Not reviewed"))])
    timeline = LongTable(records, colWidths=widths, repeatRows=1, hAlign="LEFT")
    timeline.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eaf5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fc")]),
        ("LINEBELOW", (0, 0), (-1, -1), .25, colors.HexColor("#dce1eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(timeline)
    reviewed = [row for row in manifest["observations"] if row.get("analyst_decision") not in (None, "Not reviewed")]
    story.append(heading("Analyst review information"))
    if reviewed:
        for row in reviewed:
            story.append(paragraph(f"{row['observation_id']} - {row['analyst_decision']} | "
                                   f"Reviewed: {row.get('reviewed_at') or 'Not recorded'} | "
                                   f"Notes: {row.get('analyst_notes') or 'None'}"))
    else:
        story.append(paragraph("No analyst decisions have been recorded for this session."))
    story.extend([heading("Source references and traceability"),
                  paragraph("Source paths are recorded above. Session metadata and complete observations are preserved in "
                            "session.json and combined_observations.csv. Each camera folder contains its annotated video, "
                            "detections, structured forensic records, and saved analyst review snapshot. "
                            "This PDF covers the entire multi-camera session.")])
    for camera in manifest["cameras"]:
        fingerprints = camera.get("fingerprints", {})
        story.append(paragraph(f"{camera['camera_id']} fingerprints at report generation - "
                               f"source video SHA-256: {fingerprints.get('source_video') or 'Unavailable'}; "
                               f"detections CSV SHA-256: {fingerprints.get('detections_csv') or 'Unavailable'}; "
                               f"summary JSON SHA-256: {fingerprints.get('summary_json') or 'Unavailable'}."))
    story.append(paragraph("Source timestamps are not supplied by the current video pipeline. Session times use "
                           "video-relative offsets and analyst-provided alignment; they are not wall-clock timestamps."))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#66717d"))
        canvas.drawString(38, 22, "Forensikada | Multi-camera forensic report")
        canvas.drawRightString(page_size[0] - 38, 22, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
