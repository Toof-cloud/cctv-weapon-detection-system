"""Paginated PDF report retaining machine results and analyst decisions separately."""
from pathlib import Path
from xml.sax.saxutils import escape


def write_pdf(report, path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

    regular, bold = "Helvetica", "Helvetica-Bold"
    # Windows supplies this Unicode family; keep standard-font fallback portable.
    font_dir = Path("C:/Windows/Fonts")
    if (font_dir / "arial.ttf").exists() and (font_dir / "arialbd.ttf").exists():
        if "Forensikada" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("Forensikada", str(font_dir / "arial.ttf")))
            pdfmetrics.registerFont(TTFont("ForensikadaBold", str(font_dir / "arialbd.ttf")))
            pdfmetrics.registerFontFamily("Forensikada", normal="Forensikada", bold="ForensikadaBold")
        regular, bold = "Forensikada", "ForensikadaBold"
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = regular
    styles.add(ParagraphStyle("ReportBody", fontName=regular, fontSize=9, leading=13, spaceAfter=6,
                              splitLongWords=True, alignment=TA_LEFT))
    styles.add(ParagraphStyle("ReportTitle", fontName=bold, fontSize=23, leading=28, textColor=colors.HexColor("#2f318e"), spaceAfter=12))
    styles.add(ParagraphStyle("Section", fontName=bold, fontSize=12, leading=17, spaceBefore=14, spaceAfter=7, keepWithNext=True))
    body = styles["ReportBody"]

    def p(value, style=body):
        return Paragraph(escape("Not recorded" if value is None else str(value)).replace("\n", "<br/>"), style)

    def pair(label, value):
        return [p(label), p(value)]

    width = A4[0] - 84
    def field_table(rows):
        table = Table(rows, colWidths=[145, width - 145], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f3f8")),
            ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor("#dce0e8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    source, processing, summary = report["source"], report["processing"], report["summary"]
    story = [p("FORENSIKADA / FORENSIC ANALYSIS", styles["Section"]),
             p("Observation review report", styles["ReportTitle"]),
             p(f"Report: {report['report_id']}\nGenerated (UTC): {report['generated_at_utc']}"),
             p("Video and detection information", styles["Section"]),
             field_table([
                 pair("Video filename", source["name"]), pair("Source reference", source["reference"]),
                 pair("Model used", processing["model_reference"]),
                 pair("Video timing", f"{source['fps']:g} FPS; {source['duration_seconds']:.3f} seconds"),
                 pair("Analysis completed (UTC)", processing["completed_at_utc"]),
                 pair("Threshold / frames", f"{processing['confidence_threshold']:.0%}; {processing['analyzed_frames']} analyzed frames"),
                 pair("Detection summary", f"{summary['total_frame_detections']} observations; {summary['frames_with_detections']} positive frames; "
                      + "; ".join(f"{key}: {value}" for key, value in summary["counts_per_class"].items())),
                 pair("Analyst review summary", report["review_summary"]),
                 pair("Camera ID / recording time", "Not recorded by the current pipeline"),
             ]),
             p("Interpretation", styles["Section"]),
             p("Automated Validation Result and Analyst Review Decision are independent fields. "
               "Not performed means the detection pipeline did not supply a separate observation-validation result. "
               "A model confidence score is not an analyst decision. Every observation is retained, including rejected and uncertain observations."),
             p("Frames are zero-based. Times are video-relative offsets, not recording dates. "
               "Boxes are [x1, y1, x2, y2] in source-frame pixels. Counts are frame observations, not unique weapons."),
             p("Observations and analyst reviews", styles["Section"])]
    for row in report["detections"]:
        from mockup_ui.observation_review import status_label
        heading = p(f"Observation {row['observation_id']}", styles["Section"])
        table = field_table([
            pair("Detection details", f"{row['object_label']} | Frame {row['frame_number']} | {row['video_relative_timestamp_seconds']:.2f}s\n"
                 f"Confidence {row['confidence_score']:.2%} | Box [{row['x1']}, {row['y1']}, {row['x2']}, {row['y2']}]"),
            pair("Automated Validation Result", status_label(row["automated_validation_status"])),
            pair("Analyst Review Decision", row["analyst_decision"] or "Not reviewed"),
            pair("Review timestamp (UTC)", row["reviewed_at"]),
        ])
        story.extend([KeepTogether([heading, table]),
                      p("Analyst notes", styles["Section"]), p(row["analyst_notes"] or "No notes provided."),
                      p("Source references", styles["Section"]),
                      p(row["source_video_reference"]), p(f"Model: {row['model_reference']}"), Spacer(1, 6)])
    if not report["detections"]:
        story.append(p("No observations met the configured detection threshold. There are no analyst decisions to review."))
    story.extend([p("Traceability", styles["Section"]),
                  p(f"Saved summary: {report['artifacts']['saved_summary']['path']}"),
                  p(f"Source SHA-256 at report generation: {source['file_at_report_generation']['sha256'] or 'Unavailable'}"),
                  p(report["definitions"]["fingerprints"])])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor("#66717d"))
        canvas.drawString(42, 23, "Forensikada | Automated results and analyst decisions")
        canvas.drawRightString(A4[0] - 42, 23, f"Page {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=42, leftMargin=42,
                                 topMargin=36, bottomMargin=42, title="Forensic observation review report", author="Forensikada")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return Path(path)
