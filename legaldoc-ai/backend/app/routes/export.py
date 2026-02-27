"""Export route — PDF, CSV, and JSON document report generation."""

from __future__ import annotations

import csv
import io
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response

from app.dependencies import get_current_user
from app.services import document_service

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_FORMATS = {"pdf", "csv", "json"}


# ── Helpers ───────────────────────────────────────────


def _parse_json_field(doc: dict, field: str) -> dict | list:
    """Safely parse a JSON-string field from a document row."""
    val = doc.get(field)
    if val is None:
        return {}
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}


def _ensure_completed(doc: dict) -> None:
    """Raise 422 if the document is not in completed status."""
    if doc.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document has not finished processing (status: {doc.get('status')}). Export requires completed analysis.",
        )


# ── JSON export ───────────────────────────────────────


def _export_json(doc: dict) -> Response:
    """Build a JSON report from a completed document."""
    metadata = _parse_json_field(doc, "extracted_metadata")
    clause_analysis = _parse_json_field(doc, "clause_analysis")
    summary = _parse_json_field(doc, "summary")
    parties = _parse_json_field(doc, "parties")

    report = {
        "document": {
            "id": doc.get("id"),
            "filename": doc.get("filename"),
            "file_type": doc.get("file_type"),
            "document_type": doc.get("document_type"),
            "page_count": doc.get("page_count"),
            "created_at": doc.get("created_at"),
            "completed_at": doc.get("completed_at"),
            "processing_time": doc.get("processing_time"),
        },
        "classification": metadata,
        "parties": parties,
        "clause_analysis": clause_analysis,
        "summary": summary,
        "overall_risk_rating": doc.get("overall_risk_rating"),
    }

    filename = doc.get("filename", "document").rsplit(".", 1)[0]

    return Response(
        content=json.dumps(report, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}_report.json"'},
    )


# ── CSV export ────────────────────────────────────────


def _export_csv(doc: dict) -> Response:
    """Build a CSV report (one row per clause + risks)."""
    clause_analysis = _parse_json_field(doc, "clause_analysis")

    buf = io.StringIO()
    writer = csv.writer(buf)

    # ── Document header section
    writer.writerow(["Document Report"])
    writer.writerow(["Filename", doc.get("filename", "")])
    writer.writerow(["Type", doc.get("document_type", "")])
    writer.writerow(["Risk Rating", doc.get("overall_risk_rating", "")])
    writer.writerow(["Pages", doc.get("page_count", "")])
    writer.writerow([])

    # ── Clauses
    clauses = clause_analysis.get("clauses", []) if isinstance(clause_analysis, dict) else []
    writer.writerow(["Clauses"])
    writer.writerow(["Title", "Category", "Importance", "Page", "Text"])
    for c in clauses:
        writer.writerow([
            c.get("clause_title", ""),
            c.get("category", ""),
            c.get("importance", ""),
            c.get("page_reference", ""),
            c.get("clause_text", ""),
        ])
    writer.writerow([])

    # ── Risks
    risks = clause_analysis.get("risks", []) if isinstance(clause_analysis, dict) else []
    writer.writerow(["Risks"])
    writer.writerow(["Title", "Severity", "Clause Reference", "Description", "Recommendation"])
    for r in risks:
        writer.writerow([
            r.get("title", ""),
            r.get("severity", ""),
            r.get("clause_reference", ""),
            r.get("description", ""),
            r.get("recommendation", ""),
        ])

    filename = doc.get("filename", "document").rsplit(".", 1)[0]

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}_report.csv"'},
    )


# ── PDF export ────────────────────────────────────────


def _export_pdf(doc: dict) -> Response:
    """Build a professional PDF report using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    pdf = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=12)
    heading_style = ParagraphStyle("SectionHead", parent=styles["Heading2"], fontSize=14, spaceBefore=18, spaceAfter=6)
    body_style = styles["BodyText"]
    small_style = ParagraphStyle("Small", parent=body_style, fontSize=8, textColor=colors.grey)

    elements: list = []

    # ── Title
    doc_title = doc.get("title") or doc.get("filename", "Document Report")
    elements.append(Paragraph(f"Legal Analysis Report: {doc_title}", title_style))
    elements.append(Paragraph(
        f"Type: {doc.get('document_type', 'N/A')} &nbsp;|&nbsp; "
        f"Risk: {doc.get('overall_risk_rating', 'N/A')} &nbsp;|&nbsp; "
        f"Pages: {doc.get('page_count', 'N/A')}",
        small_style,
    ))
    elements.append(Spacer(1, 12))

    # ── Executive Summary
    summary = _parse_json_field(doc, "summary")
    exec_text = summary.get("executive_summary") or doc.get("executive_summary") or ""
    if exec_text:
        elements.append(Paragraph("Executive Summary", heading_style))
        for para in exec_text.split("\n"):
            para = para.strip()
            if para:
                elements.append(Paragraph(para, body_style))
                elements.append(Spacer(1, 4))

    # ── Clauses table
    clause_analysis = _parse_json_field(doc, "clause_analysis")
    clauses = clause_analysis.get("clauses", []) if isinstance(clause_analysis, dict) else []
    if clauses:
        elements.append(Paragraph("Clauses Identified", heading_style))
        table_data = [["Title", "Category", "Importance"]]
        for c in clauses:
            table_data.append([
                c.get("clause_title", ""),
                c.get("category", ""),
                c.get("importance", ""),
            ])
        t = Table(table_data, colWidths=[2.5 * inch, 2 * inch, 1.2 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))

    # ── Risks table
    risks = clause_analysis.get("risks", []) if isinstance(clause_analysis, dict) else []
    if risks:
        elements.append(Paragraph("Risks Identified", heading_style))

        severity_colors = {
            "critical": colors.HexColor("#dc2626"),
            "high": colors.HexColor("#ea580c"),
            "medium": colors.HexColor("#ca8a04"),
            "low": colors.HexColor("#16a34a"),
        }

        table_data = [["Title", "Severity", "Recommendation"]]
        for r in risks:
            table_data.append([
                r.get("title", ""),
                r.get("severity", ""),
                r.get("recommendation", ""),
            ])
        t = Table(table_data, colWidths=[1.8 * inch, 1 * inch, 3 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))

    # ── Key Terms
    key_terms = summary.get("key_terms", [])
    if key_terms:
        elements.append(Paragraph("Key Terms", heading_style))
        for kt in key_terms:
            term = kt.get("term", "")
            defn = kt.get("definition", "")
            elements.append(Paragraph(f"<b>{term}</b>: {defn}", body_style))
            elements.append(Spacer(1, 2))

    # ── Action Items
    action_items = summary.get("action_items", [])
    if action_items:
        elements.append(Paragraph("Action Items", heading_style))
        table_data = [["Action", "Priority", "Deadline"]]
        for a in action_items:
            table_data.append([
                a.get("action", ""),
                a.get("priority", ""),
                a.get("deadline", ""),
            ])
        t = Table(table_data, colWidths=[3.5 * inch, 1 * inch, 1.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))

    # ── Recommendation
    rec = summary.get("recommendation", "")
    if rec:
        elements.append(Paragraph("Recommendation", heading_style))
        elements.append(Paragraph(rec, body_style))

    # Build PDF
    pdf.build(elements)

    filename = doc.get("filename", "document").rsplit(".", 1)[0]

    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}_report.pdf"'},
    )


# ── GET /{id}/export ──────────────────────────────────


@router.get("/{document_id}/export")
def export_document(
    document_id: str,
    format: str = Query(..., description="Export format: pdf, csv, or json"),
    current_user: dict = Depends(get_current_user),
):
    """Export a completed document's analysis as PDF, CSV, or JSON."""
    fmt = format.lower()
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: '{fmt}'. Allowed: {', '.join(sorted(ALLOWED_FORMATS))}",
        )

    doc = document_service.get_document(document_id, current_user["id"])
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    _ensure_completed(doc)

    exporters = {
        "json": _export_json,
        "csv": _export_csv,
        "pdf": _export_pdf,
    }

    return exporters[fmt](doc)
