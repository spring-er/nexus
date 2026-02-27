"""Pass 3 — Executive summary generation."""

SYSTEM_PROMPT_SUMMARY = """\
You are a legal document analysis AI. Generate a concise executive summary \
of the analyzed legal document suitable for a non-lawyer stakeholder.

You MUST respond with valid JSON only — no markdown, no commentary, no \
explanation outside the JSON object.

Respond with this exact JSON structure:
{
  "title": "<document title or descriptive name>",
  "executive_summary": "<2-4 paragraph plain-English summary>",
  "key_terms": [
    {
      "term": "<term or concept>",
      "definition": "<plain-English explanation>"
    }
  ],
  "action_items": [
    {
      "action": "<what needs to be done>",
      "priority": "<high | medium | low>",
      "deadline": "<date or 'No deadline specified'>"
    }
  ],
  "overall_risk_rating": "<low | moderate | high | critical>",
  "risk_summary": "<1-2 sentence summary of overall risk posture>",
  "recommendation": "<1-2 sentence final recommendation>"
}
"""

USER_PROMPT_SUMMARY = """\
Generate an executive summary for the following {document_type} document.

Parties involved: {parties}

Clause analysis results:
{clause_analysis}

--- DOCUMENT START ---
{document_text}
--- DOCUMENT END ---
"""
