"""
ui/knowledge_tab.py — Knowledge Base Management Tab

This module builds the Gradio tab where users manage their personal
knowledge base. They can:
    - Upload files (PDF, TXT, MD)
    - Add web URLs
    - Add YouTube videos
    - View indexed documents
    - Delete documents
    - See knowledge base statistics

GRADIO CONCEPTS USED:
    - gr.File: File upload component
    - gr.Textbox: Text input for URLs
    - gr.Button: Clickable buttons that trigger Python functions
    - gr.Dataframe: Tabular display of documents
    - gr.Markdown: Dynamic text display for stats
    - Event handlers: .click() connects a button to a function

HOW UPLOAD WORKS IN GRADIO:
    When a user uploads a file, Gradio saves it to a temporary directory
    and gives us the temp path. We need the original filename (for type
    detection), so we use the file object's .name attribute.
"""

import gradio as gr

from rag.ingestion import ingest_file, ingest_uploaded_file, ingest_url
from rag.youtube import ingest_youtube
from rag.vector_store import add_chunks, list_documents, delete_document, get_stats


def create_knowledge_tab() -> gr.Blocks:
    """
    Build and return the Knowledge Base management tab.

    Returns:
        A gr.Blocks component with upload, listing, and stats.
    """

    def _upload_file(file) -> str:
        """
        Handle file upload from Gradio.

        Gradio provides the file path. We ingest it (parse + chunk)
        and add it to the vector store.
        """
        if file is None:
            return "No file selected."

        try:
            # Gradio gives us the temp file path
            file_path = file.name if hasattr(file, "name") else str(file)
            # Extract the original filename from the path
            import os
            original_name = os.path.basename(file_path)

            chunks = ingest_uploaded_file(file_path, original_name)
            count = add_chunks(chunks)
            return (
                f"Successfully ingested **{original_name}**\n\n"
                f"- {count} chunks created and embedded\n"
                f"- Added to knowledge base"
            )
        except Exception as e:
            return f"**Error:** {str(e)}"

    def _add_url(url: str) -> str:
        """Handle URL ingestion."""
        if not url or not url.strip():
            return "Please enter a URL."

        url = url.strip()

        try:
            # Detect if it's a YouTube URL
            if "youtube.com" in url or "youtu.be" in url:
                chunks = ingest_youtube(url)
            else:
                chunks = ingest_url(url)

            count = add_chunks(chunks)
            return (
                f"Successfully ingested **{url}**\n\n"
                f"- {count} chunks created and embedded\n"
                f"- Added to knowledge base"
            )
        except Exception as e:
            return f"**Error:** {str(e)}"

    def _refresh_documents() -> list[list]:
        """Get the current list of documents for the dataframe."""
        docs = list_documents()
        if not docs:
            return []

        # Format for Gradio Dataframe: list of lists
        rows = []
        for doc in docs:
            type_label = {
                "pdf": "PDF", "txt": "TXT", "md": "MD",
                "url": "Web", "youtube": "YouTube",
            }.get(doc["source_type"], doc["source_type"])

            rows.append([
                doc["title"],
                type_label,
                doc["chunk_count"],
                doc["source"],
                doc["doc_id"],
            ])
        return rows

    def _delete_doc(doc_id: str) -> tuple[str, list[list]]:
        """Delete a document and return updated list."""
        if not doc_id or not doc_id.strip():
            return "Please enter a Document ID to delete.", _refresh_documents()

        try:
            count = delete_document(doc_id.strip())
            if count > 0:
                msg = f"Deleted {count} chunks for document `{doc_id}`"
            else:
                msg = f"No document found with ID `{doc_id}`"
            return msg, _refresh_documents()
        except Exception as e:
            return f"**Error:** {str(e)}", _refresh_documents()

    def _get_stats_display() -> str:
        """Format knowledge base statistics as markdown."""
        stats = get_stats()
        if stats["total_chunks"] == 0:
            return (
                "**Knowledge Base is empty.**\n\n"
                "Upload files, add URLs, or paste YouTube links to get started."
            )

        type_breakdown = "\n".join(
            f"  - {k.upper()}: {v} chunks"
            for k, v in stats["sources_by_type"].items()
        )

        return (
            f"**Documents:** {stats['total_documents']}\n\n"
            f"**Total Chunks:** {stats['total_chunks']}\n\n"
            f"**By Type:**\n{type_breakdown}"
        )

    # ── Build the Gradio Layout ────────────────────────────────────

    with gr.Blocks() as knowledge_tab:
        gr.Markdown("## Knowledge Base")
        gr.Markdown(
            "Upload documents to build your personal knowledge base. "
            "The chatbot will use these documents to answer your questions."
        )

        with gr.Row():
            # Left column: Upload controls
            with gr.Column(scale=1):
                gr.Markdown("### Add Documents")

                # File upload section
                file_upload = gr.File(
                    label="Upload File (PDF, TXT, MD)",
                    file_types=[".pdf", ".txt", ".md", ".markdown"],
                    type="filepath",
                )
                upload_btn = gr.Button("Upload & Index", variant="primary")
                upload_status = gr.Markdown("")

                gr.Markdown("---")

                # URL input section
                url_input = gr.Textbox(
                    label="Add URL or YouTube Link",
                    placeholder="https://example.com/article or https://youtube.com/watch?v=...",
                    lines=1,
                )
                url_btn = gr.Button("Fetch & Index", variant="primary")
                url_status = gr.Markdown("")

                gr.Markdown("---")

                # Stats display
                gr.Markdown("### Statistics")
                stats_display = gr.Markdown(_get_stats_display())
                refresh_stats_btn = gr.Button("Refresh Stats")

            # Right column: Document listing
            with gr.Column(scale=2):
                gr.Markdown("### Indexed Documents")

                doc_table = gr.Dataframe(
                    headers=["Title", "Type", "Chunks", "Source", "Doc ID"],
                    value=_refresh_documents(),
                    interactive=False,
                    wrap=True,
                )
                refresh_btn = gr.Button("Refresh List")

                gr.Markdown("---")

                # Delete section
                with gr.Row():
                    delete_input = gr.Textbox(
                        label="Document ID to Delete",
                        placeholder="Paste a Doc ID from the table above",
                        scale=3,
                    )
                    delete_btn = gr.Button(
                        "Delete Document", variant="stop", scale=1
                    )
                delete_status = gr.Markdown("")

        # ── Event Handlers ─────────────────────────────────────────

        # File upload: click button → ingest file → show status → refresh table
        upload_btn.click(
            fn=_upload_file,
            inputs=[file_upload],
            outputs=[upload_status],
        ).then(
            fn=_refresh_documents,
            outputs=[doc_table],
        ).then(
            fn=_get_stats_display,
            outputs=[stats_display],
        )

        # URL ingestion
        url_btn.click(
            fn=_add_url,
            inputs=[url_input],
            outputs=[url_status],
        ).then(
            fn=_refresh_documents,
            outputs=[doc_table],
        ).then(
            fn=_get_stats_display,
            outputs=[stats_display],
        )

        # Refresh buttons
        refresh_btn.click(fn=_refresh_documents, outputs=[doc_table])
        refresh_stats_btn.click(fn=_get_stats_display, outputs=[stats_display])

        # Delete document
        delete_btn.click(
            fn=_delete_doc,
            inputs=[delete_input],
            outputs=[delete_status, doc_table],
        ).then(
            fn=_get_stats_display,
            outputs=[stats_display],
        )

    return knowledge_tab
