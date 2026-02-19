"""
ui/dashboard_tab.py — Dashboard Tab for Gradio UI

This module builds the Dashboard tab showing usage analytics:
    - Total calls, tokens, and estimated cost
    - Breakdown by model, provider, and feature
    - Success rate and average latency
    - Recent activity log

UI LAYOUT:
    ┌──────────┬──────────┬──────────┬──────────┐
    │  Calls   │  Tokens  │   Cost   │ Latency  │  ← Stat cards
    └──────────┴──────────┴──────────┴──────────┘
    ┌─────────────────────┬──────────────────────┐
    │  By Model           │  By Provider         │  ← Breakdowns
    │  (table)            │  (table)             │
    └─────────────────────┴──────────────────────┘
    ┌────────────────────────────────────────────┐
    │  Recent Activity (last 20 interactions)    │  ← Log
    └────────────────────────────────────────────┘

WHY LOCAL ANALYTICS?
    Tools like LangSmith and Langfuse are great for production, but they
    require accounts, API keys, and cloud connectivity. Our local dashboard
    works offline, costs nothing, and teaches the same concepts:
    - Data collection (logging)
    - Aggregation (summarizing)
    - Visualization (displaying insights)
"""

import gradio as gr

from monitoring.tracker import get_summary_stats, load_all_records


def create_dashboard_tab() -> gr.Blocks:
    """
    Build and return the Dashboard tab.

    Returns:
        A gr.Blocks component with analytics displays.
    """

    def _build_dashboard() -> tuple[str, list, list, list, str]:
        """
        Build all dashboard components from current data.

        Returns:
            Tuple of (stats_markdown, model_table, provider_table,
                      feature_table, recent_log).
        """
        stats = get_summary_stats()

        # ── Stat Cards (as markdown) ──
        stats_md = (
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| **Total LLM Calls** | {stats['total_calls']} |\n"
            f"| **Total Tokens** | {stats['total_tokens']:,} |\n"
            f"| **Estimated Cost** | ${stats['total_cost']:.4f} |\n"
            f"| **Avg Latency** | {stats['avg_latency_ms']:.0f}ms |\n"
            f"| **Success Rate** | {stats['success_rate']}% |\n"
        )

        # ── By Model Table ──
        model_rows = []
        for model, data in stats.get("by_model", {}).items():
            model_rows.append([
                model,
                data["calls"],
                f"{data['tokens']:,}",
                f"${data['cost']:.4f}",
            ])

        # ── By Provider Table ──
        provider_rows = []
        for provider, data in stats.get("by_provider", {}).items():
            provider_rows.append([
                provider,
                data["calls"],
                f"{data['tokens']:,}",
                f"${data['cost']:.4f}",
            ])

        # ── By Feature Table ──
        feature_rows = []
        for feature, data in stats.get("by_feature", {}).items():
            feature_rows.append([
                feature,
                data["calls"],
                f"{data['tokens']:,}",
                f"${data['cost']:.4f}",
            ])

        # ── Recent Activity ──
        records = load_all_records()
        recent = records[-20:][::-1]  # Last 20, newest first

        if recent:
            log_lines = ["| Time | Model | Tokens | Cost | Latency | Status |",
                         "|------|-------|--------|------|---------|--------|"]
            for r in recent:
                ts = r.get("timestamp", "")[:19]  # Trim to seconds
                model = r.get("model", "?")
                if len(model) > 25:
                    model = model[:22] + "..."
                tokens = r.get("total_tokens", 0)
                cost = r.get("estimated_cost", 0)
                latency = r.get("latency_ms", 0)
                status = "OK" if r.get("success", True) else "FAIL"
                log_lines.append(
                    f"| {ts} | {model} | {tokens} | ${cost:.4f} | {latency:.0f}ms | {status} |"
                )
            recent_md = "\n".join(log_lines)
        else:
            recent_md = "*No activity logged yet. Start chatting to see data here!*"

        return stats_md, model_rows, provider_rows, feature_rows, recent_md

    # ── Build the Gradio Layout ────────────────────────────────────

    with gr.Blocks() as dashboard_tab:
        gr.Markdown("## Dashboard")
        gr.Markdown(
            "Monitor your LLM usage, costs, and performance. "
            "Data is collected locally from all chat and agent interactions."
        )

        refresh_btn = gr.Button("Refresh Dashboard", variant="primary")

        # Overview stats
        gr.Markdown("### Overview")
        stats_display = gr.Markdown()

        with gr.Row():
            # By Model
            with gr.Column():
                gr.Markdown("### Usage by Model")
                model_table = gr.Dataframe(
                    headers=["Model", "Calls", "Tokens", "Cost"],
                    interactive=False,
                )

            # By Provider
            with gr.Column():
                gr.Markdown("### Usage by Provider")
                provider_table = gr.Dataframe(
                    headers=["Provider", "Calls", "Tokens", "Cost"],
                    interactive=False,
                )

        # By Feature
        gr.Markdown("### Usage by Feature")
        feature_table = gr.Dataframe(
            headers=["Feature", "Calls", "Tokens", "Cost"],
            interactive=False,
        )

        # Recent log
        gr.Markdown("### Recent Activity")
        recent_display = gr.Markdown()

        # Wire refresh button
        refresh_btn.click(
            fn=_build_dashboard,
            outputs=[
                stats_display, model_table, provider_table,
                feature_table, recent_display,
            ],
        )

        # Load data on tab render
        dashboard_tab.load(
            fn=_build_dashboard,
            outputs=[
                stats_display, model_table, provider_table,
                feature_table, recent_display,
            ],
        )

    return dashboard_tab
