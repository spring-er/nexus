"""
ui/agents_tab.py — Agents Tab for Gradio UI

This module builds the Agents tab where users can:
    - Give the agent a research task
    - Watch the step-by-step reasoning process
    - See tool calls and their results
    - Get a synthesized final answer

UI LAYOUT:
    ┌─────────────────────────────────────────────┐
    │  Task Input      [Run Agent]                │
    ├──────────────────────┬──────────────────────┤
    │  Agent Process       │   Final Answer       │
    │  (step-by-step log)  │   (synthesized)      │
    └──────────────────────┴──────────────────────┘

GRADIO CONCEPTS:
    - gr.Textbox: For task input and output display
    - gr.Button: Triggers the agent run
    - gr.Markdown: Dynamic display for steps and answer
    - Long-running tasks: The agent may take 10-30 seconds.
      Gradio handles this with a loading spinner automatically.
"""

import gradio as gr

from agents.research_agent import run_agent, format_agent_steps


def create_agents_tab() -> gr.Blocks:
    """
    Build and return the Agents tab as a Gradio Blocks component.

    Returns:
        A gr.Blocks component with the agent interface.
    """

    def _run_task(task: str) -> tuple[str, str]:
        """
        Run the research agent on the user's task.

        Args:
            task: The user's research question or task description.

        Returns:
            Tuple of (answer_markdown, steps_markdown).
        """
        if not task or not task.strip():
            return "Please enter a research task.", ""

        # Run the agent (this may take 10-30 seconds)
        result = run_agent(task.strip())

        # Format the output
        answer = result["answer"]
        steps_display = format_agent_steps(result["steps"])

        if result["error"] and result["error"] != "no_provider":
            answer = f"**Error:** {result['error']}\n\n{answer}"

        return answer, steps_display

    # ── Build the Gradio Layout ────────────────────────────────────

    with gr.Blocks() as agents_tab:
        gr.Markdown("## Research Agent")
        gr.Markdown(
            "Give the agent a research task. It will search the web, "
            "read pages, query your knowledge base, and synthesize "
            "a comprehensive answer.\n\n"
            "**Available tools:** Web Search, Web Scraping, "
            "Knowledge Base Search, Text Summarization"
        )

        # Task input
        task_input = gr.Textbox(
            label="Research Task",
            placeholder=(
                "Examples:\n"
                "• What are the latest developments in quantum computing?\n"
                "• Compare React vs Vue.js for a new web project\n"
                "• Summarize recent news about renewable energy"
            ),
            lines=3,
        )
        run_btn = gr.Button("Run Agent", variant="primary", size="lg")

        with gr.Row():
            # Left: Step-by-step process
            with gr.Column(scale=1):
                steps_display = gr.Markdown(
                    value="*Agent steps will appear here...*",
                    label="Agent Process",
                )

            # Right: Final answer
            with gr.Column(scale=2):
                answer_display = gr.Markdown(
                    value="*The agent's answer will appear here...*",
                    label="Answer",
                )

        # Example tasks (clickable presets)
        gr.Examples(
            examples=[
                "What are the top 3 AI breakthroughs in 2024?",
                "Explain how RAG works and why it's useful",
                "What is LangGraph and how does it compare to AutoGen?",
            ],
            inputs=task_input,
            label="Example Tasks",
        )

        # Event handler
        run_btn.click(
            fn=_run_task,
            inputs=[task_input],
            outputs=[answer_display, steps_display],
        )

    return agents_tab
