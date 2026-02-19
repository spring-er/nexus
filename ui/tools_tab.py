"""
ui/tools_tab.py — Tools Tab for Gradio UI

This module builds the Tools tab where users can:
    - Browse all available tools with descriptions
    - Test tools directly with custom inputs
    - See what the agent has access to

UI DESIGN:
    The tab has two sections:
    1. Tool Directory — a table listing all tools with descriptions
    2. Tool Tester — select a tool, provide input, run it, see output

WHY A TOOLS TAB?
    Transparency: Users should know what tools the agent can use.
    Debugging: If the agent gives a wrong answer, users can test
    tools individually to see if the tool is the problem.
    Learning: Users can explore tool capabilities interactively.
"""

import gradio as gr

from agents.tools import ALL_TOOLS
from agents.tool_library import calculator, run_python, datetime_tool, save_to_file
from agents.tools import web_search, scrape_webpage, knowledge_base_search, summarize_text


def create_tools_tab() -> gr.Blocks:
    """
    Build and return the Tools tab as a Gradio Blocks component.

    Returns:
        A gr.Blocks component with tool directory and tester.
    """

    # Build tool info for the directory table
    tool_info = []
    for t in ALL_TOOLS:
        # Extract first line of docstring as short description
        desc = t.description.strip().split("\n")[0]
        tool_info.append([t.name, desc])

    # Map tool names to functions for the tester
    tool_map = {t.name: t for t in ALL_TOOLS}

    def _run_tool(tool_name: str, tool_input: str) -> str:
        """Run a selected tool with the provided input."""
        if not tool_name:
            return "Please select a tool."
        if not tool_input or not tool_input.strip():
            return "Please provide an input."

        tool_fn = tool_map.get(tool_name)
        if not tool_fn:
            return f"Tool '{tool_name}' not found."

        try:
            # Tools expect specific argument names.
            # We map each tool to its expected parameter.
            param_map = {
                "web_search": "query",
                "scrape_webpage": "url",
                "knowledge_base_search": "query",
                "summarize_text": "text",
                "calculator": "expression",
                "run_python": "code",
                "datetime_tool": "query",
                "save_to_file": None,  # Special handling (2 params)
            }

            param_name = param_map.get(tool_name, "query")

            if tool_name == "save_to_file":
                # Expect "filename|||content" format
                if "|||" in tool_input:
                    parts = tool_input.split("|||", 1)
                    result = tool_fn.invoke({
                        "filename": parts[0].strip(),
                        "content": parts[1].strip(),
                    })
                else:
                    return (
                        "For save_to_file, use format: "
                        "filename.txt|||content to save"
                    )
            else:
                result = tool_fn.invoke({param_name: tool_input.strip()})

            return str(result)

        except Exception as e:
            return f"**Error:** {str(e)}"

    # ── Build the Gradio Layout ────────────────────────────────────

    with gr.Blocks() as tools_tab:
        gr.Markdown("## Tools")
        gr.Markdown(
            "These tools are available to the Research Agent. "
            "You can also test them directly below."
        )

        # Tool directory
        gr.Markdown("### Tool Directory")
        gr.Dataframe(
            value=tool_info,
            headers=["Tool Name", "Description"],
            interactive=False,
            wrap=True,
        )

        gr.Markdown("---")

        # Tool tester
        gr.Markdown("### Tool Tester")
        gr.Markdown("Select a tool and provide input to test it directly.")

        with gr.Row():
            with gr.Column(scale=1):
                tool_selector = gr.Dropdown(
                    choices=[t.name for t in ALL_TOOLS],
                    label="Select Tool",
                    info="Choose a tool to test",
                )

                tool_input = gr.Textbox(
                    label="Input",
                    placeholder=(
                        "Enter input for the tool...\n"
                        "calculator: 347 * 829\n"
                        "datetime_tool: +30 days\n"
                        "run_python: print('hello')"
                    ),
                    lines=4,
                )

                run_btn = gr.Button("Run Tool", variant="primary")

            with gr.Column(scale=2):
                tool_output = gr.Textbox(
                    label="Output",
                    lines=10,
                    interactive=False,
                )

        # Examples for quick testing
        gr.Examples(
            examples=[
                ["calculator", "sqrt(144) + pi * 2"],
                ["datetime_tool", "today"],
                ["datetime_tool", "+90 days"],
                ["run_python", "for i in range(5):\n    print(f'Square of {i}: {i**2}')"],
                ["calculator", "factorial(10)"],
            ],
            inputs=[tool_selector, tool_input],
            label="Quick Examples",
        )

        # Event handler
        run_btn.click(
            fn=_run_tool,
            inputs=[tool_selector, tool_input],
            outputs=[tool_output],
        )

    return tools_tab
