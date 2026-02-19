"""
agents/research_agent.py — LangGraph Research Agent

This module implements a research agent using LangGraph. The agent can
search the web, read web pages, query the knowledge base, and synthesize
information to answer complex questions.

WHAT IS A LANGGRAPH AGENT?
    LangGraph models agent behavior as a graph (state machine):

    ┌──────────────┐          ┌──────────────┐
    │   AGENT      │ ──────>  │    TOOLS     │
    │  (LLM thinks)│ <──────  │(execute tool)│
    └──────┬───────┘          └──────────────┘
           │
           │ no more tool calls
           v
    ┌──────────────┐
    │     END      │
    │(final answer)│
    └──────────────┘

    The agent LOOPS between thinking and using tools until it decides
    it has enough information to give a final answer.

THE REACT PATTERN:
    ReAct = Reasoning + Acting. The agent:
    1. REASONS about what to do ("I need to search for X")
    2. ACTS by calling a tool (web_search("X"))
    3. OBSERVES the result ("I found these facts...")
    4. REPEATS until confident in its answer

HOW THIS CONNECTS TO PHASE 1:
    We reuse the same LLM providers from Phase 1! LangGraph's
    ChatOpenAI class works with any OpenAI-compatible API (OpenRouter,
    Groq, etc.), so our existing API keys and model registry work here.

ARCHITECTURE DECISION — LangGraph vs. plain loop:
    We COULD write a simple while loop (call LLM → check for tool calls →
    execute → repeat). LangGraph adds:
    - State management (tracks the full conversation)
    - Graph visualization (you can see the flow)
    - Checkpointing (resume interrupted tasks)
    - Industry-standard pattern (good for your portfolio)
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agents.tools import ALL_TOOLS
from config import settings
from llm_providers.system_prompts import AGENT_SYSTEM_PROMPT


def _get_agent_llm() -> ChatOpenAI | None:
    """
    Create an LLM instance for the agent using available providers.

    LangGraph uses LangChain's ChatOpenAI class, which works with any
    OpenAI-compatible API. We try providers in order of preference:
    1. Groq (fast inference, free tier)
    2. OpenRouter (many models, some free)

    WHY ChatOpenAI?
        LangGraph needs an LLM that supports tool calling (function calling).
        ChatOpenAI supports this natively. It's not just for OpenAI — any
        OpenAI-compatible API works by changing the base_url.

    Returns:
        A ChatOpenAI instance, or None if no provider is available.
    """
    # Try Groq first — fastest inference, great for agents
    if settings.groq_api_key:
        return ChatOpenAI(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,  # Low temp for agents (we want reliability)
            max_tokens=2048,
        )

    # Try OpenRouter — wide model selection
    if settings.openrouter_api_key:
        return ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct:free",
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_tokens=2048,
        )

    return None


def create_research_agent():
    """
    Create a LangGraph research agent with tools.

    This uses LangGraph's create_react_agent() — a pre-built graph
    that implements the ReAct pattern:
        Agent (think) → Tools (act) → Agent (observe) → ... → End

    create_react_agent() handles:
        - The graph structure (nodes + edges)
        - State management (message history)
        - Tool call parsing and execution
        - Loop termination (when agent gives final answer)

    Returns:
        A tuple of (agent_graph, llm) or (None, None) if no LLM is available.
    """
    llm = _get_agent_llm()

    if llm is None:
        return None, None

    # create_react_agent builds the full graph:
    # 1. Binds tools to the LLM (so it knows what tools exist)
    # 2. Creates an "agent" node that calls the LLM
    # 3. Creates a "tools" node that executes tool calls
    # 4. Adds edges: agent → tools (if tool call) or agent → END (if no tool call)
    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=AGENT_SYSTEM_PROMPT,
    )

    return agent, llm


def run_agent(query: str) -> dict:
    """
    Run the research agent on a query and return the result.

    This is the main entry point for agent tasks. It:
    1. Creates the agent graph
    2. Invokes it with the user's query
    3. Collects the response and tool usage info
    4. Returns a structured result

    Args:
        query: The user's research question or task.

    Returns:
        A dict with:
            - answer: The agent's final answer
            - steps: List of steps taken (tool calls + results)
            - error: Error message if something went wrong (None otherwise)
    """
    agent, llm = create_research_agent()

    if agent is None:
        return {
            "answer": (
                "No LLM provider available for agents. "
                "Please configure a Groq or OpenRouter API key in your .env file."
            ),
            "steps": [],
            "error": "no_provider",
        }

    try:
        # Invoke the agent graph.
        # The input is a dict with a "messages" key containing the user's query.
        # LangGraph will:
        #   1. Send the query to the LLM
        #   2. If the LLM returns tool calls, execute them
        #   3. Send tool results back to the LLM
        #   4. Repeat until the LLM gives a final text response
        result = agent.invoke(
            {"messages": [("user", query)]},
        )

        # Extract the messages from the result
        messages = result.get("messages", [])

        # Build step-by-step log of what the agent did
        steps = []
        for msg in messages:
            # Tool calls from the agent
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({
                        "type": "tool_call",
                        "tool": tc["name"],
                        "input": str(tc["args"]),
                    })

            # Tool results
            if hasattr(msg, "type") and msg.type == "tool":
                steps.append({
                    "type": "tool_result",
                    "tool": getattr(msg, "name", "unknown"),
                    "output": msg.content[:500],  # Truncate long results
                })

        # The final message is the agent's answer
        final_answer = messages[-1].content if messages else "No response generated."

        return {
            "answer": final_answer,
            "steps": steps,
            "error": None,
        }

    except Exception as e:
        return {
            "answer": f"Agent encountered an error: {str(e)}",
            "steps": [],
            "error": str(e),
        }


def format_agent_steps(steps: list[dict]) -> str:
    """
    Format agent steps as a readable markdown log.

    Shows what the agent did step-by-step, so users can see
    the reasoning process. This builds trust and helps debug.

    Args:
        steps: List of step dicts from run_agent().

    Returns:
        Markdown-formatted string showing the agent's process.
    """
    if not steps:
        return ""

    lines = ["### Agent Process\n"]

    step_num = 0
    for step in steps:
        if step["type"] == "tool_call":
            step_num += 1
            lines.append(
                f"**Step {step_num}:** Used `{step['tool']}`\n"
                f"- Input: `{step['input'][:100]}`"
            )
        elif step["type"] == "tool_result":
            # Show a preview of the result
            preview = step["output"][:200].replace("\n", " ")
            lines.append(f"- Result: {preview}...")

    return "\n\n".join(lines)
