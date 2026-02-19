# Nexus AI

**A full-stack AI research & productivity assistant** built with Python, demonstrating modern AI engineering patterns: multi-LLM orchestration, RAG, agentic workflows, tool use, and observability.

## Architecture

```
nexus/
├── app.py                  # Main entry point — assembles all components
├── config.py               # Centralized settings (Pydantic + .env)
├── requirements.txt        # All dependencies
│
├── llm_providers/          # Phase 1: Multi-LLM Chatbot
│   ├── base.py             # Abstract base class (Strategy pattern)
│   ├── openrouter.py       # OpenRouter provider (free models)
│   ├── groq_provider.py    # Groq provider (fast inference)
│   ├── gemini_provider.py  # Google Gemini provider
│   ├── ollama_provider.py  # Ollama local models
│   ├── router.py           # ProviderManager — unified interface
│   └── system_prompts.py   # Prompt engineering templates
│
├── rag/                    # Phase 2: Knowledge Base (RAG)
│   ├── ingestion.py        # Document parsing & chunking
│   ├── youtube.py          # YouTube transcript ingestion
│   ├── embeddings.py       # HuggingFace local embeddings
│   ├── vector_store.py     # LanceDB vector storage & search
│   └── query_engine.py     # RAG retrieval + citation formatting
│
├── agents/                 # Phase 3-4: AI Agents & Tools
│   ├── tools.py            # Tool registry (8 tools)
│   ├── tool_library.py     # Calculator, code executor, datetime, file writer
│   └── research_agent.py   # LangGraph ReAct agent
│
├── monitoring/             # Phase 5: Observability
│   └── tracker.py          # Usage logging, cost estimation, analytics
│
└── ui/                     # Gradio UI (all phases)
    ├── chat_tab.py         # Chat interface with RAG toggle
    ├── knowledge_tab.py    # Document upload & management
    ├── agents_tab.py       # Research agent interface
    ├── tools_tab.py        # Tool directory & tester
    └── dashboard_tab.py    # Usage analytics dashboard
```

## Features

### Multi-LLM Chat (Phase 1)
- **4 providers**: OpenRouter, Groq, Google Gemini, Ollama (local)
- **12+ free models**: Llama 3.1/3.3, Gemma 2, Mistral, Qwen, Gemini Flash/Pro
- Adjustable temperature, max tokens, and system prompt
- Provider auto-detection based on configured API keys

### Knowledge Base — RAG (Phase 2)
- **Document ingestion**: PDF, TXT, MD files and web URLs
- **YouTube transcripts**: Pull and index any YouTube video
- **Local embeddings**: HuggingFace all-MiniLM-L6-v2 (no API cost)
- **LanceDB vector store**: Persistent local storage, no server needed
- **Source citations**: Every answer shows which documents were used

### AI Agents (Phase 3)
- **LangGraph ReAct workflow**: Think, act, observe loop
- **Web research**: DuckDuckGo search + web page scraping
- **Knowledge base integration**: Agents can search your documents
- **Step-by-step transparency**: See every tool call and result

### Tools (Phase 4)
- **Calculator**: Precise math (trig, logs, factorial)
- **Python executor**: Sandboxed code execution
- **Date/time**: Current time, date math, countdowns
- **File writer**: Save reports to disk
- **Interactive tester**: Try any tool directly from the UI

### Dashboard (Phase 5)
- **Token tracking**: Input/output tokens per call
- **Cost estimation**: Per-model pricing (free tier = $0)
- **Latency monitoring**: Response time measurement
- **Breakdowns**: By model, provider, and feature
- **Activity log**: Recent interactions with status

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url> && cd nexus
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your keys (all optional — configure what you have)
```

**Free API keys:**
- [Groq](https://console.groq.com) — Fast Llama 3.3 inference
- [OpenRouter](https://openrouter.ai) — Many free models
- [Google AI Studio](https://aistudio.google.com) — Gemini Flash/Pro

### 3. Run

```bash
python app.py
# Opens at http://localhost:7860
```

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| UI | Gradio | Rapid prototyping, built-in chat component |
| LLM | OpenAI SDK | Universal — works with OpenRouter, Groq, Ollama |
| RAG | LlamaIndex + LanceDB | Industry-standard chunking + local vector DB |
| Agents | LangGraph | Graph-based agent workflows, ReAct pattern |
| Embeddings | sentence-transformers | Local, free, no API needed |
| Search | DuckDuckGo | Free web search, no API key |
| Config | Pydantic Settings | Type-safe settings from .env |
| Monitoring | Custom JSONL logger | Simple, local, no external service |

## Key Concepts Demonstrated

- **Strategy Pattern**: Swappable LLM providers behind a common interface
- **RAG Pipeline**: Ingest, embed, store, retrieve, cite
- **ReAct Agents**: Reasoning + Acting in a loop with tool calling
- **Prompt Engineering**: System prompts for chat, RAG, and agents
- **Observability**: Token counting, cost tracking, latency monitoring
- **Security**: Sandboxed code execution, blocked modules, path traversal prevention
