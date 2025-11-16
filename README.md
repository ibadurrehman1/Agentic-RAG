# Agentic RAG (LangGraph) - Multi-file, Well-Documented Implementation

This project implements a custom retrieval agent (agentic RAG) using LangGraph and LangChain, structured with clear separation of concerns and documented code. It closely follows the tutorial example while organizing code into reusable modules and applying simple design patterns (Factory, Builder, Facade).

## Features
- Web document loading and preprocessing (split into chunks).
- In-memory vector store indexing with OpenAI embeddings.
- Retriever tool exposed to the agent.
- Agent nodes:
  - Generate query or respond (decides to use the retriever tool or answer directly)
  - Grade retrieved documents (conditional edge)
  - Rewrite question (if retrieval was irrelevant)
  - Generate final answer (if retrieval was relevant)
- Graph assembly using LangGraph.
- CLI `main.py` to pass URLs and a question and get the response.

## Requirements
See `requirements.txt`.

## Installation
```bash
pip install -r requirements.txt
```

You need an OpenAI API key:
```bash
$env:OPENAI_API_KEY="sk-..."   # PowerShell (Windows)
# or
export OPENAI_API_KEY="sk-..." # bash/zsh (Linux/macOS)
```
If not set, the app will securely prompt for it.

## Usage
```bash
python main.py --files ./docs/medical_transcript.pdf ./notes/summary.md --question "What does the document say about medication schedule?" --stream --visualize graph.png
```

Arguments:
- `--files`: One or more local file paths to index (space-separated).
- `--question`: The user question to answer.
- `--stream`: (optional) Stream graph node updates.
- `--visualize <path>`: (optional) Save a PNG visualization of the compiled graph.

## Project Structure
```
agentic_rag/
  __init__.py
  config.py            # API key setup, defaults, shared settings
  models.py            # ModelFactory for chat models and embeddings
  loaders.py           # WebBaseLoader fetch + RecursiveCharacterTextSplitter
  vectorstore.py       # InMemoryVectorStore builder and retriever tool factory
  graph.py             # LangGraph assembly (nodes, edges, compilation)
  pipeline.py          # Facade to run the end-to-end flow
  nodes/
    __init__.py
    decision.py        # generate_query_or_respond
    grading.py         # grade_documents conditional edge
    rewrite.py         # rewrite_question
    answer.py          # generate_answer
main.py                # CLI entry point
requirements.txt
README.md
```

## Notes
- The implementation uses `init_chat_model("gpt-4o")` for parity with the tutorial.
- Vector store is in-memory for simplicity; swap out for a persistent store as needed.
- The grader uses a structured output model (`pydantic`) and routes accordingly.

# Agentic-RAG
Agentic RAG with LangGraph v1 implementation and interactive Chainlit interface for context-aware AI responses.
