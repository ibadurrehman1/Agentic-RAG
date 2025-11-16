# Agentic RAG - Medical Transcript Analysis System

An intelligent Retrieval-Augmented Generation (RAG) system built with LangGraph and Chainlit, designed for analyzing medical transcripts and patient-doctor conversations. The system uses an agentic workflow to intelligently retrieve relevant information and generate context-aware responses.

## 🚀 Features

- **Interactive Web Interface**: Beautiful Chainlit-based UI for document upload and Q&A
- **Multi-Format Document Support**: Upload PDF, TXT, MD, CSV, and JSON files
- **Intelligent Document Retrieval**: Semantic search using OpenAI embeddings
- **Agentic Workflow**: LangGraph-powered decision-making system that:
  - Decides when to retrieve documents vs. answering directly
  - Grades document relevance automatically
  - Rewrites questions when retrieval is insufficient
  - Generates concise, context-aware answers
- **Source Attribution**: Automatically cites document sources and page numbers
- **Streaming Responses**: Real-time token streaming for better UX
- **Centralized Prompts**: All prompts managed in a single file for easy customization

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for package dependencies

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ibadurrehman1/Agentic-RAG.git
   cd Agentic-RAG
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=sk-your-api-key-here
   DEFAULT_CHAT_MODEL=gpt-4o  # Optional, defaults to gpt-4.1
   DEFAULT_TEMPERATURE=0      # Optional, defaults to 0
   ```
   
   Or set the API key directly:
   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="sk-..."
   
   # Linux/macOS
   export OPENAI_API_KEY="sk-..."
   ```

## 🎯 Usage

### Starting the Application

Run the Chainlit application:

```bash
python main.py
```

Or directly with Chainlit:

```bash
chainlit run main.py
```

The application will open in your browser at `http://localhost:8000`.

### Using the Interface

1. **Upload Documents**: When the app starts, you'll be prompted to upload medical transcript files (.pdf, .txt, .md, .csv, .json)
2. **Wait for Indexing**: The system will process and index your documents
3. **Ask Questions**: Once indexed, ask questions about the uploaded documents
4. **View Sources**: The system will automatically show source citations with page numbers

### Example Questions

- "What medications were discussed?"
- "What was the patient's diagnosis?"
- "Summarize the treatment plan"
- "What were the doctor's recommendations?"

## 🏗️ Architecture

### Project Structure

```
agentic_rag/
  ├── __init__.py
  ├── config.py          # Configuration and environment variables
  ├── models.py          # ModelFactory for LLM and embeddings
  ├── loaders.py         # DocumentFetcher and TextPreprocessor
  ├── vectorstore.py     # VectorStoreBuilder and retriever tools
  ├── nodes.py           # Graph node implementations
  ├── prompt.py          # Centralized prompt templates
  └── graph.py           # LangGraph workflow assembly
main.py                   # Chainlit application entry point
requirements.txt          # Python dependencies
README.md                 # This file
```

### Workflow Graph

The system uses a LangGraph workflow with the following nodes:

1. **`generate_query_or_respond`**: Decides whether to use the retriever tool or answer directly
2. **`retrieve_documents`**: Retrieves relevant document chunks using semantic search
3. **`grade_documents`**: Evaluates if retrieved documents are relevant to the question
4. **`rewrite_question`**: Improves the question if retrieval was insufficient
5. **`generate_answer`**: Generates the final answer using retrieved context

```
START → generate_query_or_respond
         ├─→ [No tool needed] → END
         └─→ [Tool needed] → retrieve_documents
                            → grade_documents
                            ├─→ [Relevant] → generate_answer → END
                            └─→ [Not relevant] → rewrite_question → generate_query_or_respond (loop)
```

## 🔧 Configuration

### Model Settings

Edit `agentic_rag/config.py` or set environment variables:

- `DEFAULT_CHAT_MODEL`: LLM model to use (default: "gpt-4.1")
- `DEFAULT_TEMPERATURE`: Model temperature (default: 0)

### Prompt Customization

All prompts are centralized in `agentic_rag/prompt.py`:

- `GENERATE_QUERY_OR_RESPOND_PROMPT`: System prompt for the main agent
- `GRADE_PROMPT`: Document relevance grading prompt
- `REWRITE_PROMPT`: Question rewriting prompt
- `GENERATE_PROMPT`: Final answer generation prompt

### Chunking Settings

In `main.py`, adjust text preprocessing:

```python
preprocessor = TextPreprocessor(chunk_size=100, chunk_overlap=50)
```

- `chunk_size`: Maximum characters per chunk
- `chunk_overlap`: Overlap between chunks for context preservation

## 📚 Key Components

### ModelFactory (`models.py`)

Manages LLM and embedding model initialization:
- `response_model()`: Primary chat model for responses
- `grader_model()`: Model for document grading (temperature=0)
- `embeddings()`: OpenAI embeddings model

### DocumentFetcher (`loaders.py`)

Handles document loading from various file formats:
- Supports PDF, TXT, MD, CSV, JSON
- Extracts text and metadata

### TextPreprocessor (`loaders.py`)

Splits documents into chunks:
- Recursive character text splitting
- Configurable chunk size and overlap
- Preserves document metadata

### VectorStoreBuilder (`vectorstore.py`)

Creates and manages the vector store:
- In-memory vector store using LangChain
- OpenAI embeddings integration
- Retriever tool creation for agent use

### Graph Nodes (`nodes.py`)

Implements the workflow nodes:
- `make_generate_query_or_respond()`: Decision node
- `make_grade_documents()`: Relevance grading
- `make_rewrite_question()`: Question improvement
- `make_generate_answer()`: Answer generation

## 🔍 How It Works

1. **Document Upload**: User uploads medical transcript files
2. **Processing**: Documents are split into chunks and embedded
3. **Indexing**: Chunks are stored in an in-memory vector store
4. **Question Processing**: When a question is asked:
   - The agent decides if retrieval is needed
   - If yes, retrieves relevant chunks
   - Grades relevance automatically
   - Rewrites question if needed (loops back)
   - Generates answer with source citations
5. **Response**: Streams the answer with source references

## 🎨 Customization

### Changing the Domain

To adapt for non-medical use cases:

1. Update `GENERATE_QUERY_OR_RESPOND_PROMPT` in `prompt.py`
2. Modify the file upload message in `main.py`
3. Adjust other prompts as needed

### Adding Persistent Storage

Replace the in-memory vector store in `vectorstore.py` with:
- ChromaDB
- Pinecone
- Weaviate
- Any LangChain-compatible vector store

### Extending the Graph

Add new nodes in `nodes.py` and wire them in `graph.py`:
- Additional retrieval strategies
- Multi-step reasoning
- External API calls

## 📝 Dependencies

- `langgraph`: Graph-based agent workflows
- `langchain[openai]`: Core LangChain functionality
- `langchain-community`: Community integrations
- `langchain-text-splitters`: Text chunking utilities
- `pypdf`: PDF document parsing
- `chainlit`: Interactive web UI
- `python-dotenv`: Environment variable management

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Chainlit](https://github.com/Chainlit/chainlit)
- [OpenAI](https://openai.com/)

---