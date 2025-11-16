import uuid
import subprocess

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agentic_rag.models import ModelFactory
from agentic_rag.loaders import DocumentFetcher, TextPreprocessor
from agentic_rag.vectorstore import VectorStoreBuilder
from agentic_rag.graph import build_graph

load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    # Initialize session
    cl.user_session.set("session_id", str(uuid.uuid4()))

    uploaded_files = None

    # Ask user for documents
    while uploaded_files is None:
        uploaded_files = await cl.AskFileMessage(
            content="Welcome to the Medical Transcript Analysis System. Please upload patient-doctor conversation transcripts or medical documents (.pdf, .txt, .md, .csv, .json).\n\nAfter uploading, you can ask questions about diagnoses, treatments, medications, or any other medical information from the documents.",
            accept={
                "text/plain": [".txt", ".md", ".csv", ".json"],
                "application/pdf": [".pdf"],
            },
            max_size_mb=50,
            timeout=300,
        ).send()

    file_paths = [f.path for f in uploaded_files]

    status_msg = cl.Message(
        content="Processing medical documents and building knowledge base..."
    )
    await status_msg.send()

    # ---- Document Loading & Chunking ----
    fetcher = DocumentFetcher()
    raw_documents = fetcher.fetch_from_paths(file_paths)

    preprocessor = TextPreprocessor(chunk_size=800, chunk_overlap=200)
    chunks = preprocessor.split(raw_documents)

    # ---- Embeddings & VectorStore ----
    model_factory = ModelFactory()
    embeddings = model_factory.embeddings()

    vector_store = VectorStoreBuilder(embeddings).index(chunks)
    retriever = vector_store.retriever()

    retriever_tool = vector_store.retriever_tool(
        name="retrieve_documents",
        description="Search and return information from the uploaded documents.",
    )

    # ---- Build the Agent Graph ----
    graph = build_graph(
        response_model=model_factory.response_model(),
        grader_model=model_factory.grader_model(),
        retriever_tool=retriever_tool,
    )

    graph.get_graph().draw_mermaid_png()

    # Save to session
    cl.user_session.set("graph", graph)
    cl.user_session.set("retriever", retriever)

    status_msg.content = "Medical documents have been successfully indexed. You can now ask questions about patient records, diagnoses, treatments, medications, or any other medical information from the uploaded documents."
    await status_msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    session_id = cl.user_session.get("session_id")

    if graph is None:
        await cl.Message(
            content="Medical analysis system not initialized. Please restart the session and upload medical transcript files."
        ).send()
        return

    response_msg = cl.Message(content="")

    response_msg.content = "Thinking..."

    await response_msg.send()

    collected_artifacts = []

    is_thinking = True

    # Stream responses from the graph
    for stream_type, payload in graph.stream(
        {"messages": [HumanMessage(content=message.content)], "artifacts": []},
        config={"configurable": {"thread_id": session_id}},
        stream_mode=["messages", "updates"],
    ):

        if stream_type == "messages":
            chunk, meta = payload
            if chunk.content and meta["langgraph_node"] in [
                "generate_answer",
                "generate_query_or_respond",
            ]:
                if is_thinking:
                    is_thinking = False
                    response_msg.content = ""
                    await response_msg.update()
                await response_msg.stream_token(chunk.content)

        elif stream_type == "updates":
            if "generate_answer" in payload:
                artifacts = payload["generate_answer"].get("artifacts", [])
                collected_artifacts.extend(artifacts)

    # ---- Attach Sources (if any) ----
    source_elements = []

    if collected_artifacts:
        for doc in collected_artifacts:
            filename = doc.metadata["source"].split("\\")[-1]
            element_name = f"{filename} (Page {doc.metadata['page']+1})"

            source_elements.append(
                cl.Text(
                    content=doc.page_content,
                    name=element_name,
                    display="side",
                )
            )

        # Build "Sources" section
        source_text = "\n\nSources:\n - " + "\n - ".join(
            [elem.name for elem in source_elements]
        )
        await response_msg.stream_token(source_text)

    if source_elements:
        response_msg.elements = source_elements

    await response_msg.update()


if __name__ == "__main__":
    subprocess.run(["chainlit", "run", "main.py"])
