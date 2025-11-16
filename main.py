import uuid
from typing import List

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_rag.models import ModelFactory
from agentic_rag.loaders import DocumentFetcher, TextPreprocessor
from agentic_rag.vectorstore import VectorStoreBuilder
from agentic_rag.graph import build_graph


load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("session_id", str(uuid.uuid4()))
    files = None

    # # Wait for the user to upload files
    while files is None:
        files = await cl.AskFileMessage(
            content="Upload medical transcript files (.pdf, .txt, .md, .csv, .json). Then ask your question.",
            accept={
                "text/plain": [".txt", ".md", ".csv", ".json"],
                "application/pdf": [".pdf"],
            },
            max_size_mb=50,
            timeout=300,
        ).send()

    all_paths = [f.path for f in files]
    msg = cl.Message(content="Indexing documents...")
    await msg.send()

    # Build documents and retriever
    fetcher = DocumentFetcher()
    raw_docs = fetcher.fetch_from_paths(all_paths)
    pre = TextPreprocessor(chunk_size=100, chunk_overlap=50)
    doc_splits = pre.split(raw_docs)

    factory = ModelFactory()
    embeddings = factory.embeddings()
    vector_db = VectorStoreBuilder(embeddings).index(doc_splits)
    retriever = vector_db.retriever()
    retriever_tool = vector_db.retriever_tool(
        name="retrieve_documents",
        description="Search and return information from the documents.",
    )

    graph = build_graph(
        response_model=factory.response_model(),
        grader_model=factory.grader_model(),
        retriever_tool=retriever_tool,
    )

    graph.get_graph().draw_mermaid_png()

    cl.user_session.set("graph", graph)
    cl.user_session.set("retriever", retriever)

    msg.content = "Documents indexed. You can now ask questions."
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    retriever = cl.user_session.get("retriever")
    session_id = cl.user_session.get("session_id")

    if graph is None or retriever is None:
        await cl.Message(
            content="No graph or retriever initialized. Please restart and upload files."
        ).send()
        return

    # Prepare state and run graph

    msg = cl.Message(content="")
    artifacts = []

    for mode, message_data in graph.stream(
        {"messages": [HumanMessage(content=message.content)], "artifacts": []},
        config={"configurable": {"thread_id": session_id}},
        stream_mode=["messages", "updates"],
    ):

        if mode == "messages":
            chunk, metadata = message_data
            if chunk.content and metadata["langgraph_node"] in [
                "generate_answer",
                "generate_query_or_respond",
            ]:
                await msg.stream_token(chunk.content)

        elif mode == "updates":

            # check if any one key is generate_answer
            if any(key.startswith("generate_answer") for key in message_data.keys()):
                if message_data["generate_answer"]["artifacts"]:
                    artifacts.extend(message_data["generate_answer"]["artifacts"])

    print("ARTIFACTS: ", artifacts)

    text_elements = []

    # If there are source documents, attach them like the example
    if artifacts:
        for source_idx, source_doc in enumerate(artifacts):

            # Extract the filename only
            source_name = source_doc.metadata["source"].split("\\")[-1]

            # Create the side-display text element just like example
            text_elements.append(
                cl.Text(
                    content=source_doc.page_content,
                    name=f"{source_name} (Page {source_doc.metadata['page']})",
                    display="side",
                )
            )

        # Build the "Sources:" footer message exactly like example
        source_names = [el.name for el in text_elements]

        if source_names:
            source_list = "\n - ".join(source_names)
            source = "\n\n\n Sources:\n - " + source_list
            await msg.stream_token(source)

    if text_elements != []:
        msg.elements = text_elements

    await msg.update()


if __name__ == "__main__":
    import subprocess

    subprocess.run(["chainlit", "run", "main.py"])
