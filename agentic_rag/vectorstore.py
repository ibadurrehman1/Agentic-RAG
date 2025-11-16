from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.tools.retriever import create_retriever_tool


class VectorStoreBuilder:
    """
    Builder for an in-memory vector store and a retriever tool for the agent.
    """

    def __init__(self, embeddings) -> None:
        self._embeddings = embeddings
        self._vectorstore = None
        self._retriever = None

    def index(self, doc_splits: List[Document]) -> "VectorStoreBuilder":
        self._vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits,
            embedding=self._embeddings,
        )
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 10})
        return self

    def retriever(self):
        return self._retriever

    def retriever_tool(
        self,
        name: str = "retrieve_documents",
        description: str = "Search and return information from the documents.",
    ):
        if self._retriever is None:
            raise RuntimeError("Call index(...) before creating a retriever tool.")
        return create_retriever_tool(
            self._retriever, name, description, response_format="content_and_artifact"
        )
