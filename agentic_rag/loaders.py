from typing import Iterable, List
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentFetcher:
    """
    Load documents from local file paths by extension.
    Currently supports: .pdf (PyPDFLoader), .txt/.md/.log/.csv/.json (TextLoader as plaintext).
    """

    def fetch_from_paths(self, paths: Iterable[str]) -> List[Document]:
        all_docs: List[Document] = []
        for p in paths:
            path = Path(p)
            if not path.exists() or not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                docs = PyPDFLoader(str(path)).load()
            elif suffix in {".txt", ".md", ".log", ".csv", ".json"}:
                docs = TextLoader(str(path), encoding="utf-8").load()
            else:
                # Fallback: treat as utf-8 text
                try:
                    docs = TextLoader(str(path), encoding="utf-8").load()
                except Exception:
                    docs = []
            all_docs.extend(docs)
        return all_docs


class TextPreprocessor:
    """
    Split documents into smaller overlapping chunks using a token-aware splitter.
    Matches tutorial defaults: chunk_size=100, chunk_overlap=50.
    """

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 50) -> None:
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        return self._splitter.split_documents(documents)
