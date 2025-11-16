from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

from .config import DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE


class ModelFactory:
    """
    Factory for constructing chat models and embeddings.

    - Chat model is initialized via LangChain's init_chat_model to support provider-agnostic creation.
    - Embeddings use OpenAIEmbeddings as in the tutorial.
    """

    def __init__(
        self,
        chat_model_name: str = DEFAULT_CHAT_MODEL,
        temperature: int = DEFAULT_TEMPERATURE,
    ) -> None:
        self._chat_model_name = chat_model_name
        self._temperature = temperature
        self._response_model = None
        self._grader_model = None
        self._embeddings = None

    def response_model(self):
        """
        Get or create the primary chat model used for responses and general prompting.
        """
        if self._response_model is None:
            self._response_model = init_chat_model(
                self._chat_model_name, temperature=self._temperature
            )
        return self._response_model

    def grader_model(self):
        """
        Get or create the grading model (can be the same as response model).
        """
        if self._grader_model is None:
            self._grader_model = init_chat_model(self._chat_model_name, temperature=0)
        return self._grader_model

    def embeddings(self, model: Optional[str] = None):
        """
        Get or create embeddings model. Uses OpenAIEmbeddings by default.
        """
        if self._embeddings is None:
            self._embeddings = (
                OpenAIEmbeddings(model=model) if model else OpenAIEmbeddings()
            )
        return self._embeddings
