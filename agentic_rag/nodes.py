from typing import Dict, Any, Literal, List

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from .prompt import (
    GENERATE_QUERY_OR_RESPOND_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    GENERATE_PROMPT,
)


def _get_last_human_content(messages: List[Any]) -> str:
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            return getattr(m, "content", "")
    # Fallback to first message content
    return getattr(messages[0], "content", "")


def _get_messages_till_last_human(messages: List[Any]) -> List[Any]:
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            return messages[: messages.index(m) + 1]
    return messages


def make_generate_query_or_respond(response_model, retriever_tool, CustomState):
    def generate_query_or_respond(state: CustomState) -> Dict[str, Any]:

        llm = response_model.bind_tools([retriever_tool])
        system_prompt = GENERATE_QUERY_OR_RESPOND_PROMPT
        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
        response = llm.invoke(messages)
        return {"messages": [response], "artifacts": []}

    return generate_query_or_respond


class _GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def make_grade_documents(grader_model, CustomState):
    def grade_documents(
        state: CustomState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        question = _get_messages_till_last_human(state["messages"])
        context = state["messages"][-1].content

        prompt = GRADE_PROMPT.format(context=context)
        response = grader_model.with_structured_output(_GradeDocuments).invoke(
            question + [{"role": "ai", "content": prompt}]
        )
        score = response.binary_score
        return "generate_answer" if score == "yes" else "rewrite_question"

    return grade_documents


def make_rewrite_question(response_model, CustomState):
    def rewrite_question(state: CustomState) -> Dict[str, Any]:
        messages = state["messages"]
        question = _get_last_human_content(messages)
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])

        context = response.content + "\n\n" + "Never try to answer of your own."
        return {"messages": [{"type": "ai", "content": context}]}

    return rewrite_question


def format_context(documents: List[Document]) -> str:
    context = ""
    for doc in documents:
        context += "----------------------------------------\n"
        context += "Source: " + doc.metadata["source"]
        context += "Page: " + str(doc.metadata["page"])
        context += "Content: " + doc.page_content
        context += "\n\n"
    return context


def make_generate_answer(response_model, CustomState):
    def generate_answer(state: CustomState) -> Dict[str, Any]:
        question = _get_last_human_content(state["messages"])
        documents = state["messages"][-1].artifact
        context = format_context(documents)
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response], "artifacts": documents}

    return generate_answer
