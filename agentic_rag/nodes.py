from typing import Dict, Any, Literal, List

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.documents import Document


def _get_last_human_content(messages: List[Any]) -> str:
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            return getattr(m, "content", "")
    # Fallback to first message content
    return getattr(messages[0], "content", "")


def make_generate_query_or_respond(response_model, retriever_tool, CustomState):
    def generate_query_or_respond(state: CustomState) -> Dict[str, Any]:

        llm = response_model.bind_tools([retriever_tool])
        system_prompt = """You are an Assistant which have a conversation between patinet and Doctor. Your Task is to answer the user's question.

        If user is Greeting just greet them back. 
        If they try to talk about other things stop them from doing it.
        if they ask anything about the patient and doctor conversation use the tool to get the result and then respond to them.
        """
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


_GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


def make_grade_documents(grader_model, CustomState):
    def grade_documents(
        state: CustomState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        question = _get_last_human_content(state["messages"])
        context = state["messages"][-1].content
        prompt = _GRADE_PROMPT.format(question=question, context=context)
        response = grader_model.with_structured_output(_GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        score = response.binary_score
        return "generate_answer" if score == "yes" else "rewrite_question"

    return grade_documents


_REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def make_rewrite_question(response_model, CustomState):
    def rewrite_question(state: CustomState) -> Dict[str, Any]:
        messages = state["messages"]
        question = _get_last_human_content(messages)
        prompt = _REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"type": "human", "content": response.content}]}

    return rewrite_question


_GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


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
        prompt = _GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response], "artifacts": documents}

    return generate_answer
