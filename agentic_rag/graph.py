from typing import Optional, List

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.documents import Document

from .nodes import (
    make_generate_query_or_respond,
    make_generate_answer,
    make_grade_documents,
    make_rewrite_question,
)


class CustomState(MessagesState):
    artifacts: List[Document]


def build_graph(
    response_model, grader_model, retriever_tool, workflow_name: Optional[str] = None
):
    """
    Assemble the LangGraph workflow using functional nodes.
    """
    workflow = StateGraph(CustomState)

    # Create node callables
    generate_query_or_respond = make_generate_query_or_respond(
        response_model, retriever_tool, CustomState
    )
    rewrite_question = make_rewrite_question(response_model, CustomState)
    generate_answer = make_generate_answer(response_model, CustomState)
    grade_documents = make_grade_documents(grader_model, CustomState)

    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve_documents", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)
    workflow.add_node(grade_documents)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve_documents",
            END: END,
        },
    )

    workflow.add_conditional_edges("retrieve_documents", grade_documents)

    workflow.add_edge("grade_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    return graph
