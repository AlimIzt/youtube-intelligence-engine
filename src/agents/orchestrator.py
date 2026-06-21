"""LangGraph agent that routes user queries to the right tool.

Mirrors the Lab 8 ToolCalling pattern: a chatbot node binds the tools, a
conditional edge sends to the ToolNode when a tool call is emitted, then loops
back so the LLM can answer using the tool output (ReAct-style).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import mlflow
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from config import settings
from src.agents.tools import ALL_TOOLS

SYSTEM = (
    "You are the YouTube Intelligence agent. You have tools to answer questions "
    "(comment_qa), summarize opinions (comment_summary), report sentiment "
    "(sentiment_insight), and list topics (topic_insight). Pick the single best "
    "tool for the user's request, then answer using its result. If no tool fits, "
    "answer directly."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@lru_cache(maxsize=1)
def build_agent():
    llm = ChatOllama(model=settings.llm_model, temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    @mlflow.trace(span_type="AGENT")
    def chatbot(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges(
        "chatbot", tools_condition, {"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "chatbot")  # loop back to answer (ReAct)
    return builder.compile()


def ask(query: str) -> str:
    agent = build_agent()
    result = agent.invoke(
        {"messages": [("system", SYSTEM), ("user", query)]}
    )
    return result["messages"][-1].content


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    mlflow.langchain.autolog()
    agent = build_agent()

    print("\nYouTube Intelligence Agent (type 'exit' to quit)\n")
    with mlflow.start_run(run_name="agent_session"):
        while True:
            q = input("User: ")
            if q.lower() in {"exit", "quit"}:
                break
            out = agent.invoke({"messages": [("system", SYSTEM), ("user", q)]})
            print(f"\nAssistant: {out['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
