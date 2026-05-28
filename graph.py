"""LangGraph 装配：clarifier → (条件) → fan-out 三方分析 → END。

synthesizer 已从图中解耦，由 UI 层在用户完成表态后直接调用。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.clarifier import clarifier_node
from agents.diviner import diviner_node
from agents.historian import historian_node
from agents.modernist import modernist_node
from agents.state import GraphState


def _route_after_clarify(state: GraphState) -> list[str] | str:
    """澄清完成 → fan-out 到三个并行子 agent；未澄清 → END。"""
    if state.get("pending_question"):
        return END
    return ["historian", "diviner", "modernist"]


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("clarifier", clarifier_node)
    g.add_node("historian", historian_node)
    g.add_node("diviner", diviner_node)
    g.add_node("modernist", modernist_node)

    g.add_edge(START, "clarifier")
    g.add_conditional_edges(
        "clarifier",
        _route_after_clarify,
        ["historian", "diviner", "modernist", END],
    )
    g.add_edge("historian", END)
    g.add_edge("diviner", END)
    g.add_edge("modernist", END)

    return g.compile()


GRAPH = build_graph()
