"""LangGraph 装配：clarifier → (条件) → fan-out → synthesizer。"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.clarifier import clarifier_node
from agents.diviner import diviner_node
from agents.historian import historian_node
from agents.modernist import modernist_node
from agents.state import GraphState
from agents.synthesizer import synthesizer_node


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
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "clarifier")
    g.add_conditional_edges(
        "clarifier",
        _route_after_clarify,
        ["historian", "diviner", "modernist", END],
    )
    # 三个子 agent 完成后自动 fan-in 到 synthesizer
    g.add_edge("historian", "synthesizer")
    g.add_edge("diviner", "synthesizer")
    g.add_edge("modernist", "synthesizer")
    g.add_edge("synthesizer", END)

    return g.compile()


GRAPH = build_graph()


if __name__ == "__main__":
    # 端到端手测：模拟一轮已澄清的状态直接跑
    import asyncio

    async def main():
        state: GraphState = {
            "query": "我应不应该从国企辞职去创业做 AI？",
            "chat_history": [
                {"role": "user", "content": "30岁男，已婚未育，有80万存款，AI行业有3年经验"},
            ],
        }
        async for event in GRAPH.astream(state, stream_mode="updates"):
            for node, payload in event.items():
                print(f"\n=== [{node}] ===")
                for k, v in payload.items():
                    snippet = (v or "")[:300] if isinstance(v, str) else v
                    print(f"  {k}: {snippet}")

    asyncio.run(main())
