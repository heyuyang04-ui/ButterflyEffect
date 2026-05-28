"""LangGraph 共享 State 定义。"""
from __future__ import annotations

from typing import Annotated, TypedDict
from operator import add


class GraphState(TypedDict, total=False):
    query: str
    chat_history: Annotated[list[dict], add]
    pending_question: str
    clarified_brief: str
    history_view: str
    divination_view: str
    modern_view: str
    final_report: str
