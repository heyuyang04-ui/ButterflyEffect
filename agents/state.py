"""LangGraph 共享 State 定义。"""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    query: str
    chat_history: Annotated[list[dict], add]
    pending_question: str
    clarified_brief: str
    history_view: str
    divination_view: str
    modern_view: str
    # 用户对三方分析的态度：认可 | 迷茫 | 否认
    history_reaction: str
    modern_reaction: str
    divination_reaction: str
    final_report: str
