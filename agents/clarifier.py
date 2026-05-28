"""主 agent —— 多轮反问澄清。"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm_factory import get_llm
from .state import GraphState

SYSTEM_PROMPT = (
    "你是一位老练的「决策访谈员」，正在帮用户把一个模糊的问题打磨成可分析的决策。\n"
    "你不给建议、不评价、不预测——只负责把背景挖清楚。\n"
    "\n"
    "你的工作流：\n"
    "1. 看用户的原始问题和已有对话历史，判断是否已经掌握以下关键背景：\n"
    "   1) 决策主体：谁要做？年龄、身份、家庭/财务/健康关键约束\n"
    "   2) 决策内容：具体在选什么？有哪些备选项？\n"
    "   3) 时空背景：什么时候要做？地点、行业、所处阶段\n"
    "   4) 真实动机：为什么纠结？最担心失去什么？最想得到什么\n"
    "   5) 退路：如果失败，最坏会怎样？能承受吗\n"
    "2. 缺哪一项就问哪一项——一次只问一个最关键的问题，不要连珠炮\n"
    "3. 当 5 项都基本清楚后，或已反问达到 3 次，立即停止提问，输出结构化的澄清简报\n"
    "\n"
    "【重要限制】最多只能反问 3 次。若已反问 3 次，无论信息是否完整，必须立即将 need_more 设为 false，\n"
    "用现有信息整合一份澄清简报，不得继续提问。\n"
    "\n"
    "【输出协议】严格输出 JSON，不要任何额外文字、不要 markdown 代码块包裹：\n"
    '{"need_more": true | false, "question": "若 need_more=true，这里写下一个反问；否则空字符串", '
    '"brief": "若 need_more=false，这里写一段 200-400 字的中文澄清简报；否则空字符串"}\n'
    "\n"
    "【brief 模板】\n"
    "决策主体：……\n"
    "决策内容：……（A 方案 vs B 方案 vs 维持现状）\n"
    "时空背景：……\n"
    "真实动机：……\n"
    "可承受底线：……"
)


def _extract_json(text: str) -> dict:
    """从模型输出里抠出 JSON，容忍 ```json 包裹和前后噪声。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1:
            text = text[first : last + 1]
    return json.loads(text)


MAX_CLARIFY_ROUNDS = 3


def clarifier_node(state: GraphState) -> dict:
    llm = get_llm("clarifier")
    history = state.get("chat_history", [])

    # 历史里 assistant 发言条数 = 已反问轮次
    rounds_done = sum(1 for t in history if t.get("role") == "assistant")
    force_finish = rounds_done >= MAX_CLARIFY_ROUNDS

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.append(HumanMessage(content=f"用户的原始问题：{state.get('query', '')}"))
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    if force_finish:
        messages.append(
            HumanMessage(
                content="[系统提示：已达到最大反问次数，请立即将 need_more 设为 false，整合现有信息输出澄清简报。]"
            )
        )

    raw = llm.invoke(messages).content

    try:
        data = _extract_json(raw)
        need_more = bool(data.get("need_more", False))
        question = (data.get("question") or "").strip()
        brief = (data.get("brief") or "").strip()
    except Exception:
        need_more = not force_finish
        question = raw.strip() if not force_finish else ""
        brief = raw.strip() if force_finish else ""

    # 强制完成：无论模型输出什么，不再继续提问
    if force_finish:
        need_more = False
        if not brief:
            brief = raw.strip()

    if need_more and question:
        return {"pending_question": question, "clarified_brief": ""}
    return {"pending_question": "", "clarified_brief": brief or raw}
