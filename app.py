"""Gradio Web UI —— 多视角决策辅助系统（三阶段交互）。

阶段1：澄清对话（聊天框）
阶段2：三栏分析展示（带实时状态指示） + 用户表态
阶段3：synthesizer 综合用户态度输出最终报告
"""
from __future__ import annotations

import gradio as gr

from agents.state import GraphState
from agents.synthesizer import synthesizer_node
from graph import GRAPH

WELCOME = """# 🪞 决策三镜
**输入你正在纠结的事**，主访谈员先反问几轮把背景挖清楚；
然后由 **史学家（过去）**、**战略咨询师（现在）**、**术数师（将来）** 并行作答；
你对每个视角表态后，主访谈员综合你的反应给出最终建议。"""

EXAMPLES = [
    "我应不应该从国企辞职去创业做 AI？",
    "下个月要不要把房子卖了换个城市发展？",
    "春节要不要回老家相亲？",
    "孩子小升初要不要拼鸡娃？",
]

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
footer { display: none !important; }

.welcome-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #fbf3ff 100%);
    border-radius: 16px; padding: 20px 28px; margin-bottom: 16px;
}

/* 澄清聊天框 */
#chat-box { box-shadow: none !important; }

/* 三栏面板通用 */
.panel-col {
    background: #fafafa; border-radius: 14px;
    padding: 18px; min-height: 340px;
    display: flex; flex-direction: column;
}
.panel-history { border-top: 4px solid #92400e; }
.panel-modern  { border-top: 4px solid #1e40af; }
.panel-diviner { border-top: 4px solid #6b21a8; }

.panel-content { flex: 1; font-size: 14px; line-height: 1.7; overflow-y: auto; }

/* 顾问状态标签 */
.status-badge {
    display: inline-block; font-size: 12px; font-weight: 600;
    border-radius: 12px; padding: 2px 10px; margin-bottom: 8px;
}
.status-thinking p { color: #b45309 !important; }
.status-done    p { color: #15803d !important; }
.status-error   p { color: #b91c1c !important; }
.status-idle    p { color: #6b7280 !important; }

/* Radio 横排胶囊 */
.reaction-radio .wrap { flex-direction: row !important; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.reaction-radio label {
    border: 1px solid #d1d5db; border-radius: 20px;
    padding: 4px 16px; cursor: pointer; font-size: 13px;
    transition: all .15s ease;
}
.reaction-radio label:has(input:checked) {
    background: #5b21b6; color: #fff; border-color: #5b21b6;
}
.reaction-radio .label-wrap { display: none !important; }

/* 整体分析按钮 */
#synth-btn { margin: 20px auto; display: block; min-width: 200px; font-size: 16px; }

/* 最终报告 */
#final-report {
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 14px; padding: 24px; margin-top: 8px;
    font-size: 14px; line-height: 1.8;
}
"""

# 状态标签文本常量
ST_IDLE     = "⬜ 等待中"
ST_THINKING = "⏳ 思考中…"
ST_DONE     = "✅ 已完成"
ST_ERROR    = "⚠️ 出错"


def _new_state() -> dict:
    return {
        "query": "",
        "chat_history": [],
        "clarified_brief": "",
        "history_view": "",
        "modern_view": "",
        "divination_view": "",
        "history_reaction": "",
        "modern_reaction": "",
        "divination_reaction": "",
        "final_report": "",
        "phase": "init",
    }


# ──────────────────────────────────────────────
# 工具：构造统一大小的 yield 元组（15 项）
# 顺序：chat, state, msg, send_btn,
#        view_history, view_modern, view_divination,
#        st_history, st_modern, st_divination,
#        clarify_col, panel_row, synth_btn, final_report,
#        reset_btn
# ──────────────────────────────────────────────

def _Y(chat, state,
       msg=None, send=None,
       vh=None, vm=None, vd=None,
       sh=None, sm=None, sd=None,
       clarify=None, panels=None, synth=None, report=None,
       reset=None):
    return (
        chat, state,
        gr.update() if msg   is None else msg,
        gr.update() if send  is None else send,
        gr.update() if vh    is None else vh,
        gr.update() if vm    is None else vm,
        gr.update() if vd    is None else vd,
        gr.update() if sh    is None else sh,
        gr.update() if sm    is None else sm,
        gr.update() if sd    is None else sd,
        gr.update() if clarify is None else clarify,
        gr.update() if panels  is None else panels,
        gr.update() if synth   is None else synth,
        gr.update() if report  is None else report,
        gr.update() if reset   is None else reset,
    )


# ──────────────────────────────────────────────
# 阶段1：澄清对话
# ──────────────────────────────────────────────

async def on_user_send(user_msg: str, chat_msgs: list, state: dict):
    user_msg = (user_msg or "").strip()

    if not user_msg:
        yield _Y(chat_msgs, state)
        return

    if state["phase"] == "done":
        chat_msgs = chat_msgs + [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": "本轮分析已完成。点 **🔄 新对话** 开始下一轮。"},
        ]
        yield _Y(chat_msgs, state, msg=gr.update(value=""))
        return

    if state["phase"] == "init":
        state["query"] = user_msg
        state["phase"] = "clarifying"
    else:
        state["chat_history"].append({"role": "user", "content": user_msg})

    chat_msgs = chat_msgs + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": "_正在思考…_"},
    ]
    yield _Y(chat_msgs, state,
             msg=gr.update(value="", interactive=False),
             send=gr.update(interactive=False))

    graph_input: GraphState = {
        "query": state["query"],
        "chat_history": list(state["chat_history"]),
    }

    pending_question = ""
    clarified_brief  = ""
    history_view = modern_view = divination_view = ""

    # 三栏内容当前值
    cur_h = f"_正在翻阅典籍…_"
    cur_m = f"_正在分析…_"
    cur_d = f"_正在起卦…_"

    # 三栏状态
    st_h = ST_IDLE
    st_m = ST_IDLE
    st_d = ST_IDLE

    try:
        async for event in GRAPH.astream(graph_input, stream_mode="updates"):
            for node, payload in event.items():
                if not isinstance(payload, dict):
                    continue

                if "pending_question" in payload:
                    pending_question = payload.get("pending_question") or pending_question
                if "clarified_brief" in payload:
                    clarified_brief = payload.get("clarified_brief") or clarified_brief

                # ── clarifier 节点 ──
                if node == "clarifier":
                    if pending_question:
                        chat_msgs[-1] = {"role": "assistant", "content": pending_question}
                    elif clarified_brief:
                        chat_msgs[-1] = {
                            "role": "assistant",
                            "content": (
                                "📝 **背景已挖清楚，正在召唤三位顾问…**\n\n"
                                "<details><summary>📋 点开看澄清简报</summary>\n\n"
                                f"{clarified_brief}\n\n</details>"
                            ),
                        }
                        # 三方分析即将开始 → 切换到三栏并显示「思考中」
                        st_h = st_m = st_d = ST_THINKING
                        yield _Y(chat_msgs, state,
                                 msg=gr.update(interactive=False),
                                 send=gr.update(interactive=False),
                                 vh=gr.update(value=cur_h),
                                 vm=gr.update(value=cur_m),
                                 vd=gr.update(value=cur_d),
                                 sh=gr.update(value=st_h, elem_classes="status-badge status-thinking"),
                                 sm=gr.update(value=st_m, elem_classes="status-badge status-thinking"),
                                 sd=gr.update(value=st_d, elem_classes="status-badge status-thinking"),
                                 clarify=gr.update(visible=False),
                                 panels=gr.update(visible=True),
                                 synth=gr.update(visible=True, interactive=False))
                        continue

                # ── historian ──
                if node == "historian":
                    content = payload.get("history_view") or ""
                    if content:
                        history_view = content
                        cur_h = content
                        state["history_view"] = content
                        st_h = ST_DONE

                # ── modernist ──
                if node == "modernist":
                    content = payload.get("modern_view") or ""
                    if content:
                        modern_view = content
                        cur_m = content
                        state["modern_view"] = content
                        st_m = ST_DONE

                # ── diviner ──
                if node == "diviner":
                    content = payload.get("divination_view") or ""
                    if content:
                        divination_view = content
                        cur_d = content
                        state["divination_view"] = content
                        st_d = ST_DONE

                # 实时推送状态
                def _sc(s):
                    if s == ST_DONE:     return "status-badge status-done"
                    if s == ST_THINKING: return "status-badge status-thinking"
                    if s == ST_ERROR:    return "status-badge status-error"
                    return "status-badge status-idle"

                yield _Y(chat_msgs, state,
                         msg=gr.update(interactive=False),
                         send=gr.update(interactive=False),
                         vh=gr.update(value=cur_h),
                         vm=gr.update(value=cur_m),
                         vd=gr.update(value=cur_d),
                         sh=gr.update(value=st_h, elem_classes=_sc(st_h)),
                         sm=gr.update(value=st_m, elem_classes=_sc(st_m)),
                         sd=gr.update(value=st_d, elem_classes=_sc(st_d)),
                         panels=gr.update(visible=True) if clarified_brief else None,
                         synth=gr.update(visible=True, interactive=False) if clarified_brief else None)

    except Exception as e:
        err_short = f"{type(e).__name__}: {str(e)[:120]}"
        err_msg = f"⚠️ **分析出错**\n\n```\n{err_short}\n```\n\n请点 **🔄 新对话** 重试。"
        chat_msgs[-1] = {"role": "assistant", "content": err_msg}
        state["phase"] = "done"
        # 未完成的栏标记为出错
        if st_h == ST_THINKING: st_h = ST_ERROR
        if st_m == ST_THINKING: st_m = ST_ERROR
        if st_d == ST_THINKING: st_d = ST_ERROR

        def _sc(s):
            if s == ST_DONE:  return "status-badge status-done"
            if s == ST_ERROR: return "status-badge status-error"
            return "status-badge status-idle"

        yield _Y(chat_msgs, state,
                 msg=gr.update(value="", interactive=False),
                 send=gr.update(interactive=False),
                 sh=gr.update(value=st_h, elem_classes=_sc(st_h)),
                 sm=gr.update(value=st_m, elem_classes=_sc(st_m)),
                 sd=gr.update(value=st_d, elem_classes=_sc(st_d)))
        return

    # ── 一轮结束 ──
    if pending_question:
        state["chat_history"].append({"role": "assistant", "content": pending_question})
        state["phase"] = "clarifying"
        yield _Y(chat_msgs, state,
                 msg=gr.update(value="", interactive=True, placeholder="继续回答…"),
                 send=gr.update(interactive=True))
    else:
        state["clarified_brief"] = clarified_brief
        state["phase"] = "analyzing"

        def _sc(s):
            if s == ST_DONE:  return "status-badge status-done"
            if s == ST_ERROR: return "status-badge status-error"
            return "status-badge status-idle"

        yield _Y(chat_msgs, state,
                 msg=gr.update(value="", interactive=False),
                 send=gr.update(interactive=False),
                 vh=gr.update(value=cur_h),
                 vm=gr.update(value=cur_m),
                 vd=gr.update(value=cur_d),
                 sh=gr.update(value=st_h, elem_classes=_sc(st_h)),
                 sm=gr.update(value=st_m, elem_classes=_sc(st_m)),
                 sd=gr.update(value=st_d, elem_classes=_sc(st_d)),
                 clarify=gr.update(visible=False),
                 panels=gr.update(visible=True),
                 synth=gr.update(visible=True, interactive=False),
                 report=gr.update(visible=False))


# ──────────────────────────────────────────────
# 阶段2：用户表态
# ──────────────────────────────────────────────

def on_reaction_change(h_val, m_val, d_val, state: dict):
    state["history_reaction"]    = h_val or ""
    state["modern_reaction"]     = m_val or ""
    state["divination_reaction"] = d_val or ""
    all_selected = bool(h_val and m_val and d_val)
    if all_selected and state.get("phase") == "analyzing":
        state["phase"] = "awaiting_reaction"
    label = "🔮 整体分析" if all_selected else "🔮 整体分析（请先对三个视角各选一个态度）"
    return state, gr.update(interactive=all_selected, value=label)


# ──────────────────────────────────────────────
# 阶段3：综合分析
# ──────────────────────────────────────────────

async def on_synthesize(state: dict):
    # 第一帧：立刻锁住按钮 + 显示加载状态，让用户知道在运行
    yield (state,
           gr.update(value="_🔮 综合分析中，请稍候…_", visible=True),
           gr.update(interactive=False, value="⏳ 综合分析中…"))

    try:
        import asyncio
        result = await asyncio.to_thread(synthesizer_node, state)
        state["final_report"] = result.get("final_report", "")
        state["phase"] = "done"
        yield (state,
               gr.update(value=state["final_report"], visible=True),
               gr.update(interactive=False, value="✅ 整体分析完成"))
    except Exception as e:
        err = f"⚠️ **综合分析出错**\n\n```\n{type(e).__name__}: {str(e)[:200]}\n```\n\n请重试。"
        state["phase"] = "done"
        yield (state,
               gr.update(value=err, visible=True),
               gr.update(interactive=True, value="🔮 整体分析（点击重试）"))


# ──────────────────────────────────────────────
# 重置
# ──────────────────────────────────────────────

def on_reset():
    s = _new_state()
    return (
        [],                 # chat
        s,                  # state
        gr.update(value="", interactive=True,
                  placeholder="把你正在纠结的事写在这里…"),  # msg
        gr.update(interactive=True),   # send_btn
        gr.update(value="_正在翻阅典籍…_"),  # view_history
        gr.update(value="_正在分析…_"),      # view_modern
        gr.update(value="_正在起卦…_"),      # view_divination
        gr.update(value=ST_IDLE, elem_classes="status-badge status-idle"),  # st_history
        gr.update(value=ST_IDLE, elem_classes="status-badge status-idle"),  # st_modern
        gr.update(value=ST_IDLE, elem_classes="status-badge status-idle"),  # st_divination
        gr.update(visible=True),   # clarify_col
        gr.update(visible=False),  # panel_row
        gr.update(visible=False, interactive=False,
                  value="🔮 整体分析（请先对三个视角各选一个态度）"),  # synth_btn
        gr.update(value="", visible=False),  # final_report
        gr.update(),               # reset_btn（占位）
        # radio 清空
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
    )


# ──────────────────────────────────────────────
# 布局
# ──────────────────────────────────────────────

REACTION_CHOICES = ["我认可", "我迷茫", "我否认"]

with gr.Blocks(title="决策三镜") as demo:

    state = gr.State(value=_new_state())

    with gr.Column(elem_classes="welcome-card"):
        gr.Markdown(WELCOME)

    # ── 澄清区 ──
    with gr.Column(visible=True) as clarify_col:
        chat = gr.Chatbot(elem_id="chat-box", height=480, show_label=False)
        with gr.Row():
            msg = gr.Textbox(
                placeholder="把你正在纠结的事写在这里…",
                scale=8, container=False, lines=1, max_lines=5, autofocus=True,
            )
            send_btn = gr.Button("发送 ↑", variant="primary", scale=1)

        gr.Markdown("**💡 试试这些问题：**")
        with gr.Row():
            for ex in EXAMPLES:
                gr.Button(ex, size="sm").click(lambda x=ex: x, outputs=msg)

    # ── 三栏分析区（初始隐藏） ──
    with gr.Row(visible=False) as panel_row:

        with gr.Column(elem_classes="panel-col panel-history"):
            gr.Markdown("### 📜 过去 · 历史")
            st_history = gr.Markdown(ST_IDLE, elem_classes="status-badge status-idle")
            view_history = gr.Markdown("_正在翻阅典籍…_", elem_classes="panel-content")
            radio_history = gr.Radio(REACTION_CHOICES, label="你的态度",
                                     elem_classes="reaction-radio")

        with gr.Column(elem_classes="panel-col panel-modern"):
            gr.Markdown("### 🧭 现在 · 现代")
            st_modern = gr.Markdown(ST_IDLE, elem_classes="status-badge status-idle")
            view_modern = gr.Markdown("_正在分析…_", elem_classes="panel-content")
            radio_modern = gr.Radio(REACTION_CHOICES, label="你的态度",
                                    elem_classes="reaction-radio")

        with gr.Column(elem_classes="panel-col panel-diviner"):
            gr.Markdown("### ☯ 将来 · 周易")
            st_divination = gr.Markdown(ST_IDLE, elem_classes="status-badge status-idle")
            view_divination = gr.Markdown("_正在起卦…_", elem_classes="panel-content")
            radio_divination = gr.Radio(REACTION_CHOICES, label="你的态度",
                                        elem_classes="reaction-radio")

    synth_btn = gr.Button(
        "🔮 整体分析（请先对三个视角各选一个态度）",
        variant="primary", visible=False, interactive=False,
        elem_id="synth-btn",
    )

    gr.Markdown("### 📋 综合决策报告")
    final_report = gr.Markdown("", visible=False, elem_id="final-report")

    with gr.Row():
        reset_btn = gr.Button("🔄 新对话", size="sm")
        gr.Markdown(
            "<div style='text-align:right;color:#888;font-size:13px'>"
            "三视角并行 · 仅作辅助 · 决策权在你"
            "</div>"
        )

    # ── 事件绑定 ──

    # send_outputs: 15 项
    send_outputs = [
        chat, state, msg, send_btn,
        view_history, view_modern, view_divination,
        st_history, st_modern, st_divination,
        clarify_col, panel_row, synth_btn, final_report,
        reset_btn,
    ]
    send_btn.click(on_user_send, inputs=[msg, chat, state], outputs=send_outputs)
    msg.submit(on_user_send,    inputs=[msg, chat, state], outputs=send_outputs)

    reaction_inputs  = [radio_history, radio_modern, radio_divination, state]
    reaction_outputs = [state, synth_btn]
    radio_history.change(on_reaction_change,   inputs=reaction_inputs, outputs=reaction_outputs)
    radio_modern.change(on_reaction_change,    inputs=reaction_inputs, outputs=reaction_outputs)
    radio_divination.change(on_reaction_change, inputs=reaction_inputs, outputs=reaction_outputs)

    synth_btn.click(on_synthesize, inputs=[state],
                    outputs=[state, final_report, synth_btn])

    # reset_outputs: 15 + 3 radio = 18 项
    reset_btn.click(
        on_reset,
        outputs=[
            chat, state, msg, send_btn,
            view_history, view_modern, view_divination,
            st_history, st_modern, st_divination,
            clarify_col, panel_row, synth_btn, final_report,
            reset_btn,
            radio_history, radio_modern, radio_divination,
        ],
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="indigo"))
