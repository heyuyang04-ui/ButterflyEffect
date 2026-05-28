"""Gradio Web UI —— 多视角决策辅助系统（ChatGPT 风格单聊天流）。"""
from __future__ import annotations

import gradio as gr

from agents.state import GraphState
from graph import GRAPH

WELCOME = """# 🪞 决策三镜
**输入你正在纠结的事**，主访谈员会先反问几轮，把背景挖清楚；
然后由 **史学家**、**术数师**、**战略咨询师** 三位顾问并行作答；
最后由主访谈员综合三方观点给出报告。"""

EXAMPLES = [
    "我应不应该从国企辞职去创业做 AI？",
    "下个月要不要把房子卖了换个城市发展？",
    "春节要不要回老家相亲？",
    "孩子小升初要不要拼鸡娃？",
]

CSS = """
.gradio-container { max-width: 860px !important; margin: 0 auto !important; }
#chat { box-shadow: none !important; }
#chat .message-wrap { font-size: 15px; line-height: 1.7; }
#chat .bot { background: #f7f7f8 !important; border: none !important; }
#chat .user { background: #ffffff !important; }
.message-row.bubble.bot-row { padding: 6px 0; }
footer { display: none !important; }
.welcome-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #fbf3ff 100%);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
"""


def _new_state() -> dict:
    return {
        "query": "",
        "chat_history": [],
        "phase": "init",  # init | clarifying | done
    }


async def on_user_send(user_msg: str, chat_msgs: list, state: dict):
    """用户发言。可能是初始 query，也可能是回答反问。"""
    user_msg = (user_msg or "").strip()
    if not user_msg:
        yield chat_msgs, state, gr.update(), gr.update()
        return
    if state["phase"] == "done":
        # 已完成的会话，提示重置
        chat_msgs = chat_msgs + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "本轮分析已完成。点 **🔄 新对话** 开始下一轮。"},
        ]
        yield chat_msgs, state, gr.update(value=""), gr.update(interactive=True)
        return

    if state["phase"] == "init":
        state["query"] = user_msg
        state["phase"] = "clarifying"
    else:
        state["chat_history"].append({"role": "user", "content": user_msg})

    chat_msgs = chat_msgs + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "_正在思考…_"},
    ]
    yield chat_msgs, state, gr.update(value="", interactive=False), gr.update(interactive=False)

    graph_input: GraphState = {
        "query": state["query"],
        "chat_history": list(state["chat_history"]),
    }

    pending_question = ""
    clarified_brief = ""

    # 占位消息的索引（按需赋值）
    pending_idx = len(chat_msgs) - 1
    history_idx = None
    diviner_idx = None
    modernist_idx = None

    graph_error = None
    try:
        async for event in GRAPH.astream(graph_input, stream_mode="updates"):
            for node, payload in event.items():
                if not isinstance(payload, dict):
                    continue

                if "pending_question" in payload:
                    pending_question = payload.get("pending_question") or pending_question
                if "clarified_brief" in payload:
                    clarified_brief = payload.get("clarified_brief") or clarified_brief

                if node == "clarifier":
                    if pending_question:
                        chat_msgs[pending_idx] = {"role": "assistant", "content": pending_question}
                    elif clarified_brief:
                        chat_msgs[pending_idx] = {
                            "role": "assistant",
                            "content": (
                                "📝 **背景已挖清楚，开始三方分析…**\n\n"
                                "<details><summary>📋 点开看澄清简报</summary>\n\n"
                                f"{clarified_brief}\n\n</details>"
                            ),
                        }
                        chat_msgs.append({"role": "assistant", "content": "📜 **史学家**\n\n_正在翻阅典籍…_"})
                        history_idx = len(chat_msgs) - 1
                        chat_msgs.append({"role": "assistant", "content": "☯ **术数师**\n\n_正在起卦…_"})
                        diviner_idx = len(chat_msgs) - 1
                        chat_msgs.append({"role": "assistant", "content": "🧭 **战略咨询师**\n\n_正在分析…_"})
                        modernist_idx = len(chat_msgs) - 1

                if node == "historian" and history_idx is not None:
                    content = payload.get("history_view") or ""
                    if content:
                        chat_msgs[history_idx] = {"role": "assistant", "content": content}
                if node == "diviner" and diviner_idx is not None:
                    content = payload.get("divination_view") or ""
                    if content:
                        chat_msgs[diviner_idx] = {"role": "assistant", "content": content}
                if node == "modernist" and modernist_idx is not None:
                    content = payload.get("modern_view") or ""
                    if content:
                        chat_msgs[modernist_idx] = {"role": "assistant", "content": content}

                if node == "synthesizer":
                    content = payload.get("final_report") or ""
                    if content:
                        chat_msgs.append({"role": "assistant", "content": content})

                yield chat_msgs, state, gr.update(interactive=False), gr.update(interactive=False)

    except Exception as e:
        graph_error = e
        err_msg = f"⚠️ **分析过程中发生错误**\n\n```\n{type(e).__name__}: {e}\n```\n\n请点 **🔄 新对话** 重试。"
        chat_msgs[pending_idx] = {"role": "assistant", "content": err_msg}
        yield chat_msgs, state, gr.update(interactive=False), gr.update(interactive=False)

    # 一轮结束后维护 state
    if graph_error:
        state["phase"] = "done"
        yield (
            chat_msgs,
            state,
            gr.update(value="", interactive=False, placeholder="发生错误，请点【🔄 新对话】重试"),
            gr.update(interactive=False),
        )
        return

    if pending_question:
        state["chat_history"].append({"role": "assistant", "content": pending_question})
        # 仍在澄清阶段，可继续输入
        yield (
            chat_msgs,
            state,
            gr.update(value="", interactive=True, placeholder="继续回答…"),
            gr.update(interactive=True),
        )
    else:
        state["phase"] = "done"
        yield (
            chat_msgs,
            state,
            gr.update(value="", interactive=False, placeholder="本轮已完成，点【🔄 新对话】开始下一轮"),
            gr.update(interactive=False),
        )


def on_reset():
    return (
        [],
        _new_state(),
        gr.update(value="", interactive=True, placeholder="把你正在纠结的事写在这里…"),
        gr.update(interactive=True),
    )


def on_example(ex: str):
    return ex


with gr.Blocks(title="决策三镜") as demo:
    state = gr.State(value=_new_state())

    with gr.Column(elem_classes="welcome-card"):
        gr.Markdown(WELCOME)

    chat = gr.Chatbot(
        elem_id="chat",
        height=560,
        show_label=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="把你正在纠结的事写在这里…（例：要不要从国企辞职去创业做 AI？）",
            scale=8,
            container=False,
            lines=1,
            max_lines=6,
            autofocus=True,
        )
        send_btn = gr.Button("发送 ↑", variant="primary", scale=1)

    with gr.Row():
        reset_btn = gr.Button("🔄 新对话", size="sm")
        gr.Markdown(
            "<div style='text-align:right;color:#888;font-size:13px'>"
            "三视角并行·主访谈员综合 · 仅作辅助 · 决策权在你"
            "</div>"
        )

    gr.Markdown("**💡 试试这些问题：**")
    with gr.Row():
        for example in EXAMPLES:
            ex_btn = gr.Button(example, size="sm")
            ex_btn.click(lambda x=example: x, outputs=msg)

    outputs = [chat, state, msg, send_btn]
    send_btn.click(on_user_send, inputs=[msg, chat, state], outputs=outputs)
    msg.submit(on_user_send, inputs=[msg, chat, state], outputs=outputs)
    reset_btn.click(on_reset, outputs=outputs)


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="indigo"))
