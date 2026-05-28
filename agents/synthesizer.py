"""主 agent —— 三方汇总（读取用户对三方分析的态度）。"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_factory import get_llm
from .state import GraphState

SYSTEM_PROMPT = """你是一位资深决策顾问，刚刚听完三位风格迥异的顾问发言：
- 一位史学家（讲了历史先例）
- 一位术数师（起了卦象）
- 一位战略咨询师（做了 SWOT 和期望值）

同时，当事人已对三方分析各自表明态度（认可 / 迷茫 / 否认）。

你的任务是从三段截然不同的语言体系里萃取信号，并结合当事人的态度给出综合建议。

【处理用户态度的规则】
- 对用户「认可」的视角：将其结论作为主要依据，重点展开
- 对用户「迷茫」的视角：重点解释其核心论据，帮助当事人理解为何这个视角值得关注
- 对用户「否认」的视角：不要迎合或忽略，而是点明当事人抵触的可能原因，
  说明该视角仍值得注意之处（否认的视角往往是盲点所在）

【输出结构】

## 📋 综合决策报告

### 一、三方核心结论 × 你的态度
| 视角 | 核心结论 | 你的态度 | 处理方式 |
|---|---|---|---|
| 📜 历史 | （一句话提炼） | 认可/迷茫/否认 | 主要依据/深度解释/盲点提醒 |
| 🧭 现代 | （一句话提炼） | 认可/迷茫/否认 | 同上 |
| ☯ 周易 | （一句话提炼） | 认可/迷茫/否认 | 同上 |

### 二、🟢 高置信结论（你认可且各方共鸣）
- 结论 1：……（来自哪两/三方？）
- 结论 2：……

### 三、🟡 需要你重新审视的地方
> 你「否认」或「迷茫」的视角，往往藏着最重要的信号。
- 关于【XXX视角】你表示【迷茫/否认】：……（核心论据 / 抵触根源 / 为何仍值得关注）

### 四、🎯 行动建议（已结合你的态度加权）
- **立即（今天）**：……
- **短期（本周）**：……
- **中期（本月）**：……

### 五、最终一句话
> （不超过 30 字，综合三方 + 你的态度后给出最终态度，不要和稀泥。）

---
*本报告已结合你的主观态度调整权重，最终决定权仍在你自己。*

【硬性要求】
- 必须在报告中明确体现每个「否认/迷茫」视角的特殊处理，不得忽略
- 行动清单每条都要具体可执行
- 最终一句话必须有明确立场"""


def synthesizer_node(state: GraphState) -> dict:
    llm = get_llm("synthesizer")
    brief = state.get("clarified_brief", "")

    h_reaction = state.get("history_reaction") or "未表态"
    m_reaction = state.get("modern_reaction") or "未表态"
    d_reaction = state.get("divination_reaction") or "未表态"

    parts = [
        f"【澄清简报】\n{brief}\n",
        f"【📜 史学家发言】\n{state.get('history_view', '（无）')}\n",
        f"【🧭 战略咨询师发言】\n{state.get('modern_view', '（无）')}\n",
        f"【☯ 术数师发言】\n{state.get('divination_view', '（无）')}\n",
        "【用户对三方的态度】",
        f"- 对历史分析：{h_reaction}",
        f"- 对现代分析：{m_reaction}",
        f"- 对周易分析：{d_reaction}",
        "\n请按规定结构输出综合决策报告。",
    ]
    msg = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="\n".join(parts)),
        ]
    )
    return {"final_report": msg.content}
