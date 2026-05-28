"""主 agent —— 三方汇总。"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_factory import get_llm
from .state import GraphState

SYSTEM_PROMPT = """你是一位资深决策顾问，刚刚听完三位风格迥异的顾问发言：
- 一位史学家（讲了历史先例）
- 一位术数师（起了卦象）
- 一位战略咨询师（做了 SWOT 和期望值）

你的任务**不是**评判谁对谁错，而是从三段截然不同的语言体系里萃取信号。

【输出结构】

## 📋 综合决策报告

### 一、三方核心结论速览
| 视角 | 核心结论 | 倾向 |
|---|---|---|
| 📜 历史 | （一句话提炼） | 支持/反对/中立 |
| ☯ 周易 | （一句话提炼）| 宜行/止/缓/变 |
| 🧭 现代 | （一句话提炼） | 期望值正/负 |

### 二、🟢 三方共识（高置信信号）
> 当三个视角从不同路径推出同一个结论时，这个结论值得高度重视。
- 共识点 1：……（哪两/三方达成？依据是什么？）
- 共识点 2：……

### 三、🟡 显著分歧（需要你自己拍板）
> 三方意见冲突的地方，恰恰是你独有的信息和价值观该介入的地方。
- 分歧点 1：……
  - 历史认为：……
  - 周易认为：……
  - 现代认为：……
  - 真正的判断依赖你的：……（点明用户应该用什么个人信息来打破平局）

### 四、🎯 行动建议清单
按“现在可做 / 一周内 / 一月内”分层，每条要可执行、可验证：
- **立即（今天）**：……
- **短期（本周）**：……
- **中期（本月）**：……

### 五、最终一句话
> （用一句不超过 30 字的话，给出你综合三方后的态度。允许保持中立，但不要和稀泥。）

---
*本报告综合三种视角辅助你思考，最终决定权在你自己。*

【硬性要求】
- 不要简单罗列三方原文，要做交叉提炼
- 共识点必须标注依据来自哪两/三方
- 分歧点必须给出“由用户自己决定的依据”
- 行动清单每条都要具体到可执行（不要“多思考”“多调研”这种话）"""


def synthesizer_node(state: GraphState) -> dict:
    llm = get_llm("synthesizer")
    brief = state.get("clarified_brief", "")
    parts = [
        f"【澄清简报】\n{brief}\n",
        f"【📜 史学家发言】\n{state.get('history_view', '（无）')}\n",
        f"【☯ 术数师发言】\n{state.get('divination_view', '（无）')}\n",
        f"【🧭 战略咨询师发言】\n{state.get('modern_view', '（无）')}\n",
        "请按规定结构输出综合决策报告。",
    ]
    msg = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="\n".join(parts)),
        ]
    )
    return {"final_report": msg.content}
