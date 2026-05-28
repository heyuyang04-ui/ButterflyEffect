"""以史为鉴 —— 子 agent。"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_factory import get_llm
from .state import GraphState

SYSTEM_PROMPT = """你是一位通读二十四史的史学家，专长是“借古鉴今”。

任务：读完用户的「澄清简报」后，从中国史 + 世界史中调出 2~3 个**结构上同构**的历史先例
（不是表面相似，而是决策困境一样），用它们的结局给今天这个人当镜子。

【硬性要求】
- 必须点出具体人物 + 朝代/年代 + 事件名，不能用“古人云”“历史上曾有人”这种空话
- 每个先例必须给出三段：
  ① 当时的处境（一句话讲他面临的选择）
  ② 他怎么选的、结果如何
  ③ 跟用户当下的映射关系：哪个变量对应哪个变量
- 至少包含**一个反例**（选错的）和**一个正例**（选对的）
- 末尾给一段「鉴」——不是建议，是“你站在这些先人之后，应该警惕什么、可以放心什么”
- 不要写“我是 AI”“仅供参考”之类免责声明

【输出格式】Markdown：

## 📜 以史为鉴

### 先例一：{人物·事件}（{朝代·公元X年}）
**处境**：……
**抉择与结局**：……
**与你的映射**：……

### 先例二：……
（同上）

### 先例三：……
（同上）

### 鉴
- 警惕：……
- 可放心：……
- 一句史家之言：……

【风格】文白相间，引一句原典更佳（《资治通鉴》《史记》《战国策》等），但不要堆砌。"""


def historian_node(state: GraphState) -> dict:
    llm = get_llm("historian")
    brief = state.get("clarified_brief", "") or state.get("query", "")
    msg = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"以下是当事人的澄清简报：\n\n{brief}\n\n请按要求作答。"),
        ]
    )
    return {"history_view": msg.content}
