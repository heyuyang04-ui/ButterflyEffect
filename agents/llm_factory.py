"""根据 config.yaml + .env 路由到对应 provider 的 LLM 实例。

设计：
- .env 用"四件套预设"（{NAME}_API_KEY / _BASE_URL / _MODEL / _PROTOCOL）
- config.yaml 里每个 agent 只引用 preset 名 + 自己的 temperature
- 工厂按 PROTOCOL 字段决定走 OpenAI 兼容 SDK 还是 Anthropic 原生 SDK
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_preset(preset: str) -> dict[str, str]:
    """从环境变量里读出 preset 的四件套。"""
    return {
        "api_key": os.getenv(f"{preset}_API_KEY", ""),
        "base_url": os.getenv(f"{preset}_BASE_URL", ""),
        "model": os.getenv(f"{preset}_MODEL", ""),
        "protocol": os.getenv(f"{preset}_PROTOCOL", "openai").strip().lower(),
    }


def get_llm(role: str):
    """根据 agent 角色名返回 LangChain Chat 模型实例。"""
    cfg = _load_config()
    role_cfg = cfg.get("agents", {}).get(role)
    if not role_cfg:
        raise ValueError(f"agent '{role}' 未在 config.yaml 的 agents 段中定义")

    preset = role_cfg.get("preset")
    if not preset:
        raise ValueError(f"agent '{role}' 未指定 preset")
    temperature = role_cfg.get("temperature", 0.5)

    env = _read_preset(preset)
    if not env["model"]:
        raise RuntimeError(
            f"preset '{preset}' 缺少 {preset}_MODEL，请在 .env 配置"
        )

    if env["protocol"] == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not env["api_key"]:
            raise RuntimeError(f"preset '{preset}' 缺少 {preset}_API_KEY")
        # Claude 4 系列（opus-4/sonnet-4/haiku-4）不接受 temperature 参数
        kwargs = {
            "model": env["model"],
            "api_key": env["api_key"],
        }
        # 走中转代理时需要覆盖默认的 anthropic 端点
        if env["base_url"]:
            kwargs["base_url"] = env["base_url"]
        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    if not env["api_key"]:
        raise RuntimeError(f"preset '{preset}' 缺少 {preset}_API_KEY")
    return ChatOpenAI(
        model=env["model"],
        temperature=temperature,
        api_key=env["api_key"],
        base_url=env["base_url"] or None,
    )
