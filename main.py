"""项目入口：启动 Gradio Web UI。"""
import gradio as gr

from app import CSS, demo


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="indigo"))
