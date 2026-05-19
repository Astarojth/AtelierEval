# app.py
import gradio as gr
from config import THEME, CSS, TITLE
from data_loader import load_tasks
from state_management import get_initial_state
import view_functions as views
from ui_components import create_ui_components
from event_bindings import bind_events

def create_app():
    """创建并配置Gradio应用"""
    # 在应用启动时加载所有任务
    load_tasks()

    # 创建Gradio界面
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown("# AtelierEval Comprehensive Assessment Platform")
        app_state = gr.State(get_initial_state())

        # 创建所有UI组件
        components = create_ui_components()
        
        # 获取所有视图输出
        all_view_outputs = views.get_all_views()

        # 绑定所有事件处理器
        bind_events(components, app_state, all_view_outputs)

        return demo

def main():
    """主函数，启动应用"""
    demo = create_app()
    demo.launch(theme=THEME, css=CSS)

if __name__ == "__main__":
    main()
