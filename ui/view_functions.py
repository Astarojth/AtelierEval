# view_functions.py
import gradio as gr

# 缓存“可作为输出的组件”列表，供事件绑定和包装器共同使用
_ALL_OUTPUT_COMPONENTS: list = []

def set_all_output_components(components_list: list):
    """在应用初始化/事件绑定时设置一次，全局生效"""
    global _ALL_OUTPUT_COMPONENTS
    _ALL_OUTPUT_COMPONENTS = filter_valid_output_components(components_list)

def get_all_output_components() -> list:
    return list(_ALL_OUTPUT_COMPONENTS)

def get_all_output_components_len() -> int:
    return len(_ALL_OUTPUT_COMPONENTS)

def filter_valid_output_components(components_list: list) -> list:
    """
    过滤掉布局容器（如 Column/Row/Group/Tabs/Accordion），仅保留可作为输出的组件
    """
    from gradio.components import Component as GrComponent
    # 某些版本的 gr.State 需要单独判断
    def _is_valid(c):
        # gr.State 在 gradio>=4 里是 components.State
        return isinstance(c, (GrComponent, gr.State))
    return [c for c in components_list if _is_valid(c)]

# 使用一个字典来存储所有视图的引用，而不是多个全局变量
# Use a dictionary to store references to all views instead of multiple global variables
_VIEWS = {}

# --- 核心功能函数 (Core Functions) ---

def set_views(views_dict: dict):
    """
    Receives and stores the dictionary of all view components from app.py.
    Also dynamically creates global variables for compatibility.
    """
    global _VIEWS
    _VIEWS = views_dict
    
    # 动态地将字典的键值对设置为本模块的全局变量
    # This allows calls like `views.intro_view` to work in other files.
    for name, view in views_dict.items():
        globals()[name] = view

def get_all_views() -> list:
    """
    Returns a list of all registered view components.
    """
    return list(_VIEWS.values())

def update_views(active_view) -> tuple:
    """
    Makes the specified active_view visible and all others invisible.
    Returns a tuple containing the updated visibility states for all views.
    """
    updates = []
    for view_component in _VIEWS.values():
        is_visible = (view_component == active_view)
        updates.append(gr.update(visible=is_visible))
    return tuple(updates)

# --- 辅助函数 (Helper Functions) ---
# 这些函数提供了更具可读性的方式来切换视图。
# These functions provide a more readable way to switch between views.

def show_error_view(): 
    return update_views(globals().get('error_view'))

def show_intro_view(): 
    return update_views(globals().get('intro_view'))

def show_open_ended_intro(): 
    return update_views(globals().get('open_ended_intro_view'))

def show_open_ended_answering():
    return update_views(globals().get('open_ended_answering_view'))

def show_open_ended_complete(): 
    return update_views(globals().get('open_ended_complete_view'))

def show_constrained_intro(): 
    return update_views(globals().get('constrained_intro_view'))

def show_constrained_answering():
    return update_views(globals().get('constrained_answering_view'))

# 新增: 为模仿题视图添加辅助函数
# NEW: Add helper functions for the imitation task views
def show_imit_intro():
    return update_views(globals().get('imit_intro_view'))

def show_imit_answering():
    return update_views(globals().get('imit_answering_view'))

def show_processing(): 
    return update_views(globals().get('processing_view'))

def show_results(): 
    return update_views(globals().get('results_view'))

# 新增：最终完成视图
def show_final_complete():
    return update_views(globals().get('final_complete_view'))

def get_current_view_updates(state):
    """根据当前阶段返回视图更新"""
    current_phase = state.get('exam_phase', 'intro')

    if 'open_ended' in current_phase:
        return show_open_ended_answering()
    elif 'constrained' in current_phase:
        return show_constrained_answering()
    elif 'imit' in current_phase:
        return show_imit_answering()
    else:
        return show_welcome()