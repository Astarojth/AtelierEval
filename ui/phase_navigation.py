# phase_navigation.py
import time
import view_functions as views
from state_management import get_initial_state

def auto_transition_to_open_ended_intro(state):
    """Auto transition to open-ended intro after a delay."""
    time.sleep(2)
    state["exam_phase"] = "open_ended_intro"
    return state, *views.show_open_ended_intro()

def start_open_ended_section(state):
    """Start the Open-Ended section."""
    state["exam_phase"] = "open_ended_answering"
    state["current_open_ended_index"] = 0
    return state

def show_open_ended_complete_transition(state):
    """Show Open-Ended completion transition."""
    state["exam_phase"] = "open_ended_complete"
    return state, *views.show_open_ended_complete()

def auto_transition_to_constrained_intro(state):
    """Auto transition to constrained intro after a delay."""
    time.sleep(2)
    state["exam_phase"] = "constrained_intro"
    return state, *views.show_constrained_intro()

def start_constrained_section(state):
    """Start the Constrained section."""
    state["exam_phase"] = "constrained_answering"
    state["current_constrained_index"] = 0
    return state

def auto_transition_to_imit_intro(state):
    """
    Switches view to the imitation section intro.
    """
    time.sleep(1.5) # 给用户一点时间看过渡信息
    state["exam_phase"] = "imit_intro"
    return state, *views.update_views(views.imit_intro_view)

def start_imit_section(state):
    """
    Initializes the imitation section state.
    """
    state["exam_phase"] = "imit_answering"
    state["current_imit_index"] = 0
    return state

def return_to_welcome():
    """Reset the state and return to the welcome screen."""
    return get_initial_state(), *views.show_intro_view()

# ============================================================================
# 动态导航辅助函数 (Dynamic Navigation Helper Functions)
# 用于支持随机题型顺序 (Support for randomized question type order)
# ============================================================================

def get_current_question_type(state):
    """
    获取当前应该进行的题型
    Get the current question type based on randomized order.
    """
    section_index = state.get("current_section_index", 0)
    question_order = state.get("question_type_order", ['open_ended', 'constrained', 'imit'])

    if section_index < len(question_order):
        return question_order[section_index]
    return None

def get_next_question_type(state):
    """
    获取下一个题型
    Get the next question type in the randomized order.
    """
    section_index = state.get("current_section_index", 0)
    question_order = state.get("question_type_order", ['open_ended', 'constrained', 'imit'])

    next_index = section_index + 1
    if next_index < len(question_order):
        return question_order[next_index]
    return None

def is_last_question_type(state):
    """
    判断当前是否是最后一个题型
    Check if the current section is the last question type.
    """
    section_index = state.get("current_section_index", 0)
    question_order = state.get("question_type_order", ['open_ended', 'constrained', 'imit'])
    return section_index >= len(question_order) - 1

def get_section_intro_view(question_type):
    """
    根据题型获取对应的intro view
    Get the intro view for a given question type.
    """
    view_map = {
        'open_ended': views.open_ended_intro_view,
        'constrained': views.constrained_intro_view,
        'imit': views.imit_intro_view
    }
    return view_map.get(question_type)

def get_section_complete_view(question_type):
    """
    根据题型获取对应的complete view
    Get the complete view for a given question type.

    注意：只有 Open-Ended 有 complete_view
    Constrained 和 Imit 没有 complete_view，会直接转换

    Note: Only Open-Ended has complete_view
    Constrained and Imit don't have complete_view, will transition directly
    """
    view_map = {
        'open_ended': views.open_ended_complete_view,
        # constrained 和 imit 没有 complete_view，返回 None
        # constrained and imit don't have complete_view, return None
    }
    return view_map.get(question_type, None)

def get_section_answering_view(question_type):
    """
    根据题型获取对应的answering view
    Get the answering view for a given question type.
    """
    view_map = {
        'open_ended': views.open_ended_answering_view,
        'constrained': views.constrained_answering_view,
        'imit': views.imit_answering_view
    }
    return view_map.get(question_type)

def show_dynamic_section_complete(state, current_section_type):
    """
    显示当前题型的完成页面（如果存在）
    Show completion transition for the current question type (if it exists).

    对于没有 complete_view 的题型（constrained, imit），保持在 answering_view
    For question types without complete_view (constrained, imit), keep answering_view
    """
    state["exam_phase"] = f"{current_section_type}_complete"
    complete_view = get_section_complete_view(current_section_type)

    if complete_view:
        # 有 complete_view 的题型（Open-Ended），显示完成页面
        # Question types with complete_view (Open-Ended), show completion page
        return state, *views.update_views(complete_view)
    else:
        # 没有 complete_view 的题型（Constrained, Imit），保持在 answering_view
        # Question types without complete_view (Constrained, Imit), keep answering_view
        answering_view = get_section_answering_view(current_section_type)
        return state, *views.update_views(answering_view)

def dynamic_transition_to_next_section(state):
    """
    动态转换到下一个题型，基于随机顺序
    Dynamically transition to the next question type based on randomized order.

    如果是最后一个题型，则不进行转换（由Complete按钮直接触发结果处理）
    If it's the last question type, no transition occurs (results processing triggered by Complete button).

    延迟时间根据当前题型是否有complete_view决定：
    - 有 complete_view: 延迟2秒（让用户看到完成页面）
    - 没有 complete_view: 延迟0.5秒（快速转换）
    """
    # 检查是否是最后一个题型
    if is_last_question_type(state):
        # 最后一个题型不需要自动转换，Complete按钮会直接触发结果处理
        return state, *views.update_views(None)  # 保持当前视图

    # 获取当前题型，检查是否有 complete_view
    current_section_index = state.get("current_section_index", 0)
    question_order = state.get("question_type_order", ['open_ended', 'constrained', 'imit'])
    current_type = question_order[current_section_index] if current_section_index < len(question_order) else None

    # 根据是否有 complete_view 决定延迟时间
    has_complete_view = get_section_complete_view(current_type) is not None
    delay_time = 2.0 if has_complete_view else 0.5  # Open-Ended: 2秒, Constrained/Imit: 0.5秒

    # 添加延迟以提供更好的用户体验
    time.sleep(delay_time)

    # 移动到下一个题型
    state["current_section_index"] += 1
    next_type = get_current_question_type(state)

    if next_type:
        # 更新exam_phase并显示下一个题型的intro
        state["exam_phase"] = f"{next_type}_intro"
        next_intro_view = get_section_intro_view(next_type)
        if next_intro_view:
            return state, *views.update_views(next_intro_view)

    # 如果没有下一个题型，返回错误视图
    return state, *views.show_error_view()

def start_first_section(state):
    """
    开始第一个题型（根据随机顺序）
    Start the first question type section based on randomized order.
    """
    # 重置section index
    state["current_section_index"] = 0
    first_type = get_current_question_type(state)

    if first_type:
        state["exam_phase"] = f"{first_type}_intro"
        first_intro_view = get_section_intro_view(first_type)
        if first_intro_view:
            return state, *views.update_views(first_intro_view)

    return state, *views.show_error_view()
