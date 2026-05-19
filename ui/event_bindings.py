# event_bindings.py
import gradio as gr
from phase_navigation import (
    auto_transition_to_open_ended_intro, start_open_ended_section,
    show_open_ended_complete_transition, auto_transition_to_constrained_intro,
    start_constrained_section, return_to_welcome,
    auto_transition_to_imit_intro, start_imit_section,
    # 新增：动态导航函数 (NEW: Dynamic navigation functions)
    show_dynamic_section_complete, dynamic_transition_to_next_section,
    is_last_question_type, get_current_question_type
)
from question_handlers import open_ended, constrained, imit
from event_handlers import EventHandlers
import view_functions as views
from utils.timestamp_utils import record_answer_timestamp

def handle_section_complete(state, section_type):
    """
    统一处理题型完成的逻辑
    Unified handler for section completion.

    无论是否为最后一个题型，都显示完成页面
    如果是最后一个题型，后续会直接进入结果处理

    Show completion page for all question types
    If it's the last, results processing will follow
    """
    # 在完成当前题型前记录最后一题的时间戳
    # Record timestamp for the last question before completing the section
    current_index = state.get(f"current_{section_type}_index", 0)
    record_answer_timestamp(state, section_type, current_index)

    return show_dynamic_section_complete(state, section_type)

def handle_auto_transition_or_process(state):
    """
    自动转换到下一题型或处理结果
    Auto-transition to next section or process results.

    如果是最后一个题型，触发结果处理
    如果不是，转换到下一个题型的intro页面

    If it's the last question type, trigger results processing
    If not, transition to next section's intro page
    """
    if is_last_question_type(state):
        # 最后一个题型，触发结果处理
        # Last question type, trigger results processing
        for result in EventHandlers.process_results_wrapper(state):
            yield result
    else:
        # 不是最后一个，转换到下一题型
        # Not the last, transition to next section
        # 需要包含processing_status和results_report的空更新以匹配输出签名
        # Need to include empty updates for processing_status and results_report to match output signature
        new_state, *view_updates = dynamic_transition_to_next_section(state)
        yield (new_state, *view_updates, gr.update(), gr.update())

def bind_events(components, app_state, all_view_outputs):
    """绑定所有事件处理器"""

    # --- 主要导航事件 ---
    components['login_btn'].click(
        EventHandlers.handle_login_button,
        inputs=[components['user_email'], app_state, components['login_btn']],
        outputs=[app_state, *all_view_outputs, components['login_error_msg']]
    )

    components['error_restart_btn'].click(
        return_to_welcome, 
        None, 
        [app_state, *all_view_outputs]
    )
    
    components['return_to_start_btn'].click(
        EventHandlers.handle_return_button,
        inputs=[app_state],
        outputs=[app_state, *all_view_outputs]
    ).then(
        lambda state: EventHandlers.update_intro_for_round2() if state.get('assessment_round', 1) == 1 else (gr.update(), gr.update(), gr.update()),
        inputs=[app_state],
        outputs=[components['intro_markdown'], components['user_email'], components['login_btn']]
    )

    # --- Open-Ended流程事件 ---
    components['start_open_ended_btn'].click(
        start_open_ended_section, 
        [app_state], 
        [app_state]
    ).then(
        open_ended.display_open_ended_question, 
        [app_state], 
        [components['open_ended_progress'], components['open_ended_question'], components['open_ended_prompt'], 
         components['open_ended_prev_btn'], components['open_ended_next_btn'], components['open_ended_complete_btn']]
    ).then(
        lambda: views.update_views(views.open_ended_answering_view), 
        None, 
        all_view_outputs
    )
    
    components['open_ended_prompt'].change(
        open_ended.store_open_ended_answer, 
        [app_state, components['open_ended_prompt']], 
        [app_state]
    )
    
    components['open_ended_prev_btn'].click(
        lambda s: open_ended.navigate_open_ended(s, "prev"), 
        [app_state], 
        [app_state]
    ).then(
        open_ended.display_open_ended_question, 
        [app_state], 
        [components['open_ended_progress'], components['open_ended_question'], components['open_ended_prompt'],
         components['open_ended_prev_btn'], components['open_ended_next_btn'], components['open_ended_complete_btn']]
    )
    
    components['open_ended_next_btn'].click(
        lambda s: open_ended.navigate_open_ended(s, "next"), 
        [app_state], 
        [app_state]
    ).then(
        open_ended.display_open_ended_question, 
        [app_state], 
        [components['open_ended_progress'], components['open_ended_question'], components['open_ended_prompt'],
         components['open_ended_prev_btn'], components['open_ended_next_btn'], components['open_ended_complete_btn']]
    )
    
    # Open-Ended Complete按钮 - 使用动态逻辑 (Open-Ended Complete Button - Use dynamic logic)
    components['open_ended_complete_btn'].click(
        lambda s: handle_section_complete(s, 'open_ended'),
        [app_state],
        [app_state, *all_view_outputs]
    ).then(
        handle_auto_transition_or_process,
        [app_state],
        [app_state, *all_view_outputs, components['processing_status'], components['results_report']]
    ).then(
        EventHandlers.update_button_text_after_processing,
        inputs=[app_state],
        outputs=[components['return_to_start_btn']]
    )
    
    # --- Constrained流程事件 ---
    components['start_constrained_btn'].click(
        start_constrained_section, 
        [app_state], 
        [app_state]
    ).then(
        constrained.display_constrained_question, 
        [app_state], 
        [components['constrained_progress'], components['constrained_question'], components['constrained_prompt'], 
         components['constrained_prev_btn'], components['constrained_next_btn'], components['constrained_complete_btn']]
    ).then(
        lambda: views.update_views(views.constrained_answering_view), 
        None, 
        all_view_outputs
    )
    
    components['constrained_prompt'].change(
        constrained.store_constrained_answer, 
        [app_state, components['constrained_prompt']], 
        [app_state]
    )
    
    components['constrained_prev_btn'].click(
        lambda s: constrained.navigate_constrained(s, "prev"), 
        [app_state], 
        [app_state]
    ).then(
        constrained.display_constrained_question, 
        [app_state], 
        [components['constrained_progress'], components['constrained_question'], components['constrained_prompt'],
         components['constrained_prev_btn'], components['constrained_next_btn'], components['constrained_complete_btn']]
    )
    
    components['constrained_next_btn'].click(
        lambda s: constrained.navigate_constrained(s, "next"), 
        [app_state], 
        [app_state]
    ).then(
        constrained.display_constrained_question, 
        [app_state], 
        [components['constrained_progress'], components['constrained_question'], components['constrained_prompt'],
         components['constrained_prev_btn'], components['constrained_next_btn'], components['constrained_complete_btn']]
    )
    
    # Constrained Complete按钮 - 使用动态逻辑 (Constrained Complete Button - Use dynamic logic)
    components['constrained_complete_btn'].click(
        lambda s: handle_section_complete(s, 'constrained'),
        [app_state],
        [app_state, *all_view_outputs]
    ).then(
        handle_auto_transition_or_process,
        [app_state],
        [app_state, *all_view_outputs, components['processing_status'], components['results_report']]
    ).then(
        EventHandlers.update_button_text_after_processing,
        inputs=[app_state],
        outputs=[components['return_to_start_btn']]
    )

    # --- Imitation流程事件 ---
    imit_outputs = [components['imit_progress'], components['imit_target_image'], components['imit_task_info'], 
                   components['imit_prompt'], components['imit_prev_btn'], components['imit_next_btn'], components['imit_complete_btn']]
    
    components['start_imit_btn'].click(
        start_imit_section, 
        [app_state], 
        [app_state]
    ).then(
        imit.display_imit_question, 
        [app_state], 
        imit_outputs
    ).then(
        lambda: views.update_views(views.imit_answering_view), 
        None, 
        all_view_outputs
    )

    components['imit_prompt'].change(
        imit.store_imit_answer, 
        [app_state, components['imit_prompt']], 
        [app_state]
    )
    
    components['imit_prev_btn'].click(
        lambda s: imit.navigate_imit(s, "prev"), 
        [app_state], 
        [app_state]
    ).then(
        imit.display_imit_question, 
        [app_state], 
        imit_outputs
    )
    
    components['imit_next_btn'].click(
        lambda s: imit.navigate_imit(s, "next"), 
        [app_state], 
        [app_state]
    ).then(
        imit.display_imit_question, 
        [app_state], 
        imit_outputs
    )

    # --- Imitation Complete按钮 - 使用动态逻辑 (Imitation Complete Button - Use dynamic logic) ---
    components['imit_complete_btn'].click(
        lambda s: handle_section_complete(s, 'imit'),
        [app_state],
        [app_state, *all_view_outputs]
    ).then(
        handle_auto_transition_or_process,
        [app_state],
        [app_state, *all_view_outputs, components['processing_status'], components['results_report']]
    ).then(
        EventHandlers.update_button_text_after_processing,
        inputs=[app_state],
        outputs=[components['return_to_start_btn']]
    )