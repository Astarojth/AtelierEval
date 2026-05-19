# event_handlers.py
import gradio as gr
from state_management import get_initial_state
from phase_navigation import return_to_welcome, start_first_section
import view_functions as views
from processing import process_and_generate_results

class EventHandlers:
    """事件处理器类，包含所有的回调函数"""
    
    @staticmethod
    def login_and_start(email, current_state):
        """登录与启动评估的函数"""
        # 简单的邮箱格式验证
        if not email or "@" not in email or "." not in email:
            error_update = gr.update(visible=True, value="<p style='color:red; text-align:center;'>Please enter a valid email address.</p>")
            return (current_state, *views.update_views(views.intro_view), error_update)

        # 登录成功，存储用户ID并进入下一阶段
        current_state['user_id'] = email
        current_state['assessment_round'] = 1  # 标记为第一轮

        success_update = gr.update(visible=False)  # 隐藏错误信息

        # 使用动态导航开始第一个题型（根据随机顺序）
        # Use dynamic navigation to start the first question type (based on random order)
        new_state, *view_outputs = start_first_section(current_state)
        return (new_state, *view_outputs, success_update)

    @staticmethod
    def start_second_assessment(current_state):
        """开始第二轮评估"""
        # 保存第一轮的结果（包含题目和答案）
        if 'round1_results' not in current_state:
            current_state['round1_results'] = {
                # 保存第一轮的题目
                'open_ended_tasks': current_state.get('open_ended_tasks', []),
                'constrained_tasks': current_state.get('constrained_tasks', []),
                'imit_tasks': current_state.get('imit_tasks', []),
                # 保存第一轮的答案
                'open_ended_answers': current_state.get('open_ended_answers', {}),
                'constrained_answers': current_state.get('constrained_answers', {}),
                'imit_answers': current_state.get('imit_answers', {}),
                # 保存第一轮的时间戳
                'open_ended_timestamps': current_state.get('open_ended_timestamps', {}),
                'constrained_timestamps': current_state.get('constrained_timestamps', {}),
                'imit_timestamps': current_state.get('imit_timestamps', {}),
                # 保存第一轮的时间戳序列 (按作答顺序)
                'answer_sequence_timestamps': current_state.get('answer_sequence_timestamps', []),
                # 保存第一轮的题型顺序
                'question_type_order': current_state.get('question_type_order', []),
                'final_report': current_state.get('final_report', '')
            }

        # 重新初始化状态但保留用户ID和轮次信息
        user_id = current_state['user_id']
        round1_results = current_state['round1_results']

        # 获取新的初始状态（会重新随机抽取题目和题型顺序）
        new_state = get_initial_state()
        new_state['user_id'] = user_id
        new_state['assessment_round'] = 2  # 标记为第二轮
        new_state['round1_results'] = round1_results

        # 使用动态导航开始第一个题型（根据新的随机顺序）
        # Use dynamic navigation to start the first question type (based on new random order)
        new_state, *view_outputs = start_first_section(new_state)
        return (new_state, *view_outputs, gr.update(visible=False))

    @staticmethod
    def handle_login_button(email, state, btn_text):
        """统一处理登录和第二轮开始的按钮点击"""
        if "Login and Start" in btn_text:
            return EventHandlers.login_and_start(email, state)
        else:  # Start The Second Assessment
            return EventHandlers.start_second_assessment(state)

    @staticmethod
    def return_to_welcome_after_round1(current_state):
        """第一轮完成后返回欢迎页面"""
        # 保存第一轮结果（包含题目、答案和时间戳）
        current_state['round1_results'] = {
            # 保存第一轮的题目
            'open_ended_tasks': current_state.get('open_ended_tasks', []),
            'constrained_tasks': current_state.get('constrained_tasks', []),
            'imit_tasks': current_state.get('imit_tasks', []),
            # 保存第一轮的答案
            'open_ended_answers': current_state.get('open_ended_answers', {}),
            'constrained_answers': current_state.get('constrained_answers', {}),
            'imit_answers': current_state.get('imit_answers', {}),
            # 保存第一轮的时间戳
            'open_ended_timestamps': current_state.get('open_ended_timestamps', {}),
            'constrained_timestamps': current_state.get('constrained_timestamps', {}),
            'imit_timestamps': current_state.get('imit_timestamps', {}),
            # 保存第一轮的时间戳序列 (按作答顺序)
            'answer_sequence_timestamps': current_state.get('answer_sequence_timestamps', []),
            'final_report': current_state.get('final_report', '')
        }
        
        return (current_state, *views.update_views(views.intro_view))

    @staticmethod
    def show_final_complete(current_state):
        """显示最终完成页面"""
        current_state['exam_phase'] = 'final_complete'
        return (current_state, *views.update_views(views.final_complete_view))

    @staticmethod
    def handle_return_button(state):
        """处理返回按钮点击，根据轮次决定下一步"""
        if state.get('assessment_round', 1) == 1:
            # 第一轮完成，准备第二轮
            return EventHandlers.return_to_welcome_after_round1(state)
        else:
            # 第二轮完成，显示最终页面
            return EventHandlers.show_final_complete(state)

    @staticmethod
    def update_intro_for_round2():
        """更新intro view为第二轮的内容"""
        intro_text = gr.update(value="""
        ## Welcome to the Creative Text-to-Image AI Skills Assessment!
        
        This assessment evaluates your proficiency in generating images with 
        text-to-image AI models through three distinct challenge types:

        - Open-Ended Creation: Transform creative briefs into effective prompts
        - Constrained Creation: Control AI output under multiple technical requirements  
        - Imitation: Reverse-engineer target images into descriptive prompts

        Each section tests different skills—from creative interpretation to logical 
        precision to visual analysis. Your responses will help us understand how 
        people translate intent into executable instructions for AI.

        You may use translation tools but not AI writing assistants.
        
        Good luck!
        """)
        email_hidden = gr.update(visible=False)
        button_text = gr.update(value="Start The Second Assessment")
        return intro_text, email_hidden, button_text

    @staticmethod
    def update_button_text_after_processing(state):
        """处理完成后更新按钮文本"""
        if state.get('assessment_round', 1) == 2:
            return gr.update(value="The Assessment Is Over. Thank you!")
        else:
            return gr.update(value="Return to Welcome Page")

    @staticmethod
    def process_results_wrapper(state):
        """处理结果生成函数的包装器"""
        # 调用原始的处理函数
        for result in process_and_generate_results(state):
            # 保存最终报告到状态中
            if len(result) > 3:
                state['final_report'] = result[-1]
            yield result