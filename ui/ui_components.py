# ui_components.py
import gradio as gr
import view_functions as views

def create_ui_components():
    """创建所有UI组件"""
    components = {}
    
    # --- 标题栏 ---
    
    # --- 欢迎界面 ---
    with gr.Column(visible=True) as intro_view:
        intro_markdown = gr.Markdown("""
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
        user_email = gr.Textbox(label="Enter Your Email to Begin", placeholder="your.email@example.com", interactive=True)
        login_btn = gr.Button("Login and Start The First Assessment", variant="primary")
        login_error_msg = gr.Markdown(visible=False)

    # --- Open-Ended界面 ---
    with gr.Column(visible=False) as open_ended_intro_view:
        gr.Markdown("""
        ##  Open-Ended Creation

        This section evaluates your ability to transform real-world creative requests into effective prompts.

        You will receive natural, conversational project briefs—just like requests from clients, editors, or teammates. Each scenario includes context, audience, and implicit expectations, but no rigid checklists.

        Your challenge: decode the intent, balance competing priorities, and craft a prompt that captures the vision.

        This tests your creative interpretation and professional communication skills.
        """)
        start_open_ended_btn = gr.Button("Start Open-Ended Section", variant="primary")

    with gr.Column(visible=False) as open_ended_answering_view:
        open_ended_progress = gr.Markdown()
        open_ended_question = gr.Markdown()
        open_ended_prompt = gr.Textbox(label="Enter your prompt here", lines=5, interactive=True)
        with gr.Row():
            open_ended_prev_btn = gr.Button("Previous", visible=False)
            open_ended_next_btn = gr.Button("Next", visible=False)
            open_ended_complete_btn = gr.Button("Complete Open-Ended Section", variant="primary", visible=False)
    
    with gr.Column(visible=False) as open_ended_complete_view:
        gr.Markdown("## ✅ Open-Ended Section Complete!\nTransitioning to the Next Part...")

    # --- Constrained界面 ---
    with gr.Column(visible=False) as constrained_intro_view:
        gr.Markdown("""
        ##  Constrained Creation

        This section evaluates your ability to precisely control AI output under multiple constraints.

        Each task presents explicit requirements: color restrictions, spatial layouts, exact quantities, and logical bindings. Success requires accurately encoding all constraints without conflict.

        Your challenge: orchestrate competing technical demands into a single, compliant prompt.

        This tests your logical reasoning and systematic control skills.
        """)
        start_constrained_btn = gr.Button("Start Constrained Section", variant="primary")

    with gr.Column(visible=False) as constrained_answering_view:
        constrained_progress = gr.Markdown()
        constrained_question = gr.Markdown()
        constrained_prompt = gr.Textbox(label="Enter your prompt here", lines=5, interactive=True)
        with gr.Row():
            constrained_prev_btn = gr.Button("Previous", visible=False)
            constrained_next_btn = gr.Button("Next", visible=False)
            constrained_complete_btn = gr.Button("Complete Constrained Section", variant="primary", visible=False)

    # --- Imitation界面 ---
    with gr.Column(visible=False) as imit_intro_view:
        gr.Markdown("""
        ##  Imitation and Reproduction

        This section evaluates your ability to reverse-engineer images into descriptive prompts.

        You will be shown a target image. Your task is to deconstruct its visual components—subject, composition, lighting, style, and technical details—and reconstruct them into a prompt that reproduces the image as closely as possible.

        Your challenge: translate what you see into precise, structured language.

        This tests your visual analysis and technical vocabulary skills.
        """)
        start_imit_btn = gr.Button("Start Imitation Section", variant="primary")

    with gr.Column(visible=False) as imit_answering_view:
        imit_progress = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=1):
                imit_target_image = gr.Image(label="Target Image", interactive=False, height=512)
            with gr.Column(scale=2):
                imit_task_info = gr.Markdown()
                imit_prompt = gr.Textbox(label="Enter your prompt here", lines=10, interactive=True)
        with gr.Row():
            imit_prev_btn = gr.Button("Previous", visible=False)
            imit_next_btn = gr.Button("Next", visible=False)
            imit_complete_btn = gr.Button("Complete Imitation Section", variant="primary", visible=False)

    # --- 状态显示界面 ---
    with gr.Column(visible=False) as processing_view:
        gr.Markdown("## Processing Your Assessment...\nThis may take a few moments.")
        processing_status = gr.Markdown("Starting...")

    with gr.Column(visible=False) as error_view:
        gr.Markdown("## ❌ Application Error\nCould not load materials. Please contact an admin.")
        error_restart_btn = gr.Button("Try Again")

    with gr.Column(visible=False) as results_view:
        results_report = gr.Markdown()
        return_to_start_btn = gr.Button("Return to Welcome Page", variant="primary")

    with gr.Column(visible=False) as final_complete_view:
        gr.Markdown("""
        ## Assessment Complete
        
        We will send the result report to your email in approximately 10 minutes.
        
        Thank you for your participation!
        """)

    # 设置视图组件
    views.set_views({
        'intro_view': intro_view,
        'open_ended_intro_view': open_ended_intro_view, 'open_ended_answering_view': open_ended_answering_view,
        'open_ended_complete_view': open_ended_complete_view, 'constrained_intro_view': constrained_intro_view,
        'constrained_answering_view': constrained_answering_view, 'imit_intro_view': imit_intro_view,
        'imit_answering_view': imit_answering_view, 'processing_view': processing_view,
        'error_view': error_view, 'results_view': results_view, 'final_complete_view': final_complete_view
    })

    # 返回组件字典
    components.update({
        'intro_markdown': intro_markdown,
        'user_email': user_email,
        'login_btn': login_btn,
        'login_error_msg': login_error_msg,
        'error_restart_btn': error_restart_btn,
        'return_to_start_btn': return_to_start_btn,

        # Open-ended components
        'start_open_ended_btn': start_open_ended_btn,
        'open_ended_progress': open_ended_progress,
        'open_ended_question': open_ended_question,
        'open_ended_prompt': open_ended_prompt,
        'open_ended_prev_btn': open_ended_prev_btn,
        'open_ended_next_btn': open_ended_next_btn,
        'open_ended_complete_btn': open_ended_complete_btn,
        
        # Constrained components
        'start_constrained_btn': start_constrained_btn,
        'constrained_progress': constrained_progress,
        'constrained_question': constrained_question,
        'constrained_prompt': constrained_prompt,
        'constrained_prev_btn': constrained_prev_btn,
        'constrained_next_btn': constrained_next_btn,
        'constrained_complete_btn': constrained_complete_btn,
        
        # Imitation components
        'start_imit_btn': start_imit_btn,
        'imit_progress': imit_progress,
        'imit_target_image': imit_target_image,
        'imit_task_info': imit_task_info,
        'imit_prompt': imit_prompt,
        'imit_prev_btn': imit_prev_btn,
        'imit_next_btn': imit_next_btn,
        'imit_complete_btn': imit_complete_btn,
        
        # Status components
        'processing_status': processing_status,
        'results_report': results_report
    })
    
    return components