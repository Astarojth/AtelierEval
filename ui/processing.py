# processing.py
from utils import hf_utils
import view_functions as views
import gradio as gr
import base64
import io
from PIL import Image
from pathlib import Path


def process_and_generate_results(state):
    """
    处理所有三个部分的所有答案，并生成最终结果。
    Process all answers from all three sections and generate final results.
    """
    state["exam_phase"] = "processing"
    yield state, *views.show_processing(), gr.update(value="Processing your answers..."), ""

    # 获取用户ID和轮次信息
    user_id = state.get("user_id", "Not Provided")
    assessment_round = state.get("assessment_round", 1)
    
    # 如果是第二轮，添加轮次标识
    round_prefix = f"Round {assessment_round} - " if assessment_round == 2 else ""

    # 从 state 中加载所有任务和答案
    open_ended_tasks = state.get("open_ended_tasks", [])
    constrained_tasks = state.get("constrained_tasks", [])
    imit_tasks = state.get("imit_tasks", [])

    open_ended_answers = state.get("open_ended_answers", {})
    constrained_answers = state.get("constrained_answers", {})
    imit_answers = state.get("imit_answers", {})

    report_parts = {"open_ended": "", "constrained": "", "imit": ""}

    try:
        # ---  Process Open-Ended results ---
        report_parts["open_ended"] += f"\n---\n\n## {round_prefix} Open-Ended Creation Questions\n\n"
        for i, task in enumerate(open_ended_tasks):
            try:
                yield state, *views.show_processing(), gr.update(value=f"Generating image for Open-Ended question {i+1}/{len(open_ended_tasks)}..."), ""
                task_id = str(task.get("task_id") or task.get("id"))
                user_prompt = open_ended_answers.get(task_id, "No prompt submitted.")
                
                # 尝试生成图像，如果失败则使用占位符
                try:
                    generated_image_url = hf_utils.generate_image_with_flux(user_prompt)
                except Exception as img_error:
                    print(f"⚠️ 图像生成失败: {img_error}")
                    generated_image_url = "https://placehold.co/512x512/ccc/FFFFFF/png?text=Image+Generation+Failed"

                description = task.get('description', 'No description available.')
                reference_prompt = task.get('ground_truth_prompt') or task.get('gt_prompt') or task.get('prompt', 'N/A')
                report_parts["open_ended"] += f"### Question {i+1}: {description}\n**Your Submitted Prompt:**\n```\n{user_prompt}\n```\n**Reference Prompt:**\n```\n{reference_prompt}\n```\n**Generated Image:**\n![Generated Image]({generated_image_url})\n\n---\n\n"
            except Exception as e:
                print(f"⚠️ 处理 Open-Ended 题目 {i+1} 时出错: {e}")
                continue

        # ---  Process Constrained results ---
        report_parts["constrained"] += f"\n---\n\n## {round_prefix} Constrained Creation Questions\n\n"
        for i, task in enumerate(constrained_tasks):
            try:
                yield state, *views.show_processing(), gr.update(value=f"Generating image for Constrained question {i+1}/{len(constrained_tasks)}..."), ""
                task_id = str(task.get("task_id") or task.get("id"))
                user_prompt = constrained_answers.get(task_id, "No prompt submitted.")
                
                # 尝试生成图像，如果失败则使用占位符
                try:
                    generated_image_url = hf_utils.generate_image_with_flux(user_prompt)
                except Exception as img_error:
                    print(f"⚠️ 图像生成失败: {img_error}")
                    generated_image_url = "https://placehold.co/512x512/ccc/FFFFFF/png?text=Image+Generation+Failed"

                description = task.get('task', 'No description available.')
                reference_prompt = task.get('prompt') or task.get('gt_prompt') or 'N/A'
                report_parts["constrained"] += f"### Question {i+1}: {description}\n**Your Submitted Prompt:**\n```\n{user_prompt}\n```\n**Reference Prompt:**\n```\n{reference_prompt}\n```\n**Generated Image:**\n![Generated Image]({generated_image_url})\n\n---\n\n"
            except Exception as e:
                print(f"⚠️ 处理 Constrained 题目 {i+1} 时出错: {e}")
                continue

        # ---  Process Imitation results ---
        report_parts["imit"] = f"\n---\n\n## {round_prefix} Imitation and Reproduction\n\n"
        
        def pil_to_base64_uri(pil_img):
            if not pil_img or not isinstance(pil_img, Image.Image):
                return "https://placehold.co/512x512/ccc/FFFFFF/png?text=Image+Invalid"
            try:
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{img_str}"
            except Exception as e:
                print(f"⚠️ 图像转换失败: {e}")
                return "https://placehold.co/512x512/ccc/FFFFFF/png?text=Image+Conversion+Failed"

        for i, task in enumerate(imit_tasks):
            try:
                yield state, *views.show_processing(), gr.update(value=f"Generating image for Imitation question {i+1}/{len(imit_tasks)}..."), ""
                task_id = str(task.get("id") or task.get("task_id") or f"im_{i + 1}")
                user_prompt = imit_answers.get(task_id, "No prompt submitted.")
                
                # 尝试生成图像，如果失败则使用占位符
                try:
                    user_generated_image_url = hf_utils.generate_image_with_flux(user_prompt)
                except Exception as img_error:
                    print(f"⚠️ 图像生成失败: {img_error}")
                    user_generated_image_url = "https://placehold.co/512x512/ccc/FFFFFF/png?text=Image+Generation+Failed"
                
                target_image_data = task.get('flux_image_path') or task.get('origin_image_path') or task.get('origin_image')
                target_image_pil = None
                if isinstance(target_image_data, dict) and 'bytes' in target_image_data and target_image_data['bytes']:
                    try:
                        target_image_pil = Image.open(io.BytesIO(target_image_data['bytes']))
                    except Exception as e:
                        print(f"Error reconstructing PIL image for report (task {task.get('id')}): {e}")
                elif isinstance(target_image_data, str) and target_image_data.strip():
                    path = Path(target_image_data)
                    if path.exists():
                        try:
                            with Image.open(path) as img:
                                target_image_pil = img.convert("RGB")
                        except Exception as e:
                            print(f"Error loading target image path for report (task {task.get('id')}): {e}")
                elif isinstance(target_image_data, Image.Image):
                    target_image_pil = target_image_data
                    
                target_image_url = pil_to_base64_uri(target_image_pil)
                golden_prompt = task.get('prompt') or task.get('gt_prompt') or 'N/A'
                report_parts["imit"] += f"### Imitation Task {i+1}\n\n| Target Image | Your Generated Image |\n|:---:|:---:|\n| ![Target Image]({target_image_url}) | ![Your Image]({user_generated_image_url}) |\n\n**Your Submitted Prompt:**\n```\n{user_prompt}\n```\n\n**Reference (Golden) Prompt:**\n```\n{golden_prompt}\n```\n\n---\n\n"
            except Exception as e:
                print(f"⚠️ 处理 Imitation 题目 {i+1} 时出错: {e}")
                continue

        # --- Final Steps ---
        # 如果是第二轮，保存时间戳数据和完整作答结果到HuggingFace数据集
        if assessment_round == 2:
            yield state, *views.show_processing(), gr.update(value="Saving data to HuggingFace..."), ""

            try:
                # 保存按作答顺序的时间戳到 all_timestamps dataset
                round1_results = state.get('round1_results', {})
                round1_timestamp_sequence = round1_results.get('answer_sequence_timestamps', [])
                round2_timestamp_sequence = state.get('answer_sequence_timestamps', [])
                hf_utils.save_answer_sequence_timestamps(user_id, round1_timestamp_sequence, round2_timestamp_sequence)
                print("✅ 时间戳数据保存完成")
            except Exception as save_error:
                print(f"⚠️ 保存时间戳数据失败，但继续处理: {save_error}")

            try:
                # 保存完整的作答结果到 all_result dataset
                round1_results = state.get('round1_results', {})
                hf_utils.save_complete_user_results(user_id, round1_results, state)
                print("✅ 完整作答结果保存完成")
            except Exception as save_error:
                print(f"⚠️ 保存完整作答结果失败，但继续处理: {save_error}")
        
        state["exam_phase"] = "results"

        # 组合最终报告
        if assessment_round == 2:
            # 第二轮，包含两轮的结果
            round1_results = state.get('round1_results', {})
            round1_report = round1_results.get('final_report', '')
            
            # 提取第一轮的简要信息
            round1_summary = ""
            if round1_report:
                lines = round1_report.split('\n')
                for line in lines:
                    if 'Questions Completed:' in line:
                        round1_summary += line + "\n"

            final_report = (
                f"# Final Assessment Results - Complete Report\n\n"
                f"**User ID:** {user_id}\n\n"
                f"## Overall Summary\n"
                f"You have completed both rounds of the assessment.\n\n"
                f"### Round 1 Summary\n"
                f"{round1_summary if round1_summary else 'Round 1 completed.'}\n\n"
                f"### Round 2 Summary\n"
                f"- **Open-Ended Questions Completed:** {len(open_ended_tasks)}\n"
                f"- **Constrained Questions Completed:** {len(constrained_tasks)}\n"
                f"- **Imitation Questions Completed:** {len(imit_tasks)}\n\n"
                f"---\n\n"
                f"# Round 2 Detailed Results\n\n"
                f"{report_parts['open_ended']}"
                f"{report_parts['constrained']}"
                f"{report_parts['imit']}"
            )
        else:
            # 第一轮报告
            final_report = (
                f"# Assessment Results - Round 1\n\n"
                f"**User ID:** {user_id}\n\n"
                f"## Summary\n"
                f"- **Open-Ended Questions Completed:** {len(open_ended_tasks)}\n"
                f"- **Constrained Questions Completed:** {len(constrained_tasks)}\n"
                f"- **Imitation Questions Completed:** {len(imit_tasks)}\n\n"
                f"---\n\n"
                f"{report_parts['open_ended']}"
                f"{report_parts['constrained']}"
                f"{report_parts['imit']}"
            )
        
        # 保存报告到状态
        state['final_report'] = final_report

        yield state, *views.show_results(), "Done!", final_report
        
    except Exception as e:
        print(f"❌ 处理结果时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 即使出错也要显示结果页面，避免卡住
        error_report = f"# Assessment Results\n\n**User ID:** {user_id}\n\n⚠️ 处理过程中遇到错误，但您的答案已记录。\n\n错误信息: {str(e)}"
        state['final_report'] = error_report
        state["exam_phase"] = "results"
        
        yield state, *views.show_results(), "Processing completed with errors", error_report
