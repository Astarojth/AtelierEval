from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

try:
    from ..obj_checker_utils import (  # type: ignore
        build_check_prompt,
        call_llm_with_retries,
        get_generated_prompt_for_task,
        get_images_in_task_dir,
        has_valid_prompt_result,
        has_valid_result_for_image,
        load_and_compress_image,
        load_checklist_map,
        load_or_init_result_json,
        log_error_to_file,
        parse_llm_response_with_retries,
        save_json,
        upsert_result,
    )
except ImportError:
    from objective.obj_checker_utils import (  # type: ignore
        build_check_prompt,
        call_llm_with_retries,
        get_generated_prompt_for_task,
        get_images_in_task_dir,
        has_valid_prompt_result,
        has_valid_result_for_image,
        load_and_compress_image,
        load_checklist_map,
        load_or_init_result_json,
        log_error_to_file,
        parse_llm_response_with_retries,
        save_json,
        upsert_result,
    )


class S2PromptObjectiveSkill:
    name = "S2PromptObjectiveSkill"

    def run(self, *, run_prompt_objective_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        return run_prompt_objective_fn()


def process_single_job(
    job: dict,
    generated_root: Path,
    checklist_dir: Path,
    check_root: Path,
    checklist_cache: Dict[str, Dict[str, dict]],
    prompt_source_cache: Dict[Tuple[str, str], Dict[str, dict]],
    cfg: dict,
    prompt_check_template: str,
    image_check_template: str,
    dry_run: bool,
    max_images_per_job: int,
) -> None:
    output_prefix = job["output_prefix"]
    task_id = job["task_id"]
    task_dir = job["task_dir"]
    json_path = job["json_path"]
    model_t2i = job["model_t2i"]
    model_llm = job["model_llm"]

    checklist_file = checklist_dir / f"{output_prefix}_checklist.json"
    if output_prefix not in checklist_cache:
        checklist_cache[output_prefix] = load_checklist_map(checklist_file)

    task_map = checklist_cache[output_prefix]
    task_meta = task_map.get(task_id)
    if not task_meta:
        print(f"  Skip: task_id '{task_id}' not found in {checklist_file.name}")
        return

    image_checkpoints = task_meta.get("image_checkpoint", [])
    prompt_checkpoints = task_meta.get("prompt_checkpoint", [])

    data = load_or_init_result_json(json_path, task_id, model_t2i, model_llm)
    error_log_path = check_root / f"error_{output_prefix}.txt"

    try:
        generated_prompt = get_generated_prompt_for_task(
            generated_root=generated_root,
            output_prefix=output_prefix,
            model_llm=model_llm,
            task_id=task_id,
            prompt_source_cache=prompt_source_cache,
        )
    except Exception as exc:
        fallback_prompt = task_meta.get("task", "")
        generated_prompt = fallback_prompt if isinstance(fallback_prompt, str) else ""
        warning = (
            f"Prompt source unavailable for {task_id} ({model_llm}): {exc}. "
            "Fallback to checklist 'task' text."
        )
        print(f"  Warning: {warning}")
        log_error_to_file(error_log_path, warning)

    data["prompt"] = generated_prompt

    if prompt_checkpoints and not has_valid_prompt_result(data, prompt_checkpoints):
        prompt_check_prompt = build_check_prompt(generated_prompt, prompt_checkpoints, prompt_check_template)
        print("  Processing prompt checkpoints...")
        if dry_run:
            print("      Dry-run mode: skipping API call and file write for prompt checkpoints.")
        else:
            try:
                llm_response = call_llm_with_retries(prompt=prompt_check_prompt, cfg=cfg)
                prompt_check_results = parse_llm_response_with_retries(
                    response=llm_response,
                    expected_checkpoints=prompt_checkpoints,
                    prompt=prompt_check_prompt,
                    cfg=cfg,
                )
                passed = sum(prompt_check_results.values())
                total = len(prompt_check_results)
                print(f"      Prompt check complete: {passed}/{total} ({(passed / total * 100):.1f}%)")
                data["prompt_checkpoint"] = prompt_check_results
                save_json(json_path, data)
            except Exception as exc:
                print(f"      Prompt checkpoint failed after retries: {exc}")
                data["prompt_checkpoint"] = "error"
                save_json(json_path, data)
    elif prompt_checkpoints:
        print("  Skip prompt checkpoints: already complete")

    images = get_images_in_task_dir(task_dir, task_id)
    if not images:
        print(f"  Skip image checkpoints: no images found in {task_dir}")
        return

    relative_images = [str(img.relative_to(generated_root)).replace("\\", "/") for img in images]
    if all(has_valid_result_for_image(data["results"], rel) for rel in relative_images):
        print(f"  Skip: already complete ({len(images)}/{len(images)} images)")
        return

    image_check_prompt = build_check_prompt(generated_prompt, image_checkpoints, image_check_template)

    processed_count = 0
    for image_index, image_path in enumerate(images, start=1):
        relative_image_path = str(image_path.relative_to(generated_root)).replace("\\", "/")

        if has_valid_result_for_image(data["results"], relative_image_path):
            print(f"    [{image_index}/{len(images)}] Skip existing: {image_path.name}")
            continue

        if max_images_per_job > 0 and processed_count >= max_images_per_job:
            print("    Reached max-images-per-job limit for this run.")
            break

        print(f"    [{image_index}/{len(images)}] Processing: {image_path.name}")
        if dry_run:
            print("      Dry-run mode: skipping API call and file write for this image.")
            processed_count += 1
            continue

        try:
            try:
                image_b64, size_bytes, mime_type = load_and_compress_image(image_path, cfg["compression"])
            except Exception as exc:
                error_msg = str(exc)
                if "cannot identify image file" in error_msg.lower():
                    full_error = f"Image decode failure: cannot identify image file '{image_path}'"
                    print(f"      {full_error}")
                    log_error_to_file(error_log_path, full_error)
                    upsert_result(data["results"], relative_image_path, "error_cannot_identify_image")
                    save_json(json_path, data)
                    processed_count += 1
                    continue
                raise

            llm_response = call_llm_with_retries(
                prompt=image_check_prompt,
                cfg=cfg,
                image_b64=image_b64,
                mime_type=mime_type,
                compressed_size=size_bytes,
            )
            check_results = parse_llm_response_with_retries(
                response=llm_response,
                expected_checkpoints=image_checkpoints,
                prompt=image_check_prompt,
                cfg=cfg,
                image_b64=image_b64,
                mime_type=mime_type,
                compressed_size=size_bytes,
            )

            passed = sum(check_results.values())
            total = len(check_results)
            print(f"      Check complete: {passed}/{total} ({(passed / total * 100):.1f}%)")

            upsert_result(data["results"], relative_image_path, check_results)
            save_json(json_path, data)
            processed_count += 1
        except Exception as exc:
            print(f"      Processing failed after retries: {exc}")
            upsert_result(data["results"], relative_image_path, "error")
            save_json(json_path, data)
            processed_count += 1
