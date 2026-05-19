import base64
import json
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    from ..configs.api_runtime import apply_api_spec_to_config, get_model_candidates
except ImportError:
    from configs.api_runtime import apply_api_spec_to_config, get_model_candidates  # type: ignore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = apply_api_spec_to_config(cfg, config_path=config_path)

    required_paths = ["generated_root", "checklist_dir", "check_output_subdir", "template_json_path"]
    for key in required_paths:
        if key not in cfg.get("paths", {}):
            raise ValueError(f"Missing paths.{key} in config")

    if not cfg.get("api", {}).get("api_key"):
        raise ValueError("Missing api.api_key in config")

    return cfg


def resolve_config_path(config_path: Path, configured_path: str) -> Path:
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def load_templates_from_json(template_json_path: Path) -> Tuple[str, str]:
    if not template_json_path.exists():
        raise FileNotFoundError(f"Template JSON not found: {template_json_path}")

    with open(template_json_path, "r", encoding="utf-8") as f:
        template_json = json.load(f)

    def parse_template_field(field_name: str) -> str:
        value = template_json.get(field_name)
        if isinstance(value, str):
            text = value
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            text = "\n".join(value)
        else:
            raise ValueError(
                f"Invalid or missing template field '{field_name}' in {template_json_path}. "
                "Expected string or list of strings."
            )

        if "{checklist}" not in text:
            raise ValueError(f"Template '{field_name}' must contain '{{checklist}}': {template_json_path}")
        return text

    prompt_check_template = parse_template_field("prompt_check_template")
    image_check_template = parse_template_field("image_check_template")

    if "{prompt}" not in prompt_check_template:
        raise ValueError(f"Template 'prompt_check_template' must contain '{{prompt}}': {template_json_path}")

    return prompt_check_template, image_check_template


def sort_names_with_preferred_order(names: List[str], preferred_order: List[str]) -> List[str]:
    rank = {name: idx for idx, name in enumerate(preferred_order)}
    return sorted(names, key=lambda x: (rank.get(x, len(rank)), x))


def extract_suffix_number(name: str) -> int:
    match = re.search(r"_(\d+)$", name)
    if match:
        return int(match.group(1))
    return 0


def discover_jobs(cfg: dict) -> Tuple[List[dict], Path, Path]:
    generated_root = Path(cfg["paths"]["generated_root"]).resolve()
    tti_root = generated_root / "2_tti"
    check_root = generated_root / cfg["paths"]["check_output_subdir"]

    output_types = cfg.get("processing", {}).get("output_types", ["co", "im", "oe"])
    t2i_order = cfg.get("processing", {}).get("t2i_model_order", [])
    llm_order = cfg.get("processing", {}).get("llm_model_order", [])

    all_jobs: List[dict] = []

    for output_prefix in output_types:
        output_tti_dir = tti_root / f"{output_prefix}_output_tti"
        if not output_tti_dir.exists():
            print(f"Warning: output directory not found: {output_tti_dir}")
            continue

        t2i_names = [d.name for d in output_tti_dir.iterdir() if d.is_dir()]
        for t2i_name in sort_names_with_preferred_order(t2i_names, t2i_order):
            t2i_dir = output_tti_dir / t2i_name

            llm_names = [d.name for d in t2i_dir.iterdir() if d.is_dir()]
            for llm_name in sort_names_with_preferred_order(llm_names, llm_order):
                llm_dir = t2i_dir / llm_name

                task_dirs = [
                    d
                    for d in llm_dir.iterdir()
                    if d.is_dir() and re.match(rf"^{output_prefix}_\d+$", d.name)
                ]
                task_dirs.sort(key=lambda p: extract_suffix_number(p.name))

                for task_dir in task_dirs:
                    task_id = task_dir.name
                    json_path = (
                        check_root
                        / f"{output_prefix}_output_check"
                        / t2i_name
                        / llm_name
                        / f"{task_id}.json"
                    )
                    all_jobs.append(
                        {
                            "output_prefix": output_prefix,
                            "model_t2i": t2i_name,
                            "model_llm": llm_name,
                            "task_id": task_id,
                            "task_dir": task_dir,
                            "json_path": json_path,
                        }
                    )

    return all_jobs, generated_root, check_root


def load_checklist_map(checklist_file: Path) -> Dict[str, dict]:
    if not checklist_file.exists():
        raise FileNotFoundError(f"Checklist file not found: {checklist_file}")

    with open(checklist_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    task_map: Dict[str, dict] = {}
    duplicates: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if not task_id:
            continue
        task_id_str = str(task_id).strip()
        if not task_id_str:
            continue
        if task_id_str in task_map:
            duplicates.append(task_id_str)
            continue
        task_map[task_id_str] = item
    if duplicates:
        uniq = sorted(set(duplicates))
        dup_preview = ", ".join(uniq[:10])
        if len(uniq) > 10:
            dup_preview += ", ..."
        raise ValueError(f"Duplicate task_id found in checklist {checklist_file}: {dup_preview}")
    return task_map


def normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def resolve_prompt_source_file(generated_root: Path, output_prefix: str, model_llm: str) -> Path:
    llm_output_dir = generated_root / "1_llm" / f"{output_prefix}_output_llm"
    expected = llm_output_dir / f"{output_prefix}_prompts_{model_llm}.json"
    if expected.exists():
        return expected

    if not llm_output_dir.exists():
        raise FileNotFoundError(f"LLM output directory not found: {llm_output_dir}")

    target = normalize_model_name(model_llm)
    for candidate in llm_output_dir.glob(f"{output_prefix}_prompts_*.json"):
        suffix = candidate.stem.replace(f"{output_prefix}_prompts_", "", 1)
        if normalize_model_name(suffix) == target:
            return candidate

    raise FileNotFoundError(
        f"Prompt source file not found for prefix={output_prefix}, model_llm={model_llm} in {llm_output_dir}"
    )


def load_prompt_source_map(prompt_source_file: Path) -> Dict[str, dict]:
    with open(prompt_source_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, list):
        prompt_map: Dict[str, dict] = {}
        for item in raw:
            if isinstance(item, dict) and item.get("task_id"):
                prompt_map[item["task_id"]] = item
        return prompt_map

    raise ValueError(f"Unsupported prompt source format: {prompt_source_file}")


def get_generated_prompt_for_task(
    generated_root: Path,
    output_prefix: str,
    model_llm: str,
    task_id: str,
    prompt_source_cache: Dict[Tuple[str, str], Dict[str, dict]],
) -> str:
    cache_key = (output_prefix, model_llm)
    if cache_key not in prompt_source_cache:
        source_file = resolve_prompt_source_file(generated_root, output_prefix, model_llm)
        prompt_source_cache[cache_key] = load_prompt_source_map(source_file)

    task_entry = prompt_source_cache[cache_key].get(task_id)
    if not isinstance(task_entry, dict):
        raise KeyError(f"task_id '{task_id}' not found in prompt source for {output_prefix}/{model_llm}")

    generated_prompt = task_entry.get("generated_prompt")
    if not isinstance(generated_prompt, str) or not generated_prompt.strip():
        raise ValueError(f"Missing generated_prompt for task_id '{task_id}'")

    return generated_prompt.strip()


def get_images_in_task_dir(task_dir: Path, task_id: str) -> List[Path]:
    image_candidates = []

    for path in task_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        match = re.match(rf"^{re.escape(task_id)}_(\d+)\.(png|jpg|jpeg)$", path.name, re.IGNORECASE)
        if not match:
            continue
        image_candidates.append((int(match.group(1)), path))

    image_candidates.sort(key=lambda x: x[0])
    return [p for _, p in image_candidates]


def load_and_compress_image(image_path: Path, compression_cfg: dict) -> Tuple[str, int, str]:
    from PIL import Image

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    target_kb = int(compression_cfg.get("target_kb", 450))
    tolerance_kb = int(compression_cfg.get("tolerance_kb", 50))
    min_quality = int(compression_cfg.get("min_quality", 40))
    max_quality = int(compression_cfg.get("max_quality", 95))

    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")

        target_bytes = target_kb * 1024
        tolerance_bytes = tolerance_kb * 1024

        low, high = min_quality, max_quality
        best_bytes = None
        best_size = None

        while low <= high:
            q = (low + high) // 2
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=q, optimize=True)
            data = buffer.getvalue()
            size = len(data)

            best_bytes = data
            best_size = size

            if abs(size - target_bytes) <= tolerance_bytes:
                break

            if size > target_bytes:
                high = q - 1
            else:
                low = q + 1

        if best_bytes is None or best_size is None:
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85, optimize=True)
            best_bytes = buffer.getvalue()
            best_size = len(best_bytes)

    base64_str = base64.b64encode(best_bytes).decode("ascii")
    mime_type = "image/jpeg"
    return base64_str, best_size, mime_type


def call_llm(
    prompt: str,
    cfg: dict,
    image_b64: Optional[str] = None,
    mime_type: Optional[str] = None,
    compressed_size: Optional[int] = None,
    model_name: Optional[str] = None,
) -> str:
    api_cfg = cfg["api"]
    request_cfg = cfg.get("llm_request", {})
    retry_cfg = cfg["retry"]

    url = f"{api_cfg['base_url']}{api_cfg['endpoint']}"
    if image_b64 is not None and compressed_size is not None:
        print(f"      Compressed image size: {compressed_size / 1024:.1f} KB")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_cfg['api_key']}",
    }

    content = [{"type": "text", "text": prompt}]
    if image_b64 is not None:
        if not mime_type:
            raise ValueError("mime_type is required when image_b64 is provided")
        data_uri = f"data:{mime_type};base64,{image_b64}"
        content.append({"type": "image_url", "image_url": {"url": data_uri}})

    payload = {
        "model": model_name or api_cfg["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": request_cfg.get("temperature", 0.0),
        "tools": request_cfg.get("tools", []),
        "tool_choice": request_cfg.get("tool_choice", {"type": "none"}),
    }
    top_p = request_cfg.get("top_p", 0.7)
    if top_p is not None:
        payload["top_p"] = top_p
    max_tokens = request_cfg.get("max_tokens", 4096)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = requests.post(url, headers=headers, json=payload, timeout=retry_cfg["timeout_seconds"])
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(str(item.get("text", "")))
        merged = "\n".join(text_chunks).strip()
        if merged:
            return merged
    raise ValueError(f"Invalid message content: {content!r}")


def call_llm_with_retries(
    prompt: str,
    cfg: dict,
    image_b64: Optional[str] = None,
    mime_type: Optional[str] = None,
    compressed_size: Optional[int] = None,
) -> str:
    retry_cfg = cfg["retry"]
    max_attempts = int(retry_cfg.get("max_attempts", 5))
    timeout_backoffs = retry_cfg.get("timeout_backoffs", [3, 6, 9, 12, 15])
    general_retry_delay = int(retry_cfg.get("general_retry_delay", 2))

    last_error = None
    model_candidates = get_model_candidates(cfg.get("api", {}))
    if not model_candidates:
        raise RuntimeError("No configured model candidates available for objective evaluation.")

    for model_name in model_candidates:
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"      Model {model_name} attempt {attempt}/{max_attempts}...")
                return call_llm(
                    prompt=prompt,
                    cfg=cfg,
                    image_b64=image_b64,
                    mime_type=mime_type,
                    compressed_size=compressed_size,
                    model_name=model_name,
                )
            except requests.exceptions.Timeout as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                delay_idx = min(attempt - 1, len(timeout_backoffs) - 1)
                delay = timeout_backoffs[delay_idx]
                print(f"      Timeout error. Retrying in {delay} seconds...")
                time.sleep(delay)
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                detail = ""
                if exc.response is not None:
                    try:
                        detail = exc.response.text.lower()
                    except Exception:
                        detail = ""
                reason = str(exc).lower()
                unavailable = (
                    exc.response is not None
                    and exc.response.status_code in {400, 404, 429}
                    and (
                        "model_not_found" in detail
                        or "does not exist" in detail
                        or "model_not_found" in reason
                        or exc.response.status_code == 429
                    )
                )
                if unavailable:
                    print(f"      Model unavailable: {model_name}. Trying next candidate...")
                    break
                if attempt == max_attempts:
                    break
                print(f"      Request failed: {exc}. Retrying in {general_retry_delay} seconds...")
                time.sleep(general_retry_delay)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                print(f"      Request failed: {exc}. Retrying in {general_retry_delay} seconds...")
                time.sleep(general_retry_delay)
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                print(f"      Unknown error: {exc}. Retrying in {general_retry_delay} seconds...")
                time.sleep(general_retry_delay)

    raise RuntimeError(f"Reached maximum retries ({max_attempts}). Last error: {last_error}")


def parse_llm_response(response: str, expected_checkpoints: List[str]) -> dict:
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        result = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON response: {exc}. Raw response: {response}") from exc

    if not isinstance(result, dict):
        raise ValueError(f"Response is not a dictionary: {type(result)}")

    # Fast path: exact keys already match expected checkpoints.
    if set(result.keys()) != set(expected_checkpoints):
        result = _normalize_checkpoint_keys(result, expected_checkpoints)

    missing = set(expected_checkpoints) - set(result.keys())
    if missing:
        raise ValueError(
            f"Missing checkpoints in response: {missing}. "
            f"Got keys: {list(result.keys())}"
        )

    for key, value in result.items():
        if value not in [0, 1]:
            raise ValueError(f"Invalid value for checkpoint '{key}': {value} (must be 0 or 1)")

    return result


def _normalize_text_for_match(text: str) -> str:
    text = text.strip()
    # Remove leading list index like: "1. ...", "2) ...", "3 - ..."
    text = re.sub(r"^\s*\d+\s*[\.\):\-]\s*", "", text)
    # Collapse spaces for robust exact-text matching.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_checkpoint_keys(result: dict, expected_checkpoints: List[str]) -> dict:
    expected_set = set(expected_checkpoints)
    if set(result.keys()) == expected_set:
        return result

    normalized_expected = {_normalize_text_for_match(k): k for k in expected_checkpoints}
    converted: Dict[str, int] = {}
    used_expected = set()

    # Strategy 1: text matching after stripping numeric prefixes and whitespace differences.
    for key, value in result.items():
        matched = normalized_expected.get(_normalize_text_for_match(str(key)))
        if matched is None or matched in used_expected:
            continue
        converted[matched] = value
        used_expected.add(matched)

    if set(converted.keys()) == expected_set:
        return converted

    # Strategy 2: numeric index keys ("1", "2", ...) or prefixed keys ("1. xxx") map by order.
    converted_by_idx: Dict[str, int] = {}
    for key, value in result.items():
        key_str = str(key).strip()
        m = re.match(r"^(\d+)\s*[\.\):\-]?\s*.*$", key_str)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if idx < 0 or idx >= len(expected_checkpoints):
            continue
        converted_by_idx[expected_checkpoints[idx]] = value

    if set(converted_by_idx.keys()) == expected_set:
        return converted_by_idx

    return result


def parse_llm_response_with_retries(
    response: str,
    expected_checkpoints: List[str],
    prompt: str,
    cfg: dict,
    image_b64: Optional[str] = None,
    mime_type: Optional[str] = None,
    compressed_size: Optional[int] = None,
) -> dict:
    retry_cfg = cfg["retry"]
    max_attempts = int(retry_cfg.get("max_attempts", 5))
    general_retry_delay = int(retry_cfg.get("general_retry_delay", 2))
    last_error = None

    try:
        return parse_llm_response(response, expected_checkpoints)
    except Exception as exc:
        print(f"      Parse failed: {exc}")
        last_error = exc

    for attempt in range(2, max_attempts + 1):
        try:
            print(f"      Re-calling LLM and parsing (attempt {attempt}/{max_attempts})...")
            new_response = call_llm_with_retries(
                prompt=prompt,
                cfg=cfg,
                image_b64=image_b64,
                mime_type=mime_type,
                compressed_size=compressed_size,
            )
            return parse_llm_response(new_response, expected_checkpoints)
        except Exception as exc:
            print(f"      Parse failed: {exc}")
            last_error = exc
            if attempt < max_attempts:
                time.sleep(general_retry_delay)

    raise RuntimeError(f"Parsing failed after {max_attempts} attempts. Last error: {last_error}")


def is_valid_checkpoint_result(value) -> bool:
    if not isinstance(value, dict) or len(value) == 0:
        return False
    return all(v in [0, 1] for v in value.values())


def has_valid_prompt_result(data: dict, expected_checkpoints: List[str]) -> bool:
    prompt_result = data.get("prompt_checkpoint")
    if not is_valid_checkpoint_result(prompt_result):
        return False
    if expected_checkpoints and set(prompt_result.keys()) != set(expected_checkpoints):
        return False
    return True


def load_or_init_result_json(json_path: Path, task_id: str, model_t2i: str, model_llm: str) -> dict:
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("task_id", task_id)
    data.setdefault("model_t2i", model_t2i)
    data.setdefault("model_llm", model_llm)

    if not isinstance(data.get("results"), list):
        data["results"] = []

    return data


def has_valid_result_for_image(results: List[dict], relative_image_path: str) -> bool:
    for item in reversed(results):
        if item.get("image") == relative_image_path:
            return is_valid_checkpoint_result(item.get("image_checkpoint"))
    return False


def upsert_result(results: List[dict], relative_image_path: str, checkpoint_value):
    results[:] = [item for item in results if item.get("image") != relative_image_path]
    results.append({"image": relative_image_path, "image_checkpoint": checkpoint_value})


def save_json(json_path: Path, data: dict):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def log_error_to_file(error_log_path: Path, error_message: str):
    try:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {error_message}\n")
    except Exception as exc:
        print(f"      Warning: failed to write error log: {exc}")


def build_check_prompt(prompt_text: str, checkpoints: List[str], prompt_template: str) -> str:
    checklist_text = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(checkpoints)])
    return prompt_template.format(prompt=prompt_text, checklist=checklist_text)
