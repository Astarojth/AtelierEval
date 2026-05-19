import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


SUPPORTED_TASK_TYPES = ("co", "oe", "im")


def resolve_path(config_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def replace_placeholder(template: str, token: str, value: str) -> str:
    rendered = template
    rendered = rendered.replace(f"{{{token}}}", value)
    rendered = rendered.replace(f"\\placeholder{{{token}}}", value)
    return rendered


def image_to_data_url(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"image file not found: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


class ChatCompletionsClient:
    def __init__(self, api_cfg: Dict[str, Any]):
        self.base_url = api_cfg["base_url"]
        self.api_key = api_cfg["api_key"]
        self.model = api_cfg["model"]
        self.timeout_seconds = int(api_cfg.get("timeout_seconds", 180))
        self.max_retries = int(api_cfg.get("max_retries", 3))
        self.backoff_seconds = int(api_cfg.get("backoff_seconds", 2))
        self.temperature = float(api_cfg.get("temperature", 0.0))
        self.top_p = api_cfg.get("top_p", 0.7)
        if self.top_p is not None:
            self.top_p = float(self.top_p)
        self.max_tokens = api_cfg.get("max_tokens", 4096)

        if not self.api_key:
            raise RuntimeError("api.api_key is empty in config.")

    def _extract_text(self, response_json: Dict[str, Any]) -> str:
        try:
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "unexpected chat completions response:\n"
                + json.dumps(response_json, ensure_ascii=False, indent=2)
            ) from exc

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
            text_out = "\n".join(parts).strip()
            if text_out:
                return text_out
        raise RuntimeError("assistant content is empty or unsupported format.")

    def generate_text(self, messages: List[Dict[str, Any]]) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()
                return self._extract_text(resp.json())
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    wait_sec = attempt * self.backoff_seconds
                    print(
                        f"[WARN] API failed ({attempt}/{self.max_retries}): {type(exc).__name__}: {exc}"
                    )
                    print(f"[WARN] retry after {wait_sec}s...")
                    time.sleep(wait_sec)

        raise RuntimeError(f"API call failed after retries: {last_error}")


class DryRunClient:
    def __init__(self) -> None:
        self.model = "dry-run-model"

    def generate_text(self, messages: List[Dict[str, Any]]) -> str:
        user_msg = messages[-1].get("content", "")
        if isinstance(user_msg, str):
            seed = user_msg
        elif isinstance(user_msg, list):
            text_parts = []
            for block in user_msg:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            seed = " ".join(text_parts)
        else:
            seed = ""
        seed = " ".join(seed.split())
        return f"[DRY_RUN] {seed[:160]}".strip()


def build_messages(
    *,
    task_type: str,
    task_item: Dict[str, Any],
    prompt_template: str,
    image_root: Path,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    if task_type in ("co", "oe"):
        task_text = str(task_item.get("task", "")).strip()
        if not task_text:
            raise ValueError(f"{task_type} task missing `task` field: {task_item.get('task_id')}")
        user_prompt = replace_placeholder(prompt_template, "TASK", task_text)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    if task_type == "im":
        origin_image = str(task_item.get("origin_image", "")).strip()
        if not origin_image:
            raise ValueError(f"im task missing `origin_image`: {task_item.get('task_id')}")
        image_path = (image_root / origin_image).resolve()
        data_url = image_to_data_url(image_path)
        user_prompt = replace_placeholder(prompt_template, "IMAGE", "Please refer to the attached image.")
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

    raise ValueError(f"unsupported task type: {task_type}")


def load_existing_map(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for task_id, item in data.items():
            if isinstance(item, dict):
                merged = dict(item)
                merged.setdefault("task_id", str(task_id))
                out[str(task_id)] = merged
        return out
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                task_id = str(item.get("task_id", "")).strip()
                if task_id:
                    out[task_id] = item
    return out


def process_task_type(
    *,
    task_type: str,
    tasks_path: Path,
    output_path: Path,
    template: str,
    image_root: Path,
    client: Any,
    system_prompt: str,
    output_field: str,
    resume: bool,
    max_items: int | None,
) -> None:
    raw_tasks = load_json(tasks_path)
    tasks: List[Dict[str, Any]] = []
    if isinstance(raw_tasks, list):
        for idx, item in enumerate(raw_tasks, start=1):
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id", f"{task_type}_{idx:03d}")).strip()
            merged = dict(item)
            merged["task_id"] = task_id
            tasks.append(merged)
    elif isinstance(raw_tasks, dict):
        for task_id, item in raw_tasks.items():
            if not isinstance(item, dict):
                continue
            merged = dict(item)
            merged["task_id"] = str(merged.get("task_id", str(task_id))).strip() or str(task_id)
            tasks.append(merged)
    else:
        raise RuntimeError(f"task file must be a list or object: {tasks_path}")

    if max_items is not None:
        tasks = tasks[:max_items]

    existing_map = load_existing_map(output_path) if resume else {}
    results_map: Dict[str, Dict[str, Any]] = {}

    total = len(tasks)
    for idx, task_item in enumerate(tasks, start=1):
        if not isinstance(task_item, dict):
            continue

        task_id = str(task_item.get("task_id", f"{task_type}_{idx:03d}")).strip()
        if resume and task_id in existing_map and existing_map[task_id].get(output_field):
            results_map[task_id] = existing_map[task_id]
            print(f"[{task_type}] {idx}/{total} {task_id}: skip (resume)")
            continue

        messages = build_messages(
            task_type=task_type,
            task_item=task_item,
            prompt_template=template,
            image_root=image_root,
            system_prompt=system_prompt,
        )
        generated = client.generate_text(messages)

        merged = dict(task_item)
        merged[output_field] = generated
        merged["llm_meta"] = {
            "model": client.model,
            "task_type": task_type,
        }
        results_map[task_id] = merged
        print(f"[{task_type}] {idx}/{total} {task_id}: done")

    save_json(output_path, results_map)
    print(f"[{task_type}] wrote {len(results_map)} items -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch LLM runner for co/oe/im_120 tasks.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_api.json",
        help="Path to runner config JSON.",
    )
    parser.add_argument(
        "--task-types",
        type=str,
        default="co,oe,im",
        help="Comma-separated task types to run, e.g. co,oe,im",
    )
    parser.add_argument(
        "--max-items-per-type",
        type=int,
        default=None,
        help="Optional debug cap for each task type.",
    )
    parser.add_argument(
        "--disable-resume",
        action="store_true",
        help="Disable resume mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and write mock outputs for quick validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = load_json(config_path)

    prompt_templates_path = resolve_path(config_dir, config["prompt_templates_path"])
    task_files_cfg = config["task_files"]
    output_files_cfg = config["output_files"]
    output_dir = resolve_path(config_dir, config.get("output_dir", "./output"))
    image_root = resolve_path(config_dir, config.get("image_root", "../../../dataset"))
    system_prompt = config.get(
        "system_prompt",
        "You are an expert assistant that writes one high-quality image generation prompt in natural language.",
    )
    output_field = config.get("output_field", "generated_prompt")
    resume = bool(config.get("resume", True)) and not args.disable_resume

    templates = load_json(prompt_templates_path)
    if not isinstance(templates, dict):
        raise RuntimeError(f"prompt template file must be a JSON object: {prompt_templates_path}")

    task_types = [x.strip() for x in args.task_types.split(",") if x.strip()]
    for task_type in task_types:
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"unsupported task type: {task_type}")

    if args.dry_run:
        client = DryRunClient()
        print("[INFO] Running in dry-run mode (no API call).")
    else:
        client = ChatCompletionsClient(config["api"])

    for task_type in task_types:
        if task_type not in templates:
            raise KeyError(f"missing prompt template for task type: {task_type}")
        if task_type not in task_files_cfg:
            raise KeyError(f"missing task file config for task type: {task_type}")
        if task_type not in output_files_cfg:
            raise KeyError(f"missing output file config for task type: {task_type}")

        tasks_path = resolve_path(config_dir, task_files_cfg[task_type])
        output_path = (output_dir / output_files_cfg[task_type]).resolve()
        template = str(templates[task_type])

        print(f"\n=== processing {task_type} ===")
        print(f"tasks:  {tasks_path}")
        print(f"output: {output_path}")

        process_task_type(
            task_type=task_type,
            tasks_path=tasks_path,
            output_path=output_path,
            template=template,
            image_root=image_root,
            client=client,
            system_prompt=system_prompt,
            output_field=output_field,
            resume=resume,
            max_items=args.max_items_per_type,
        )

    print("\nAll selected task types completed.")


if __name__ == "__main__":
    main()
