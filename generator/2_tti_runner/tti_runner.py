import argparse
import base64
import io
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


DATA_URL_RE = re.compile(r"(data:image/(?:png|jpeg|jpg|webp);base64,[A-Za-z0-9+/=]+)")


def extract_model_name_from_filename(filename: str) -> str:
    name_without_ext = filename.replace(".json", "")
    parts = name_without_ext.split("_", 2)
    if len(parts) >= 3:
        return parts[2]
    return name_without_ext


def resolve_path(config_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def decode_data_url(data_url: str) -> bytes:
    try:
        header, b64_data = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid data URL (missing comma separator between header and payload).") from exc

    if not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError(f"Not an image base64 data URL: {header}")

    return base64.b64decode(b64_data)


def find_data_url_in_obj(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        match = DATA_URL_RE.search(obj)
        if match:
            return match.group(1)
        return None

    if isinstance(obj, dict):
        for value in obj.values():
            found = find_data_url_in_obj(value)
            if found:
                return found
        return None

    if isinstance(obj, list):
        for item in obj:
            found = find_data_url_in_obj(item)
            if found:
                return found
        return None

    return None


class BaseBackend:
    output_model_dir: str

    def generate_image(self, prompt: str) -> Tuple[bytes, Dict[str, Any]]:
        raise NotImplementedError


class APIImageBackend(BaseBackend):
    def __init__(self, cfg: Dict[str, Any]):
        self.base_url = cfg["base_url"]
        self.model = cfg["model"]
        self.image_size = cfg.get("image_size", "1024x1024")
        self.response_format = cfg.get("response_format", "b64_json")
        self.timeout_seconds = int(cfg.get("timeout_seconds", 180))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.backoff_step = int(cfg.get("backoff_step", 3))
        self.extra_params = dict(cfg.get("extra_params", {}))
        self.output_model_dir = cfg.get("output_model_dir", "api_image")

        api_key = cfg.get("api_key")
        if not api_key:
            raise RuntimeError("API key missing: set `config.api_image.api_key` in your config file.")
        self.api_key = api_key

    def _call_once(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": self.image_size,
            "response_format": self.response_format,
        }
        payload.update(self.extra_params)

        resp = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()

    def _extract_image_bytes(self, response_json: Dict[str, Any]) -> bytes:
        try:
            first = response_json["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected response structure from image API:\n"
                + json.dumps(response_json, ensure_ascii=False, indent=2)
            ) from exc

        b64_json = first.get("b64_json")
        url = first.get("url")

        if b64_json:
            return base64.b64decode(b64_json)

        if isinstance(url, str):
            if DATA_URL_RE.search(url):
                return decode_data_url(url)
            if url.startswith("http://") or url.startswith("https://"):
                download = requests.get(url, timeout=self.timeout_seconds)
                download.raise_for_status()
                return download.content

        fallback_data_url = find_data_url_in_obj(response_json)
        if fallback_data_url:
            return decode_data_url(fallback_data_url)

        raise RuntimeError(
            "image API response has neither usable 'b64_json' nor 'url':\n"
            + json.dumps(response_json, ensure_ascii=False, indent=2)
        )

    def generate_image(self, prompt: str) -> Tuple[bytes, Dict[str, Any]]:
        last_exception: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                started = time.time()
                response_json = self._call_once(prompt)
                image_bytes = self._extract_image_bytes(response_json)
                return image_bytes, {
                    "latency_sec": round(time.time() - started, 2),
                    "response_json": response_json,
                }
            except Exception as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    wait_time = attempt * self.backoff_step
                    print(
                        f"[WARN] API attempt {attempt}/{self.max_retries} failed: {type(exc).__name__}: {exc}"
                    )
                    print(f"       Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] API call reached maximum retries ({self.max_retries})")

        if last_exception:
            raise last_exception
        raise RuntimeError("API call failed without a captured exception.")


class DiffusersBackend(BaseBackend):
    def __init__(self, cfg: Dict[str, Any], config_dir: Path):
        torch = __import__("torch")
        diffusers = __import__("diffusers")

        self.torch = torch
        self.output_model_dir = cfg.get("output_model_dir", "diffusers_local")
        requested_device = str(cfg.get("device", "auto"))
        auto_fallback_to_cpu = bool(cfg.get("auto_fallback_to_cpu", True))
        self.height = int(cfg.get("height", 1024))
        self.width = int(cfg.get("width", 1024))
        self.num_inference_steps = int(cfg.get("num_inference_steps", 30))
        self.guidance_scale = float(cfg.get("guidance_scale", 7.5))
        self.seed_mode = str(cfg.get("seed_mode", "random"))
        self.seed = cfg.get("seed")

        dtype_name = str(cfg.get("torch_dtype", "float16"))
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype_name)
        if torch_dtype is None:
            raise ValueError(f"Unsupported torch_dtype: {dtype_name}")

        model_path = resolve_path(config_dir, cfg["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Diffusers model file does not exist: {model_path}")

        cuda_available = bool(torch.cuda.is_available())
        if requested_device == "auto":
            self.device = "cuda" if cuda_available else "cpu"
        else:
            self.device = requested_device

        if self.device.startswith("cuda") and not cuda_available:
            if auto_fallback_to_cpu:
                print("[WARN] Requested device=cuda, but CUDA is unavailable. Falling back to CPU.")
                self.device = "cpu"
                if dtype_name == "float16":
                    print("[WARN] CPU does not support float16 inference. Switching torch_dtype=float32.")
                    torch_dtype = torch.float32
            else:
                raise RuntimeError(
                    "Config requested device=cuda, but torch.cuda.is_available()=False. "
                    "Check your CUDA environment or set config.diffusers.device to cpu/auto."
                )

        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("[INFO] Loading Diffusers model...")
        self.pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=torch_dtype,
        ).to(self.device)
        print("[OK] Diffusers model loaded.")

    def _build_generator(self):
        if self.seed_mode == "none":
            return None, None
        if self.seed_mode == "fixed":
            if self.seed is None:
                raise ValueError("seed must be set when seed_mode=fixed")
            seed = int(self.seed)
        else:
            seed = random.randint(0, 2**32 - 1)

        generator_device = "cuda" if self.device.startswith("cuda") else "cpu"
        generator = self.torch.Generator(device=generator_device).manual_seed(seed)
        return generator, seed

    def generate_image(self, prompt: str) -> Tuple[bytes, Dict[str, Any]]:
        generator, seed = self._build_generator()
        started = time.time()
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            generator=generator,
        ).images[0]
        latency = round(time.time() - started, 2)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue(), {"seed": seed, "latency_sec": latency}


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_backend(config: Dict[str, Any], config_dir: Path) -> BaseBackend:
    mode = config["mode"]
    if mode == "api_image":
        return APIImageBackend(config["api_image"])
    if mode == "diffusers":
        return DiffusersBackend(config["diffusers"], config_dir)
    raise ValueError(f"Unsupported mode: {mode}")


def save_error_report(
    report_dir: Path,
    error_tasks: Dict[str, int],
    model_name: str,
    num_images: int,
) -> None:
    if not error_tasks:
        print(f"\n[OK] All tasks successfully generated {num_images} images.")
        return

    report_path = report_dir / f"error_report_{model_name}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Error Report - Model: {model_name}\n")
        f.write(f"Generated At: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target Images Per Task: {num_images}\n")
        f.write(f"Number of Failed Tasks: {len(error_tasks)}\n")
        f.write("=" * 60 + "\n\n")
        for task_id, success_count in sorted(error_tasks.items()):
            f.write(f'Failed task "{task_id}": {success_count}\n')
    print(f"\n[INFO] Error report saved to: {report_path}")


def generate_images_for_task(
    backend: BaseBackend,
    task_id: str,
    prompt: str,
    output_dir: Path,
    num_images: int,
    resume: bool,
    debug_save_response: bool,
) -> int:
    print(f"\n{'=' * 60}")
    print(f"Processing task: {task_id}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"{'=' * 60}")

    task_output_dir = output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    existing_pngs = list(task_output_dir.glob(f"{task_id}_*.png"))
    if resume and len(existing_pngs) >= num_images:
        print(
            f"[OK] Skipping task {task_id} "
            f"(already has {len(existing_pngs)} images, target is {num_images})."
        )
        return len(existing_pngs)

    success_count = len(existing_pngs)

    for img_num in range(1, num_images + 1):
        output_path = task_output_dir / f"{task_id}_{img_num}.png"
        if resume and output_path.exists():
            print(f"Skipping image {img_num} (already exists: {output_path.name})")
            continue

        print(f"\n>>> Generating image {img_num}/{num_images}...")
        try:
            image_bytes, meta = backend.generate_image(prompt)
            output_path.write_bytes(image_bytes)

            meta_info = []
            if meta.get("seed") is not None:
                meta_info.append(f"seed={meta['seed']}")
            if meta.get("latency_sec") is not None:
                meta_info.append(f"latency={meta['latency_sec']}s")
            meta_suffix = f" ({', '.join(meta_info)})" if meta_info else ""
            print(f"[OK] Saved image to: {output_path}{meta_suffix}")
            success_count += 1

        except Exception as exc:
            print(f"[ERROR] Failed to generate image {img_num}: {exc}")
            if debug_save_response and isinstance(exc, RuntimeError):
                debug_path = task_output_dir / f"{task_id}_{img_num}_error.txt"
                debug_path.write_text(str(exc), encoding="utf-8")
            continue

    return success_count


def process_json_file(
    backend: BaseBackend,
    json_path: Path,
    prefix: str,
    output_dir: Path,
    num_images: int,
    prompt_field: str,
    resume: bool,
    debug_save_response: bool,
) -> Tuple[int, int]:
    print(f"\n{'#' * 80}")
    print(f"[INFO] Starting file: {json_path}")
    print(f"Task prefix: {prefix}")
    print(f"{'#' * 80}")

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, dict):
        raise ValueError(f"Top-level JSON must be an object: {json_path}")

    print(f"[OK] Loaded {len(tasks)} tasks")
    error_tasks: Dict[str, int] = {}

    for task_id, task_data in tasks.items():
        if not isinstance(task_data, dict):
            print(f"\n[WARN] Skipping task {task_id} (task entry is not an object)")
            error_tasks[task_id] = 0
            continue

        if task_data.get("error"):
            print(f"\n[WARN] Skipping task {task_id} (has error flag)")
            error_tasks[task_id] = 0
            continue

        prompt = task_data.get(prompt_field)
        if not prompt:
            print(f"\n[WARN] Skipping task {task_id} (missing {prompt_field})")
            error_tasks[task_id] = 0
            continue

        success_count = generate_images_for_task(
            backend=backend,
            task_id=task_id,
            prompt=prompt,
            output_dir=output_dir,
            num_images=num_images,
            resume=resume,
            debug_save_response=debug_save_response,
        )
        if success_count < num_images:
            error_tasks[task_id] = success_count
            print(f"\n[WARN] Task {task_id} only generated {success_count}/{num_images} images")
        else:
            print(f"\n[OK] Task {task_id} generated {num_images} images")

    model_name = extract_model_name_from_filename(json_path.name)
    save_error_report(output_dir, error_tasks, model_name, num_images)

    total_tasks = len(tasks)
    successful_tasks = total_tasks - len(error_tasks)
    print(f"\n{'=' * 60}")
    print("[INFO] File summary:")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Successful tasks: {successful_tasks}")
    print(f"   Failed tasks: {len(error_tasks)}")
    print(f"{'=' * 60}")

    return total_tasks, successful_tasks


def main() -> None:
    runner_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Unified TTI Runner (Diffusers + Image API)")
    parser.add_argument(
        "--config",
        default="configs/experiment_local.json",
        help="Path to JSON config (relative paths resolve from the tti_runner folder first).",
    )
    parser.add_argument(
        "--task-types",
        default="",
        help="Override task types from config, for example: co,oe,im",
    )
    parser.add_argument("--disable-resume", action="store_true", help="Disable resume mode.")
    args = parser.parse_args()

    raw_config_path = Path(args.config)
    if raw_config_path.is_absolute():
        config_path = raw_config_path.resolve()
    else:
        # Prefer resolving relative paths from the runner directory to avoid CWD dependency.
        config_path = (runner_dir / raw_config_path).resolve()
        if not config_path.exists():
            # Backward compatibility for relative-to-current-directory usage.
            config_path = raw_config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config_dir = config_path.parent
    config = load_config(config_path)

    input_dirs = config["input_dirs"]
    output_dirs = config["output_dirs"]

    configured_task_types = config.get("task_types", list(input_dirs.keys()))
    if args.task_types.strip():
        task_types = [x.strip() for x in args.task_types.split(",") if x.strip()]
    else:
        task_types = configured_task_types

    recursive = bool(config.get("recursive", True))
    resume = bool(config.get("resume", True)) and (not args.disable_resume)
    num_images = int(config.get("num_images", 1))
    prompt_field = str(config.get("prompt_field", "generated_prompt"))
    debug_save_response = bool(config.get("debug_save_response", True))

    backend = build_backend(config, config_dir)
    output_model_dir = getattr(backend, "output_model_dir", "runner")

    print("=" * 80)
    print("[INFO] Starting batch image generation (Unified Runner)")
    print(f"Config file: {config_path}")
    print(f"Backend mode: {config['mode']}")
    print(f"Task types: {task_types}")
    print(f"Images per task: {num_images}")
    print(f"Resume enabled: {resume}")
    print(f"Recursive JSON search: {recursive}")
    print("=" * 80)

    total_files = 0
    processed_files = 0
    total_tasks = 0
    total_success_tasks = 0

    for prefix in task_types:
        input_raw = input_dirs.get(prefix)
        output_raw = output_dirs.get(prefix)
        if not input_raw:
            print(f"\n[WARN] Missing input_dirs config for task type {prefix}, skipping")
            continue
        if not output_raw:
            print(f"\n[WARN] Missing output_dirs config for task type {prefix}, skipping")
            continue

        input_dir = resolve_path(config_dir, input_raw)
        output_base = resolve_path(config_dir, output_raw)

        if not input_dir.exists():
            print(f"\n[WARN] Input directory does not exist, skipping: {input_dir}")
            continue

        if recursive:
            json_files = sorted(input_dir.rglob("*.json"))
        else:
            json_files = sorted(input_dir.glob("*.json"))

        if not json_files:
            print(f"\n[WARN] No JSON files in input directory, skipping: {input_dir}")
            continue

        print(f"\n[INFO] Processing task type {prefix}: {input_dir} ({len(json_files)} JSON files)")
        for json_file in json_files:
            total_files += 1
            model_name = extract_model_name_from_filename(json_file.name)
            output_dir = output_base / output_model_dir / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                file_total, file_success = process_json_file(
                    backend=backend,
                    json_path=json_file,
                    prefix=prefix,
                    output_dir=output_dir,
                    num_images=num_images,
                    prompt_field=prompt_field,
                    resume=resume,
                    debug_save_response=debug_save_response,
                )
                processed_files += 1
                total_tasks += file_total
                total_success_tasks += file_success
            except Exception as exc:
                print(f"\n[ERROR] Failed to process file: {json_file}")
                print(f"        Error: {type(exc).__name__}: {exc}")
                continue

    print(f"\n{'=' * 80}")
    print("[INFO] Processing complete")
    print(f"Total files: {total_files}")
    print(f"Successful files: {processed_files}")
    print(f"Failed files: {total_files - processed_files}")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful tasks: {total_success_tasks}")
    print(f"Failed tasks: {total_tasks - total_success_tasks}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
