from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib import error, request

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ..configs.api_runtime import apply_api_spec_to_config
except ImportError:
    from configs.api_runtime import apply_api_spec_to_config  # type: ignore

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "runtime" / "subjective_runtime.json"
logger = logging.getLogger(__name__)


def _load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = apply_api_spec_to_config(cfg, config_path=config_path)

    for key in ["base_url", "endpoint", "model", "api_key"]:
        if key not in cfg.get("api", {}):
            raise ValueError(f"Missing api.{key} in {config_path}")
    if "prompt_templates_path" not in cfg.get("paths", {}):
        raise ValueError(f"Missing paths.prompt_templates_path in {config_path}")
    return cfg


def _resolve_config_path(config_path: Path, configured_path: str) -> Path:
    p = Path(configured_path)
    if p.is_absolute():
        return p.resolve()
    return (config_path.parent / p).resolve()


def _load_prompt_templates(prompt_templates_path: Path) -> dict:
    if not prompt_templates_path.exists():
        raise FileNotFoundError(f"Prompt templates not found: {prompt_templates_path}")
    with prompt_templates_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for root_key in ["prompt_subjective", "image_subjective"]:
        if root_key not in data:
            raise ValueError(f"Missing '{root_key}' in prompt templates: {prompt_templates_path}")

    def _as_text(block: dict, label: str, key_name: str) -> str:
        if key_name in block:
            val = block[key_name]
            if isinstance(val, str):
                return val
            raise ValueError(f"{label}.{key_name} must be a string.")
        alt_key = f"{key_name}_lines"
        if alt_key in block:
            val = block[alt_key]
            if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
                raise ValueError(f"{label}.{alt_key} must be a list of strings.")
            return "\n".join(val)
        raise ValueError(f"{label} must contain '{key_name}' or '{alt_key}'.")

    prompt_block = data["prompt_subjective"]
    image_block = data["image_subjective"]
    if not isinstance(prompt_block, dict):
        raise ValueError("prompt_subjective must be an object.")
    if not isinstance(image_block, dict):
        raise ValueError("image_subjective must be an object.")

    return {
        "prompt_subjective": {
            "system_prompt": _as_text(prompt_block, "prompt_subjective", "system_prompt"),
            "user_template": _as_text(prompt_block, "prompt_subjective", "user_template"),
        },
        "image_subjective": {
            "system_prompt": _as_text(image_block, "image_subjective", "system_prompt"),
            "user_template": _as_text(image_block, "image_subjective", "user_template"),
        },
    }


PROMPT_DIMENSION_NAMES = [
    "Instructional Clarity",
    "Creative Elaboration",
    "Terminology Proficiency",
    "Intent Formalization",
]

IMAGE_DIMENSION_NAMES = [
    "Mood & Atmosphere",
    "Visual Composition",
    "Color & Lighting",
    "Technical Flawlessness",
]

_CONFIG = _load_config(DEFAULT_CONFIG_PATH)
_PROMPT_TEMPLATES_PATH = _resolve_config_path(
    DEFAULT_CONFIG_PATH,
    str(_CONFIG["paths"]["prompt_templates_path"]),
)
_PROMPT_TEMPLATES = _load_prompt_templates(_PROMPT_TEMPLATES_PATH)

# Keep exported names unchanged for compatibility with existing callers.
API_BASE_URL = str(_CONFIG["api"]["base_url"])
API_ENDPOINT = str(_CONFIG["api"]["endpoint"])
API_MODEL = str(_CONFIG["api"]["model"])
API_KEY = str(_CONFIG["api"]["api_key"])
MODEL_FALLBACKS = [
    str(m)
    for m in _CONFIG.get("api", {}).get("model_fallbacks", ["gemini-3-pro-preview", "gpt-5.2", "gpt-5.2-chat"])
]
REQUEST_TIMEOUT_SECONDS = int(_CONFIG.get("retry", {}).get("timeout_seconds", 60))
REQUEST_BACKOFF_SECONDS = [int(x) for x in _CONFIG.get("retry", {}).get("backoff_seconds", [2, 4, 8, 12, 16])]
REQUEST_TEMPERATURE = float(_CONFIG.get("llm_request", {}).get("temperature", 0.0))
REQUEST_TOP_P = _CONFIG.get("llm_request", {}).get("top_p", 0.7)
if REQUEST_TOP_P is not None:
    REQUEST_TOP_P = float(REQUEST_TOP_P)
REQUEST_MAX_TOKENS = _CONFIG.get("llm_request", {}).get("max_tokens", 4096)
if REQUEST_MAX_TOKENS is not None:
    REQUEST_MAX_TOKENS = int(REQUEST_MAX_TOKENS)

PROMPT_SUBJECTIVE_SYSTEM_PROMPT = str(_PROMPT_TEMPLATES["prompt_subjective"]["system_prompt"])
PROMPT_SUBJECTIVE_USER_TEMPLATE = str(_PROMPT_TEMPLATES["prompt_subjective"]["user_template"])
IMAGE_SUBJECTIVE_SYSTEM_PROMPT = str(_PROMPT_TEMPLATES["image_subjective"]["system_prompt"])
IMAGE_SUBJECTIVE_USER_TEXT_TEMPLATE = str(_PROMPT_TEMPLATES["image_subjective"]["user_template"])


def _strip_code_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:]
    elif value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    return value.strip()


def _extract_json_text(text: str) -> str:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response does not contain a valid JSON object boundary.")
    return cleaned[start : end + 1]


def _validate_dimension_payload(raw: object, expected_names: List[str]) -> Dict[str, Dict[str, object]]:
    if not isinstance(raw, dict):
        raise ValueError(f"Response must be a JSON object, got {type(raw).__name__}.")
    keys = set(raw.keys())
    expected = set(expected_names)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"Dimension keys mismatch. Missing={missing}, extra={extra}.")

    normalized: Dict[str, Dict[str, object]] = {}
    for dim_name in expected_names:
        payload = raw[dim_name]
        if not isinstance(payload, dict):
            raise ValueError(f"Dimension '{dim_name}' must be an object.")
        if "score" not in payload:
            raise ValueError(f"Dimension '{dim_name}' missing 'score'.")
        score = payload["score"]
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise ValueError(f"Dimension '{dim_name}' has invalid score: {score!r}.")
        rationale = payload.get("rationale")
        note = payload.get("note")
        text_rationale = rationale if isinstance(rationale, str) else note
        if not isinstance(text_rationale, str):
            raise ValueError(
                f"Dimension '{dim_name}' must contain string 'rationale' (or compatible 'note')."
            )
        normalized[dim_name] = {"score": score, "rationale": text_rationale}
    return normalized


def parse_prompt_dimensions(raw_json_str: str) -> Dict[str, Dict[str, object]]:
    try:
        parsed = json.loads(_extract_json_text(raw_json_str))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prompt response is not valid JSON: {exc}") from exc
    return _validate_dimension_payload(parsed, PROMPT_DIMENSION_NAMES)


def parse_image_dimensions(raw_json_str: str) -> Dict[str, Dict[str, object]]:
    try:
        parsed = json.loads(_extract_json_text(raw_json_str))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Image response is not valid JSON: {exc}") from exc
    return _validate_dimension_payload(parsed, IMAGE_DIMENSION_NAMES)


def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/png"


def _encode_data_uri(mime_type: str, data: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _load_image_variants(image_path: str) -> Iterable[Tuple[str, bytes]]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_bytes = path.read_bytes()
    original_mime = _guess_mime(image_path)
    yield original_mime, original_bytes

    try:
        from PIL import Image
    except Exception:
        return

    with Image.open(path) as img:
        if img.mode == "P":
            img = img.convert("RGBA")
        longest = max(img.width, img.height)
        has_alpha = "A" in img.getbands()

        for target_longest, jpeg_q in [(2048, 95), (1536, 92), (1280, 88), (1024, 85)]:
            work = img
            if longest > target_longest:
                scale = target_longest / float(longest)
                new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                work = img.resize(new_size, Image.Resampling.LANCZOS)

            buf = BytesIO()
            if has_alpha:
                work.save(buf, format="PNG", optimize=True)
                yield "image/png", buf.getvalue()
                rgb = work.convert("RGB")
                buf = BytesIO()
                rgb.save(buf, format="JPEG", quality=jpeg_q, optimize=True, subsampling=0)
                yield "image/jpeg", buf.getvalue()
            else:
                if original_mime == "image/png":
                    work.save(buf, format="PNG", optimize=True)
                    yield "image/png", buf.getvalue()
                    buf = BytesIO()
                rgb = work.convert("RGB")
                rgb.save(buf, format="JPEG", quality=jpeg_q, optimize=True, subsampling=0)
                yield "image/jpeg", buf.getvalue()


def _extract_http_error_text(exc: error.HTTPError) -> str:
    cached = getattr(exc, "_cached_body_text", None)
    if isinstance(cached, str):
        return cached
    try:
        text = exc.read().decode("utf-8", errors="ignore")
        setattr(exc, "_cached_body_text", text)
        return text
    except Exception:
        return ""


def _post_chat_completion(payload: dict) -> str:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=f"{API_BASE_URL}{API_ENDPOINT}",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Invalid completion payload: missing choices in {result}")
    msg = choices[0].get("message", {})
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_chunks.append(str(part.get("text", "")))
        merged = "\n".join(text_chunks).strip()
        if merged:
            return merged
    raise ValueError(f"Invalid message content in completion response: {msg}")


def call_chat_model(messages: List[Dict], model_name: str) -> str:
    payload = {"model": model_name, "messages": messages, "temperature": REQUEST_TEMPERATURE}
    if REQUEST_TOP_P is not None:
        payload["top_p"] = REQUEST_TOP_P
    if REQUEST_MAX_TOKENS is not None:
        payload["max_tokens"] = REQUEST_MAX_TOKENS
    delays = REQUEST_BACKOFF_SECONDS or [2, 4, 8, 12, 16]
    last_exc: Exception | None = None
    for idx, delay in enumerate(delays, start=1):
        try:
            return _post_chat_completion(payload)
        except error.HTTPError as exc:
            if exc.code == 429:
                raise exc
            retryable = exc.code in {500, 502, 503, 504}
            if not retryable:
                raise
            last_exc = exc
            if idx == len(delays):
                break
            time.sleep(delay)
        except error.URLError as exc:
            last_exc = exc
            if idx == len(delays):
                break
            time.sleep(delay)
    if isinstance(last_exc, error.HTTPError):
        raise last_exc
    raise RuntimeError(f"Chat API request failed after retries: {last_exc}")


def _build_model_candidates(model_name: str) -> List[str]:
    ordered = [model_name] + MODEL_FALLBACKS
    unique: List[str] = []
    for m in ordered:
        if m and m not in unique:
            unique.append(m)
    return unique


def call_chat_model_with_model_fallback(messages: List[Dict], model_name: str) -> str:
    candidates = _build_model_candidates(model_name)
    last_error: Exception | None = None
    for idx, candidate in enumerate(candidates, start=1):
        try:
            return call_chat_model(messages=messages, model_name=candidate)
        except error.HTTPError as exc:
            detail = _extract_http_error_text(exc).lower()
            reason = str(exc).lower()
            unavailable = exc.code == 429 or (
                exc.code in {400, 404}
                and ("model_not_found" in detail or "model_not_found" in reason or "does not exist" in detail)
            )
            last_error = exc
            if unavailable and idx < len(candidates):
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if idx < len(candidates):
                continue
            raise
    raise RuntimeError(f"All model candidates failed: {last_error}")


def _is_image_too_large_error(exc: error.HTTPError) -> bool:
    detail = _extract_http_error_text(exc).lower()
    message = f"{exc.code} {exc.reason}".lower() + " " + detail
    keywords = ["too large", "image too large", "payload too large", "request entity too large", "maximum", "size limit"]
    return any(k in message for k in keywords)


def _call_image_chat_model_with_fallback(
    message_variants: Iterable[List[Dict]],
    model_name: str,
) -> str:
    last_error: Exception | None = None
    for messages in message_variants:
        try:
            return call_chat_model_with_model_fallback(messages, model_name)
        except error.HTTPError as exc:
            last_error = exc
            if not _is_image_too_large_error(exc):
                raise
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            raise
    raise RuntimeError(f"Image evaluation failed after trying compression variants: {last_error}")


NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DINOV2_MODEL = "facebook/dinov2-giant"
MATRYOSHKA_DIM = 512
PROMPT_EMBED_BATCH = 16
IMAGE_EMBED_BATCH = 16


class PromptClusterer:
    """Prompt embedding helper used by subjective memory retrieval."""

    def __init__(self, model_name: str = NOMIC_MODEL, matryoshka_dim: int = MATRYOSHKA_DIM):
        from sentence_transformers import SentenceTransformer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("PromptClusterer device: %s", self.device)
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.matryoshka_dim = matryoshka_dim if matryoshka_dim else self.model.get_sentence_embedding_dimension()

    def get_embeddings(self, texts: Sequence[str], description: str = "") -> np.ndarray:
        if not texts:
            return np.empty((0, self.matryoshka_dim), dtype=np.float32)

        batch_size = PROMPT_EMBED_BATCH if self.device.type == "cuda" else 8
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [t if isinstance(t, str) else str(t) for t in texts[i : i + batch_size]]
                emb = self.model.encode(batch, convert_to_tensor=True, device=self.device, batch_size=len(batch))
                if self.matryoshka_dim and self.matryoshka_dim < emb.shape[1]:
                    emb = emb[:, : self.matryoshka_dim]
                emb = F.normalize(emb, p=2, dim=1)
                outputs.append(emb.cpu().numpy())

        merged = np.vstack(outputs).astype(np.float32)
        if not np.all(np.isfinite(merged)):
            merged = np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info("Prompt embeddings done (%s): %s", description, tuple(merged.shape))
        return merged


class ImageClusterer:
    """Image embedding helper used by subjective memory retrieval."""

    def __init__(self, model_name: str = DINOV2_MODEL):
        from transformers import AutoImageProcessor, AutoModel
        from torchvision import transforms

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ImageClusterer device: %s", self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
            ]
        )

    def get_embeddings(
        self,
        image_paths: Sequence[str],
        description: str = "",
        batch_size: int = IMAGE_EMBED_BATCH,
        num_workers: int = 0,  # kept for signature compatibility
    ) -> Tuple[np.ndarray, List[str]]:
        del num_workers
        from PIL import Image

        if not image_paths:
            return np.empty((0, 1536), dtype=np.float32), []

        valid_paths: List[str] = []
        stacked: List[torch.Tensor] = []
        for raw in image_paths:
            path = Path(raw)
            try:
                with Image.open(path) as img:
                    tensor = self.transform(img.convert("RGB"))
                stacked.append(tensor)
                valid_paths.append(str(path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skip image embedding for %s: %s", path, exc)

        if not stacked:
            return np.empty((0, 1536), dtype=np.float32), []

        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(stacked), batch_size):
                batch = torch.stack(stacked[i : i + batch_size], dim=0).to(self.device)
                hidden = self.model(batch).last_hidden_state[:, 0]
                hidden = F.normalize(hidden, p=2, dim=1)
                outputs.append(hidden.cpu().numpy())

        merged = np.vstack(outputs).astype(np.float32)
        if not np.all(np.isfinite(merged)):
            merged = np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)
        logger.info("Image embeddings done (%s): %s", description, tuple(merged.shape))
        return merged, valid_paths
