from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        a_norm = np.linalg.norm(a_arr, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b_arr, axis=1, keepdims=True)
        a_norm[a_norm == 0] = 1.0
        b_norm[b_norm == 0] = 1.0
        return (a_arr / a_norm) @ (b_arr / b_norm).T


logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBJECTIVE_MEMORY_DIR = Path(__file__).resolve().parent / "memory"
DATASET_DIR = REPO_ROOT.parent / "dataset"

import subjective_eval_core as core  # noqa: E402
from subjective_eval_core import ImageClusterer, PromptClusterer  # noqa: E402


PROMPT_MEMORY_DEFAULT_FILES: Dict[str, Path] = {
    "co": SUBJECTIVE_MEMORY_DIR / "prompt_from_prompt_memory_co.json",
    "im": SUBJECTIVE_MEMORY_DIR / "prompt_from_prompt_memory_im.json",
    "oe": SUBJECTIVE_MEMORY_DIR / "prompt_from_prompt_memory_oe.json",
}

IMAGE_MEMORY_DEFAULT_FILES: Dict[str, Path] = {
    "co": SUBJECTIVE_MEMORY_DIR / "image_from_image_memory_co.json",
    "oe": SUBJECTIVE_MEMORY_DIR / "image_from_image_memory_oe.json",
}

DATASET_DEFAULT_FILES: Dict[str, Path] = {
    "co": DATASET_DIR / "co_120.json",
    "im": DATASET_DIR / "im_120.json",
    "oe": DATASET_DIR / "oe_120.json",
}


@dataclass(frozen=True)
class PromptExemplar:
    task_id: str
    task_type: str
    task_description: str
    prompt_text: str
    prompt_dimensions: Dict[str, Dict[str, object]]
    source_file: str


@dataclass(frozen=True)
class ImageExemplar:
    task_id: str
    task_type: str
    task_description: str
    image_path: str
    image_dimensions: Dict[str, Dict[str, object]]
    source_file: str


@dataclass(frozen=True)
class RetrievedPromptExemplar:
    exemplar: PromptExemplar
    similarity: float


@dataclass(frozen=True)
class RetrievedImageExemplar:
    exemplar: ImageExemplar
    similarity: float


@dataclass
class PromptMemoryIndex:
    task_type: str
    exemplars: List[PromptExemplar]
    embeddings: np.ndarray


@dataclass
class ImageMemoryIndex:
    task_type: str
    exemplars: List[ImageExemplar]
    embeddings: np.ndarray


@dataclass
class SubjectiveMemoryBank:
    prompt_memories: Dict[str, PromptMemoryIndex]
    image_memories: Dict[str, ImageMemoryIndex]
    task_description_lookup: Dict[str, str]


def infer_task_type(task_id: str, task_type: Optional[str] = None) -> str:
    """Infer normalized task type: co / im / oe."""
    if task_type:
        normalized = task_type.strip().lower()
        if normalized in {"co", "im", "oe"}:
            return normalized
    if "_" in task_id:
        prefix = task_id.split("_", 1)[0].strip().lower()
        if prefix in {"co", "im", "oe"}:
            return prefix
    raise ValueError(f"Cannot infer task type from task_id={task_id!r} and task_type={task_type!r}.")


def load_task_description_lookup(
    task_db_paths: Optional[Sequence[str | Path]] = None,
) -> Dict[str, str]:
    """
    Load task_id -> task description.

    Source:
    - `dataset/` in this repository with task-type-specific formatting (co/oe/im).

    Note:
    - `task_db_paths` is kept for backward compatibility but ignored.
    """
    lookup: Dict[str, str] = {}
    if task_db_paths:
        logger.info("task_db_paths is ignored; descriptions are loaded from dataset files only.")

    def _text(v: object) -> str:
        return str(v).strip() if v is not None else ""

    def _build_from_dataset_item(task_type: str, item: Mapping[str, object]) -> str:
        if task_type == "co":
            sections: List[Tuple[str, str]] = [
                ("Task", _text(item.get("task"))),
                ("Pairing", _text(item.get("pairing"))),
                ("Layout", _text(item.get("layout"))),
                ("Quantity", _text(item.get("quantity"))),
                ("Text", _text(item.get("text"))),
                ("Constraint", _text(item.get("constraint"))),
            ]
            lines: List[str] = []
            for label, content in sections:
                if content:
                    lines.append(f"{label}: {content}")
            return "\n".join(lines).strip()

        if task_type == "oe":
            title = _text(item.get("title"))
            task = _text(item.get("task"))
            if title and task:
                return f"Title: {title}\nTask: {task}"
            return task or title

        if task_type == "im":
            origin_image_rel = _text(item.get("origin_image"))
            origin_image_abs = (
                str((DATASET_DIR / origin_image_rel).resolve())
                if origin_image_rel
                else ""
            )
            lines = [
                f"Reference image: {origin_image_abs or origin_image_rel or 'N/A'}",
                "Please write a prompt that faithfully imitates the reference image.",
            ]
            return "\n".join(lines).strip()

        return _text(item.get("task"))

    # 1) Preferred dataset source.
    for task_type, path in DATASET_DEFAULT_FILES.items():
        if not path.exists():
            logger.warning("Dataset file does not exist: %s", path)
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        for item in records:
            task_id = str(item.get("task_id", "")).strip()
            task_text = _build_from_dataset_item(task_type, item)
            if task_id and task_text:
                lookup[task_id] = task_text

    return lookup


def _normalize_prompt_memory_paths(
    memory_paths: Optional[Mapping[str, str | Path]] = None,
) -> Dict[str, Path]:
    if memory_paths is None:
        return dict(PROMPT_MEMORY_DEFAULT_FILES)
    normalized: Dict[str, Path] = {}
    for task_type, path in memory_paths.items():
        t = task_type.strip().lower()
        if t not in {"co", "im", "oe"}:
            raise ValueError(f"Invalid prompt memory task type: {task_type!r}")
        normalized[t] = Path(path)
    return normalized


def _normalize_image_memory_paths(
    memory_paths: Optional[Mapping[str, str | Path]] = None,
) -> Dict[str, Path]:
    if memory_paths is None:
        return dict(IMAGE_MEMORY_DEFAULT_FILES)
    normalized: Dict[str, Path] = {}
    for task_type, path in memory_paths.items():
        t = task_type.strip().lower()
        if t not in {"co", "oe"}:
            raise ValueError(f"Invalid image memory task type: {task_type!r}")
        normalized[t] = Path(path)
    return normalized


def load_prompt_memories(
    memory_paths: Optional[Mapping[str, str | Path]] = None,
    task_description_lookup: Optional[Mapping[str, str]] = None,
    prompt_clusterer: Optional[PromptClusterer] = None,
) -> Dict[str, PromptMemoryIndex]:
    """
    Load prompt exemplar memories and build prompt-embedding indices.

    The embedding function is exactly reused from PromptClusterer.
    """
    paths = _normalize_prompt_memory_paths(memory_paths)
    desc_lookup = dict(task_description_lookup or load_task_description_lookup())
    clusterer = prompt_clusterer or PromptClusterer()

    memories: Dict[str, PromptMemoryIndex] = {}
    for task_type, path in sorted(paths.items()):
        records = json.loads(path.read_text(encoding="utf-8"))
        exemplars: List[PromptExemplar] = []
        prompt_texts: List[str] = []

        for item in records:
            task_id = str(item.get("task_id", "")).strip()
            prompt_text = str(item.get("prompt", "")).strip()
            if not task_id or not prompt_text:
                continue
            ex_task_type = infer_task_type(task_id, task_type)
            prompt_dims = item.get("prompt_dimensions", {})
            if not isinstance(prompt_dims, dict):
                continue
            task_desc = desc_lookup.get(task_id, "N/A")
            exemplars.append(
                PromptExemplar(
                    task_id=task_id,
                    task_type=ex_task_type,
                    task_description=task_desc,
                    prompt_text=prompt_text,
                    prompt_dimensions=prompt_dims,
                    source_file=str(path),
                )
            )
            prompt_texts.append(prompt_text)

        if prompt_texts:
            embeddings = clusterer.get_embeddings(
                prompt_texts,
                description=f"subjective prompt memories ({task_type})",
            )
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)

        memories[task_type] = PromptMemoryIndex(
            task_type=task_type,
            exemplars=exemplars,
            embeddings=embeddings,
        )
        logger.info(
            "Loaded prompt memory: task_type=%s, exemplars=%d, emb_shape=%s",
            task_type,
            len(exemplars),
            tuple(embeddings.shape),
        )
    return memories


def load_image_memories(
    memory_paths: Optional[Mapping[str, str | Path]] = None,
    task_description_lookup: Optional[Mapping[str, str]] = None,
    image_clusterer: Optional[ImageClusterer] = None,
) -> Dict[str, ImageMemoryIndex]:
    """
    Load image exemplar memories and build image-embedding indices.

    The embedding function is exactly reused from ImageClusterer.
    """
    paths = _normalize_image_memory_paths(memory_paths)
    desc_lookup = dict(task_description_lookup or load_task_description_lookup())
    clusterer = image_clusterer or ImageClusterer()

    memories: Dict[str, ImageMemoryIndex] = {}
    for task_type, path in sorted(paths.items()):
        records = json.loads(path.read_text(encoding="utf-8"))
        raw_exemplars: List[Tuple[ImageExemplar, str]] = []
        image_paths: List[str] = []

        for item in records:
            task_id = str(item.get("task_id", "")).strip()
            image_path = str(item.get("image_path", "")).strip()
            if not task_id or not image_path:
                continue
            ex_task_type = infer_task_type(task_id, task_type)
            image_dims = item.get("image_dimensions", {})
            if not isinstance(image_dims, dict):
                continue
            task_desc = desc_lookup.get(task_id, "N/A")
            image_path_obj = Path(image_path)
            resolved_image_path = (
                str((path.parent / image_path_obj).resolve())
                if not image_path_obj.is_absolute()
                else str(image_path_obj)
            )

            raw_exemplars.append(
                (
                    ImageExemplar(
                        task_id=task_id,
                        task_type=ex_task_type,
                        task_description=task_desc,
                        image_path=resolved_image_path,
                        image_dimensions=image_dims,
                        source_file=str(path),
                    ),
                    resolved_image_path,
                )
            )
            image_paths.append(resolved_image_path)

        if image_paths:
            emb, valid_paths = clusterer.get_embeddings(
                image_paths,
                description=f"subjective image memories ({task_type})",
            )
            emb = np.ascontiguousarray(emb, dtype=np.float32)
            emb_map: Dict[str, np.ndarray] = {
                str(valid_paths[i]): emb[i] for i in range(len(valid_paths))
            }
        else:
            emb = np.empty((0, 0), dtype=np.float32)
            emb_map = {}

        exemplars: List[ImageExemplar] = []
        emb_list: List[np.ndarray] = []
        for ex, resolved_image_path in raw_exemplars:
            vec = emb_map.get(resolved_image_path)
            if vec is None:
                logger.warning("Skip image exemplar with missing embedding: %s", ex.image_path)
                continue
            exemplars.append(ex)
            emb_list.append(vec)

        if emb_list:
            embeddings = np.ascontiguousarray(np.vstack(emb_list), dtype=np.float32)
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)

        memories[task_type] = ImageMemoryIndex(
            task_type=task_type,
            exemplars=exemplars,
            embeddings=embeddings,
        )
        logger.info(
            "Loaded image memory: task_type=%s, exemplars=%d, emb_shape=%s",
            task_type,
            len(exemplars),
            tuple(embeddings.shape),
        )
    return memories


def load_subjective_memory_bank(
    prompt_memory_paths: Optional[Mapping[str, str | Path]] = None,
    image_memory_paths: Optional[Mapping[str, str | Path]] = None,
    task_db_paths: Optional[Sequence[str | Path]] = None,
    prompt_clusterer: Optional[PromptClusterer] = None,
    image_clusterer: Optional[ImageClusterer] = None,
) -> SubjectiveMemoryBank:
    """Load full memory bank for subjective memory-augmented evaluation."""
    desc_lookup = load_task_description_lookup(task_db_paths=task_db_paths)
    prompt_memories = load_prompt_memories(
        memory_paths=prompt_memory_paths,
        task_description_lookup=desc_lookup,
        prompt_clusterer=prompt_clusterer,
    )
    image_memories = load_image_memories(
        memory_paths=image_memory_paths,
        task_description_lookup=desc_lookup,
        image_clusterer=image_clusterer,
    )
    return SubjectiveMemoryBank(
        prompt_memories=prompt_memories,
        image_memories=image_memories,
        task_description_lookup=desc_lookup,
    )


def _format_dimension_lines(
    dimensions: Mapping[str, Mapping[str, object]],
    ordered_names: Sequence[str],
) -> List[str]:
    lines: List[str] = []
    for name in ordered_names:
        payload = dimensions.get(name, {})
        score = payload.get("score", "N/A")
        rationale = payload.get("rationale")
        note = payload.get("note")
        text = rationale if isinstance(rationale, str) else (note if isinstance(note, str) else "")
        lines.append(f"- {name}: score={score}; rationale={text}")
    return lines


def retrieve_top_k_prompt_exemplars(
    memory: PromptMemoryIndex,
    query_embedding: np.ndarray,
    k: int = 3,
) -> List[RetrievedPromptExemplar]:
    """Retrieve Top-K prompt exemplars using cosine similarity."""
    if k <= 0 or memory.embeddings.size == 0 or len(memory.exemplars) == 0:
        return []
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query, memory.embeddings)[0]

    scored: List[Tuple[float, str, int]] = []
    for idx, sim in enumerate(sims):
        ex = memory.exemplars[idx]
        tie_key = f"{ex.task_id}|{idx:06d}"
        scored.append((float(sim), tie_key, idx))
    scored.sort(key=lambda x: (-x[0], x[1]))

    return [
        RetrievedPromptExemplar(exemplar=memory.exemplars[idx], similarity=sim)
        for sim, _, idx in scored[:k]
    ]


def retrieve_top_k_image_exemplars(
    memory: ImageMemoryIndex,
    query_embedding: np.ndarray,
    k: int = 3,
) -> List[RetrievedImageExemplar]:
    """Retrieve Top-K image exemplars using cosine similarity."""
    if k <= 0 or memory.embeddings.size == 0 or len(memory.exemplars) == 0:
        return []
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    sims = cosine_similarity(query, memory.embeddings)[0]

    scored: List[Tuple[float, str, int]] = []
    for idx, sim in enumerate(sims):
        ex = memory.exemplars[idx]
        tie_key = f"{ex.task_id}|{ex.image_path}|{idx:06d}"
        scored.append((float(sim), tie_key, idx))
    scored.sort(key=lambda x: (-x[0], x[1]))

    return [
        RetrievedImageExemplar(exemplar=memory.exemplars[idx], similarity=sim)
        for sim, _, idx in scored[:k]
    ]


def _format_prompt_exemplar_block(exemplars: Sequence[RetrievedPromptExemplar]) -> str:
    if not exemplars:
        return "No retrieved prompt exemplars."
    blocks: List[str] = []
    for rank, item in enumerate(exemplars, start=1):
        ex = item.exemplar
        dim_lines = _format_dimension_lines(ex.prompt_dimensions, core.PROMPT_DIMENSION_NAMES)
        blocks.append(
            "\n".join(
                [
                    f"Exemplar {rank}",
                    f"Task ID: {ex.task_id}",
                    f"Task Description: {ex.task_description}",
                    "Prompt:",
                    ex.prompt_text,
                    "Human Scores & Rationales:",
                    *dim_lines,
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_image_exemplar_block(exemplars: Sequence[RetrievedImageExemplar]) -> str:
    if not exemplars:
        return "No retrieved image exemplars."
    blocks: List[str] = []
    for rank, item in enumerate(exemplars, start=1):
        ex = item.exemplar
        dim_lines = _format_dimension_lines(ex.image_dimensions, core.IMAGE_DIMENSION_NAMES)
        blocks.append(
            "\n".join(
                [
                    f"Exemplar {rank}",
                    f"Task ID: {ex.task_id}",
                    f"Task Description: {ex.task_description}",
                    f"Reference Image: [reference image] {ex.image_path}",
                    "Human Scores & Rationales:",
                    *dim_lines,
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt_subjective_messages_with_memory(
    task_description: str,
    prompt_text: str,
    retrieved_exemplars_for_prompt: Sequence[RetrievedPromptExemplar],
) -> List[Dict]:
    """
    Build prompt-level subjective messages with exemplar memory context.

    Rubric and output schema are identical to zero-shot; this function only prepends exemplars.
    """
    base_user = core.PROMPT_SUBJECTIVE_USER_TEMPLATE.format(
        task_description=task_description,
        prompt_text=prompt_text,
    )
    exemplar_block = _format_prompt_exemplar_block(retrieved_exemplars_for_prompt)
    user_content = "\n\n".join(
        [
            "[Retrieved Exemplars for Calibration]",
            "Use these human-scored exemplars as calibration references before scoring the candidate.",
            exemplar_block,
            "[Candidate to Score]",
            base_user,
        ]
    )
    return [
        {"role": "system", "content": core.PROMPT_SUBJECTIVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_image_subjective_messages_with_memory(
    task_description: str,
    prompt_text: str,
    image_path: str,
    retrieved_exemplars_for_image: Sequence[RetrievedImageExemplar],
) -> Iterator[List[Dict]]:
    """
    Build image-level subjective messages with exemplar memory context.

    Rubric and output schema are identical to zero-shot; this function only prepends exemplars.
    """
    base_user_text = core.IMAGE_SUBJECTIVE_USER_TEXT_TEMPLATE.format(
        task_description=task_description,
        prompt_text=prompt_text,
    )
    exemplar_block = _format_image_exemplar_block(retrieved_exemplars_for_image)
    merged_text = "\n\n".join(
        [
            "[Retrieved Exemplars for Calibration]",
            "Use these human-scored exemplars as calibration references before scoring the candidate image.",
            exemplar_block,
            "[Candidate to Score]",
            base_user_text,
        ]
    )

    for mime_type, image_bytes in core._load_image_variants(image_path):
        yield [
            {"role": "system", "content": core.IMAGE_SUBJECTIVE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": merged_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": core._encode_data_uri(mime_type, image_bytes)},
                    },
                ],
            },
        ]


def compute_prompt_query_embedding(
    prompt_text: str,
    prompt_clusterer: Optional[PromptClusterer] = None,
) -> np.ndarray:
    """Compute normalized query embedding for prompt retrieval."""
    clusterer = prompt_clusterer or PromptClusterer()
    embeddings = clusterer.get_embeddings([prompt_text], description="subjective query prompt")
    if embeddings.shape[0] != 1:
        raise RuntimeError(f"Unexpected prompt embedding shape: {embeddings.shape}")
    return np.asarray(embeddings[0], dtype=np.float32)


def compute_image_query_embedding(
    image_path: str,
    image_clusterer: Optional[ImageClusterer] = None,
) -> np.ndarray:
    """Compute normalized query embedding for image retrieval."""
    clusterer = image_clusterer or ImageClusterer()
    embeddings, valid_paths = clusterer.get_embeddings(
        [image_path],
        description="subjective query image",
        batch_size=1,
        num_workers=0,
    )
    if embeddings.shape[0] != 1 or len(valid_paths) != 1:
        raise RuntimeError(f"Unable to embed query image: {image_path}")
    return np.asarray(embeddings[0], dtype=np.float32)


def evaluate_prompt_subjective_with_memory(
    task_id: str,
    task_type: str,
    task_description: str,
    prompt_text: str,
    model_name: str = core.API_MODEL,
    memories: Optional[SubjectiveMemoryBank] = None,
    prompt_clusterer: Optional[PromptClusterer] = None,
    top_k: int = 3,
) -> Dict:
    """
    Memory-augmented prompt-level subjective evaluation.

    Output keeps the same prompt dimension schema as zero-shot.
    """
    normalized_task_type = infer_task_type(task_id, task_type)
    if memories is None:
        desc_lookup = load_task_description_lookup()
        prompt_memories = load_prompt_memories(
            task_description_lookup=desc_lookup,
            prompt_clusterer=prompt_clusterer,
        )
        bank = SubjectiveMemoryBank(
            prompt_memories=prompt_memories,
            image_memories={},
            task_description_lookup=desc_lookup,
        )
    else:
        bank = memories
    if normalized_task_type not in bank.prompt_memories:
        raise KeyError(f"No prompt memory for task_type={normalized_task_type!r}")

    query_embedding = compute_prompt_query_embedding(prompt_text, prompt_clusterer=prompt_clusterer)
    retrieved = retrieve_top_k_prompt_exemplars(
        memory=bank.prompt_memories[normalized_task_type],
        query_embedding=query_embedding,
        k=top_k,
    )
    messages = build_prompt_subjective_messages_with_memory(
        task_description=task_description,
        prompt_text=prompt_text,
        retrieved_exemplars_for_prompt=retrieved,
    )
    raw = core.call_chat_model_with_model_fallback(messages, model_name=model_name)
    dims = core.parse_prompt_dimensions(raw)
    return {
        "task_id": task_id,
        "task_type": normalized_task_type,
        "prompt": prompt_text,
        "prompt_dimensions": dims,
        "evaluation_mode": "memory_augmented_subjective",
        "retrieved_prompt_exemplars": [
            {
                "task_id": item.exemplar.task_id,
                "similarity": item.similarity,
                "prompt": item.exemplar.prompt_text,
                "prompt_dimensions": item.exemplar.prompt_dimensions,
            }
            for item in retrieved
        ],
    }


def evaluate_image_subjective_with_memory(
    task_id: str,
    task_type: str,
    task_description: str,
    prompt_text: str,
    image_path: str,
    model_name: str = core.API_MODEL,
    memories: Optional[SubjectiveMemoryBank] = None,
    prompt_clusterer: Optional[PromptClusterer] = None,
    image_clusterer: Optional[ImageClusterer] = None,
    top_k: int = 3,
) -> Dict:
    """
    Memory-augmented subjective evaluation for prompt+image samples.

    Routing:
    - IM: prompt-level only.
    - CO/OE: prompt-level + image-level.
    """
    normalized_task_type = infer_task_type(task_id, task_type)
    bank = memories or load_subjective_memory_bank(
        prompt_clusterer=prompt_clusterer,
        image_clusterer=image_clusterer,
    )

    prompt_result = evaluate_prompt_subjective_with_memory(
        task_id=task_id,
        task_type=normalized_task_type,
        task_description=task_description,
        prompt_text=prompt_text,
        model_name=model_name,
        memories=bank,
        prompt_clusterer=prompt_clusterer,
        top_k=top_k,
    )

    if normalized_task_type == "im":
        return {
            "task_id": task_id,
            "task_type": normalized_task_type,
            "prompt": prompt_text,
            "image_path": image_path,
            "prompt_dimensions": prompt_result["prompt_dimensions"],
            "evaluation_mode": "memory_augmented_subjective",
            "retrieved_prompt_exemplars": prompt_result["retrieved_prompt_exemplars"],
        }

    if normalized_task_type not in {"co", "oe"}:
        raise ValueError(f"Unsupported task type for image subjective evaluation: {normalized_task_type}")
    if normalized_task_type not in bank.image_memories:
        raise KeyError(f"No image memory for task_type={normalized_task_type!r}")

    image_query_embedding = compute_image_query_embedding(
        image_path=image_path,
        image_clusterer=image_clusterer,
    )
    retrieved_image = retrieve_top_k_image_exemplars(
        memory=bank.image_memories[normalized_task_type],
        query_embedding=image_query_embedding,
        k=top_k,
    )
    image_messages_variants = build_image_subjective_messages_with_memory(
        task_description=task_description,
        prompt_text=prompt_text,
        image_path=image_path,
        retrieved_exemplars_for_image=retrieved_image,
    )
    image_raw = core._call_image_chat_model_with_fallback(image_messages_variants, model_name=model_name)
    image_dims = core.parse_image_dimensions(image_raw)

    return {
        "task_id": task_id,
        "task_type": normalized_task_type,
        "prompt": prompt_text,
        "image_path": image_path,
        "prompt_dimensions": prompt_result["prompt_dimensions"],
        "image_dimensions": image_dims,
        "evaluation_mode": "memory_augmented_subjective",
        "retrieved_prompt_exemplars": prompt_result["retrieved_prompt_exemplars"],
        "retrieved_image_exemplars": [
            {
                "task_id": item.exemplar.task_id,
                "image_path": item.exemplar.image_path,
                "similarity": item.similarity,
                "image_dimensions": item.exemplar.image_dimensions,
            }
            for item in retrieved_image
        ],
    }


def evaluate_image_dimensions_with_memory(
    task_id: str,
    task_type: str,
    task_description: str,
    prompt_text: str,
    image_path: str,
    model_name: str = core.API_MODEL,
    memories: Optional[SubjectiveMemoryBank] = None,
    image_clusterer: Optional[ImageClusterer] = None,
    top_k: int = 3,
) -> Dict:
    """
    Memory-augmented image-only subjective evaluation (image dimensions only).

    This keeps prompt evaluation separate when callers want one prompt score and
    multiple image scores for the same prompt.
    """
    normalized_task_type = infer_task_type(task_id, task_type)
    if normalized_task_type not in {"co", "oe"}:
        raise ValueError(
            f"Image-level subjective dimensions with memory are only supported for co/oe, got {normalized_task_type}."
        )

    bank = memories or load_subjective_memory_bank(image_clusterer=image_clusterer)
    if normalized_task_type not in bank.image_memories:
        raise KeyError(f"No image memory for task_type={normalized_task_type!r}")

    image_query_embedding = compute_image_query_embedding(
        image_path=image_path,
        image_clusterer=image_clusterer,
    )
    retrieved_image = retrieve_top_k_image_exemplars(
        memory=bank.image_memories[normalized_task_type],
        query_embedding=image_query_embedding,
        k=top_k,
    )
    image_messages_variants = build_image_subjective_messages_with_memory(
        task_description=task_description,
        prompt_text=prompt_text,
        image_path=image_path,
        retrieved_exemplars_for_image=retrieved_image,
    )
    image_raw = core._call_image_chat_model_with_fallback(image_messages_variants, model_name=model_name)
    image_dims = core.parse_image_dimensions(image_raw)

    return {
        "task_id": task_id,
        "task_type": normalized_task_type,
        "image_path": image_path,
        "image_dimensions": image_dims,
        "evaluation_mode": "memory_augmented_subjective",
        "retrieved_image_exemplars": [
            {
                "task_id": item.exemplar.task_id,
                "image_path": item.exemplar.image_path,
                "similarity": item.similarity,
                "image_dimensions": item.exemplar.image_dimensions,
            }
            for item in retrieved_image
        ],
    }


def preview_prompt_subjective_user_content_with_memory(
    task_description: str,
    prompt_text: str,
    retrieved_exemplars_for_prompt: Sequence[RetrievedPromptExemplar],
) -> str:
    """Helper for previewing the final prompt-level user message text."""
    messages = build_prompt_subjective_messages_with_memory(
        task_description=task_description,
        prompt_text=prompt_text,
        retrieved_exemplars_for_prompt=retrieved_exemplars_for_prompt,
    )
    return str(messages[1]["content"])
