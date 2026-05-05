import logging
from collections.abc import Iterator, Sequence

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = None
_TORCH_THREADS_CONFIGURED = False
_EMBEDDING_MODEL_WARMED = False
DEFAULT_EMBEDDING_BATCH_SIZE = settings.EMBEDDING_BATCH_SIZE


def _configure_torch_threads() -> None:
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return
    import torch

    torch.set_num_threads(1)
    _TORCH_THREADS_CONFIGURED = True


def get_embedding_model():
    global _EMBEDDING_MODEL
    _configure_torch_threads()
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            local_files_only=True,
        )
    return _EMBEDDING_MODEL


def iter_text_embedding_batches(
    texts: Sequence[str], batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
) -> Iterator[tuple[int, list[list[float]]]]:
    if batch_size < 1:
        raise ValueError('batch_size must be greater than 0')
    if not texts:
        return

    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        if settings.EMBEDDING_BACKEND == 'api':
            yield start, _embed_via_api(batch)
            continue

        model = get_embedding_model()
        vectors = model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        yield start, [vector.tolist() for vector in vectors]


def _embed_local(
    texts: Sequence[str], batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for _, batch_vectors in iter_text_embedding_batches(texts, batch_size=batch_size):
        vectors.extend(batch_vectors)
    return vectors


def _embed_via_api(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    url = f"{settings.EMBEDDING_API_URL.rstrip('/')}/embed"
    timeout = settings.EMBEDDING_API_TIMEOUT
    try:
        resp = requests.post(
            url,
            json={"texts": list(texts)},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Embedding API request timed out after {timeout}s: {exc}"
        ) from exc
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Embedding API unreachable at {url}: {exc}"
        ) from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"Embedding API returned status {resp.status_code}: {resp.text[:200]}"
        )
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Embedding API returned invalid JSON: {exc}"
        ) from exc
    if "embeddings" not in payload:
        raise RuntimeError(
            "Embedding API response missing 'embeddings' field"
        )
    embeddings = payload["embeddings"]
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
        )
    return embeddings


def embed_texts(
    texts: Sequence[str], batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
) -> list[list[float]]:
    if settings.EMBEDDING_BACKEND == 'api':
        return _embed_via_api(texts)
    return _embed_local(texts, batch_size=batch_size)


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def warm_embedding_model() -> None:
    global _EMBEDDING_MODEL_WARMED
    if _EMBEDDING_MODEL_WARMED:
        return

    embed_text('warmup')
    _EMBEDDING_MODEL_WARMED = True
