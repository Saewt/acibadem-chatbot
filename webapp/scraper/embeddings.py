from collections.abc import Iterator, Sequence

import torch
from django.conf import settings
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = None
_TORCH_THREADS_CONFIGURED = False
_EMBEDDING_MODEL_WARMED = False
DEFAULT_EMBEDDING_BATCH_SIZE = 8


def _configure_torch_threads() -> None:
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return
    torch.set_num_threads(1)
    _TORCH_THREADS_CONFIGURED = True


def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    _configure_torch_threads()
    if _EMBEDDING_MODEL is None:
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

    model = get_embedding_model()
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        vectors = model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        yield start, [vector.tolist() for vector in vectors]


def embed_texts(
    texts: Sequence[str], batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for _, batch_vectors in iter_text_embedding_batches(texts, batch_size=batch_size):
        vectors.extend(batch_vectors)
    return vectors


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def warm_embedding_model() -> None:
    global _EMBEDDING_MODEL_WARMED
    if _EMBEDDING_MODEL_WARMED:
        return

    get_embedding_model()
    embed_text('warmup')
    _EMBEDDING_MODEL_WARMED = True
