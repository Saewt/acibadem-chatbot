import logging
import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(Path(__file__).resolve().parent.parent / ".env")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(value, minimum)


EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 2)
RERANK_MODEL = os.environ.get(
    "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
TORCH_NUM_THREADS = _env_int("TORCH_NUM_THREADS", 1)
MODEL_DIR = os.environ.get(
    "SENTENCE_TRANSFORMERS_HOME",
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
)

_model: Optional[SentenceTransformer] = None
_ready: bool = False
_rerank_model: Optional[CrossEncoder] = None
_rerank_ready: bool = False


def _select_device() -> str:
    preference = os.environ.get("EMBEDDING_DEVICE", "cpu").strip().lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if preference == "auto" and mps_available:
        return "mps"
    if preference == "mps" and mps_available:
        return "mps"
    return "cpu"


def _load_model() -> SentenceTransformer:
    global _model, _ready
    torch.set_num_threads(TORCH_NUM_THREADS)
    device = _select_device()
    logger.info(
        "Loading embedding model=%s device=%s batch_size=%s torch_threads=%s",
        EMBEDDING_MODEL,
        device,
        EMBEDDING_BATCH_SIZE,
        TORCH_NUM_THREADS,
    )
    _model = SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=MODEL_DIR,
    )
    _model = _model.to(device)
    _model.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)
    _ready = True
    logger.info("Embedding model ready device=%s", device)
    return _model


def _load_rerank_model() -> CrossEncoder:
    global _rerank_model, _rerank_ready
    device = _select_device()
    logger.info("Loading rerank model=%s device=%s", RERANK_MODEL, device)
    _rerank_model = CrossEncoder(RERANK_MODEL, cache_folder=MODEL_DIR)
    if device != "cpu":
        _rerank_model.model.to(device)
    _rerank_ready = True
    logger.info("Rerank model ready device=%s", device)
    return _rerank_model


app = FastAPI(title="Embedding API")


@app.on_event("startup")
def _startup() -> None:
    _load_model()
    _load_rerank_model()


class EmbedRequest(BaseModel):
    texts: list[str]


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 12


@app.post("/embed")
def embed(req: EmbedRequest) -> JSONResponse:
    if not _ready or _model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    if not req.texts:
        return JSONResponse({"embeddings": []})
    embeddings: list[list[float]] = []
    for start in range(0, len(req.texts), EMBEDDING_BATCH_SIZE):
        batch = req.texts[start : start + EMBEDDING_BATCH_SIZE]
        vectors = _model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings.extend(v.tolist() for v in vectors)
    return JSONResponse({"embeddings": embeddings})


@app.post("/rerank")
def rerank(req: RerankRequest) -> JSONResponse:
    if not _rerank_ready or _rerank_model is None:
        raise HTTPException(status_code=503, detail="Rerank model not ready")
    if not req.documents:
        return JSONResponse({"indices": [], "scores": []})
    pairs = [(req.query, doc) for doc in req.documents]
    scores: list[float] = _rerank_model.predict(pairs).tolist()
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_items = indexed[:req.top_k]
    return JSONResponse({
        "indices": [idx for idx, _ in top_items],
        "scores": [score for _, score in top_items],
    })


@app.get("/health")
def health() -> JSONResponse:
    if _ready:
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "loading"}, status_code=503)
