import logging
import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

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
TORCH_NUM_THREADS = _env_int("TORCH_NUM_THREADS", 1)
MODEL_DIR = os.environ.get(
    "SENTENCE_TRANSFORMERS_HOME",
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
)

_model: Optional[SentenceTransformer] = None
_ready: bool = False


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
    # warm-up encode
    _model.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)
    _ready = True
    logger.info("Embedding model ready device=%s", device)
    return _model


app = FastAPI(title="Embedding API")


@app.on_event("startup")
def _startup() -> None:
    _load_model()


class EmbedRequest(BaseModel):
    texts: list[str]


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


@app.get("/health")
def health() -> JSONResponse:
    if _ready:
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "loading"}, status_code=503)
