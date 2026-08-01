import logging
import os

logger = logging.getLogger(__name__)

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
_SHOW_PROGRESS = os.getenv("EMBEDDING_PROGRESS", "1") != "0"

_model = None


def _get_model():
    # Loaded lazily so the API starts instantly; the first /analyze or /ask
    # call pays the (one-time) model load, and subsequent calls reuse it.
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading embedding model %s (first run downloads it)...", _MODEL_NAME
        )
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model ready")
    return _model


def warm_up():
    """Load the model ahead of the first request (called at server startup)."""
    _get_model()


def embed_chunks(chunks):
    texts = [c["chunk"] for c in chunks]
    # Progress bar prints to the terminal — useful feedback when indexing a
    # large repository takes a while.
    return _get_model().encode(
        texts,
        batch_size=_BATCH_SIZE,
        show_progress_bar=_SHOW_PROGRESS,
    )


def embed_query(query):
    return _get_model().encode([query], show_progress_bar=False)
