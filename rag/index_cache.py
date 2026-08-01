import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join("data", ".index_cache")


def cache_key(repo_path, head_sha, fingerprint):
    """Identify a build of the index.

    Includes the commit and the settings that affect the vectors, so changing
    the model or chunk size invalidates old entries instead of silently
    reusing embeddings that no longer match.
    """
    raw = f"{os.path.basename(repo_path)}:{head_sha}:{fingerprint}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _paths(key):
    base = os.path.join(CACHE_DIR, key)
    return base + ".faiss", base + ".json"


def load(key):
    """Return (index, chunks, summary) if a usable cache entry exists."""
    index_path, meta_path = _paths(key)
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return None

    try:
        import faiss

        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta["chunks"], meta["summary"]
    except Exception:
        # A corrupt or half-written entry should just mean "rebuild it".
        logger.warning("Ignoring unreadable cache entry %s", key, exc_info=True)
        return None


def save(key, index, chunks, summary):
    index_path, meta_path = _paths(key)
    try:
        import faiss

        os.makedirs(CACHE_DIR, exist_ok=True)
        faiss.write_index(index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks, "summary": summary}, f)
        logger.info("Cached index %s", key)
    except Exception:
        # Caching is an optimization — never fail the request over it.
        logger.warning("Could not write cache entry %s", key, exc_info=True)
