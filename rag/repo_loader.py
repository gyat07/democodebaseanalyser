import logging
import os
import re
import shutil

import git

logger = logging.getLogger(__name__)

GITHUB_URL_RE = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


class InvalidRepoUrlError(ValueError):
    pass


class CloneTimeoutError(RuntimeError):
    pass


def parse_repo_url(repo_url):
    match = GITHUB_URL_RE.match(repo_url.strip())
    if not match:
        raise InvalidRepoUrlError(
            "repo_url must look like https://github.com/<owner>/<repo>"
        )
    return match.group(1), match.group(2)


def clone_repository(repo_url, save_dir="data", timeout_seconds=600, refresh=False):
    owner, name = parse_repo_url(repo_url)
    os.makedirs(save_dir, exist_ok=True)

    # Stable path per repository (owner-qualified so same-named repos from
    # different owners never collide) — lets a repeat analysis skip cloning.
    repo_path = os.path.join(save_dir, f"{owner}__{name}")

    if os.path.isdir(repo_path):
        if not refresh:
            logger.info("Reusing existing clone at %s", repo_path)
            return repo_path
        logger.info("Refreshing clone at %s", repo_path)
        shutil.rmtree(repo_path, ignore_errors=True)

    logger.info("Cloning %s/%s...", owner, name)
    try:
        # depth=1 grabs only the current snapshot instead of full history —
        # dramatically faster and smaller on large repositories.
        # kill_after_timeout is GitPython's own watchdog, which (unlike a
        # signal-based timeout) works from FastAPI's worker threads.
        git.Repo.clone_from(
            repo_url, repo_path, depth=1, kill_after_timeout=timeout_seconds
        )
    except git.GitCommandError as exc:
        shutil.rmtree(repo_path, ignore_errors=True)
        if "timeout" in str(exc).lower():
            raise CloneTimeoutError(
                f"git clone timed out after {timeout_seconds}s"
            ) from exc
        raise

    logger.info("Cloned %s/%s to %s", owner, name, repo_path)
    return repo_path


def get_head_sha(repo_path):
    """Current commit of a clone — used to key the index cache."""
    try:
        return git.Repo(repo_path).head.commit.hexsha
    except Exception:
        return "unknown"
