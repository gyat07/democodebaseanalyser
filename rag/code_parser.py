import os

SUPPORTED_EXTENSIONS = [".py", ".js", ".ts", ".java", ".cpp", ".c"]

# Directories that hold dependencies or build output rather than the
# repository's own source — skipping them keeps the index focused on the
# code you actually asked about (and makes large repos much faster).
SKIP_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".next",
    ".mypy_cache",
    ".pytest_cache",
    "site-packages",
    "vendor",
}


def load_code_files(repo_path, max_total_chars=None):
    code_files = []
    total_chars = 0

    for root, dirs, files in os.walk(repo_path):
        # Pruning dirs in-place stops os.walk from descending into them.
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            if not any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            code_files.append({"file_path": file_path, "content": content})
            total_chars += len(content)

            if max_total_chars is not None and total_chars >= max_total_chars:
                return code_files

    return code_files
