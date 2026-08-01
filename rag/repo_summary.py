import os

from rag.code_parser import SKIP_DIRS

LANGUAGE_BY_EXTENSION = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".java": "Java",
    ".cpp": "C++",
    ".c": "C",
}


def generate_repo_summary(repo_path, chunks):
    total_files = 0
    languages = set()
    modules = []

    for root, dirs, files in os.walk(repo_path):
        # Same pruning as the indexer, so the reported counts describe the
        # repository's own code rather than its vendored dependencies.
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            total_files += 1

            _, ext = os.path.splitext(file)
            language = LANGUAGE_BY_EXTENSION.get(ext)
            if language:
                languages.add(language)

            if ext == ".py":
                modules.append(file)

    return {
        "languages": sorted(languages),
        "main_modules": modules[:5],
        "total_files": total_files,
        "total_chunks": len(chunks),
    }
