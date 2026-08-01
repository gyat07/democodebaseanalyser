def chunk_code(code_files, chunk_size=500, overlap=50, max_chunks=None):
    chunks = []

    for file in code_files:
        text = file["content"]
        path = file["file_path"]
        # Single pass over lines instead of rescanning each chunk's window
        # for a newline — same "don't cut mid-line" behavior, much less work
        # on large files.
        lines = text.splitlines(keepends=True)

        buf = ""
        for line in lines:
            if len(line) > chunk_size:
                if buf:
                    chunks.append({"file_path": path, "chunk": buf})
                    if max_chunks is not None and len(chunks) >= max_chunks:
                        return chunks
                    buf = ""
                # A single line longer than chunk_size (e.g. minified code)
                # has no newline to snap to — hard-split it.
                for i in range(0, len(line), chunk_size):
                    chunks.append(
                        {"file_path": path, "chunk": line[i : i + chunk_size]}
                    )
                    if max_chunks is not None and len(chunks) >= max_chunks:
                        return chunks
                continue

            if buf and len(buf) + len(line) > chunk_size:
                chunks.append({"file_path": path, "chunk": buf})
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks
                buf = buf[-overlap:] if overlap > 0 else ""

            buf += line

        if buf:
            chunks.append({"file_path": path, "chunk": buf})
            if max_chunks is not None and len(chunks) >= max_chunks:
                return chunks

    return chunks
