def retrieve(index, query_embedding, chunks, top_k=5):
    if not chunks:
        return []

    k = min(top_k, len(chunks))
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(chunks[idx])

    return results
