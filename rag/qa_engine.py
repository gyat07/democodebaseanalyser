import os

from dotenv import load_dotenv

load_dotenv()


def generate_answer(question, retrieved_chunks, api_key=None):
    # A fresh client per call (not cached) — on a shared Streamlit deployment
    # every visitor's session can supply its own key, and caching a client
    # keyed to whichever key arrived first would leak it across visitors.
    from groq import Groq

    client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"), timeout=30.0)

    context = "\n\n".join(
        f"File: {c['file_path']}\n{c['chunk']}" for c in retrieved_chunks
    )

    prompt = f"""
You are a senior software engineer analyzing a GitHub repository.

Code Context:
{context}

Question:
{question}

Explain clearly which file or module handles this functionality.
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    return completion.choices[0].message.content
