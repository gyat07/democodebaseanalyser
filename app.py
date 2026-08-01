import html
import os
import re

import streamlit as st

from rag import index_cache
from rag.chunker import chunk_code
from rag.code_parser import load_code_files
from rag.embedder import embed_chunks, embed_query
from rag.qa_engine import generate_answer
from rag.repo_loader import (
    CloneTimeoutError,
    InvalidRepoUrlError,
    clone_repository,
    get_head_sha,
)
from rag.repo_summary import generate_repo_summary
from rag.retriever import retrieve
from rag.vector_store import build_index

st.set_page_config(page_title="AI Codebase Analyzer", page_icon="🧠", layout="wide")

try:
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}

# Soft cap so a demo instance can't be knocked over by one huge repo. Raise
# or remove (set to 0) via Streamlit Cloud's "Secrets" panel if needed.
MAX_CHUNKS = int(_secrets.get("MAX_CHUNKS", 3000)) or None

EXAMPLE_QUESTIONS = [
    "How does routing work?",
    "Where is authentication handled?",
    "What is the main entry point?",
]

# ── Same design system as the Next.js frontend (web/app/globals.css) ────────
# Streamlit's markdown renderer splits a raw-HTML block on blank lines,
# closing the <style> tag early and dumping the remaining CSS as visible
# text — collapsing blank lines keeps this one continuous HTML block.
#
# Each "step" is a real st.container(border=True, key="stepN"), which
# Streamlit tags with a stable ".st-key-stepN" class on the actual container
# div — so the header, inputs, and results all render inside one true glass
# panel instead of a decorative header floating above bare widgets.
_STYLE_BLOCK = """
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;650;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: oklch(0.09 0 0);
  --surface: oklch(0.155 0.016 210);
  --surface-hover: oklch(0.185 0.018 210);
  --line: oklch(0.28 0.02 210);
  --line-strong: oklch(0.38 0.03 210);
  --ink: oklch(0.95 0.008 210);
  --muted: oklch(0.68 0.02 210);
  --faint: oklch(0.48 0.02 210);
  --primary: oklch(0.72 0.09 210);
  --primary-soft: oklch(0.72 0.09 210 / 0.14);
  --primary-glow: oklch(0.72 0.09 210 / 0.28);
  --accent: oklch(0.78 0.12 72);
  --accent-soft: oklch(0.78 0.12 72 / 0.12);
  --danger: oklch(0.68 0.16 25);
  --danger-soft: oklch(0.68 0.16 25 / 0.12);
  --success: oklch(0.78 0.12 155);
  --success-soft: oklch(0.78 0.12 155 / 0.12);
  --radius: 10px;
  --shadow: 0 24px 64px oklch(0 0 0 / 0.45);
  --font-sans: "Sora", "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
}
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: var(--font-sans) !important;
}
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: -10%;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(ellipse 70% 45% at 15% -10%, oklch(0.45 0.08 210 / 0.28), transparent 55%),
    radial-gradient(ellipse 55% 38% at 90% 4%, oklch(0.55 0.1 72 / 0.14), transparent 50%),
    radial-gradient(ellipse 45% 35% at 30% 92%, oklch(0.6 0.11 155 / 0.08), transparent 55%),
    linear-gradient(180deg, oklch(0.11 0.01 210), var(--bg) 40%);
  animation: aurora 22s var(--ease) infinite alternate;
}
@keyframes aurora {
  0% { transform: translate3d(0, 0, 0) scale(1); }
  100% { transform: translate3d(2%, 1.5%, 0) scale(1.03); }
}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebarNav"], #MainMenu, footer[data-testid="stFooter"] {
  background: transparent !important;
  visibility: hidden;
}
.block-container { max-width: 1120px; padding-top: 1.5rem; }
.topnav { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 6px 0 8px; position: relative; z-index: 1; }
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
  width: 34px; height: 34px; border-radius: 9px;
  background: linear-gradient(145deg, var(--primary), oklch(0.45 0.08 210));
  box-shadow: 0 0 0 1px oklch(1 0 0 / 0.08), 0 8px 24px var(--primary-glow);
  position: relative; animation: brand-glow 3.2s ease-in-out infinite;
}
.brand-mark::after {
  content: ""; position: absolute; inset: 8px; border: 1.5px solid oklch(1 0 0 / 0.75);
  border-radius: 4px; border-right-color: transparent; transform: rotate(-12deg);
}
@keyframes brand-glow {
  0%, 100% { box-shadow: 0 0 0 1px oklch(1 0 0 / 0.08), 0 8px 24px var(--primary-glow); }
  50% { box-shadow: 0 0 0 1px oklch(1 0 0 / 0.1), 0 8px 32px oklch(0.72 0.09 210 / 0.5); }
}
.brand-text { display: flex; flex-direction: column; gap: 1px; }
.brand-name { font-size: 0.95rem; font-weight: 600; letter-spacing: -0.02em; }
.brand-sub { font-family: var(--font-mono); font-size: 0.68rem; color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; }
.nav-chip {
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted);
  border: 1px solid var(--line); border-radius: 999px; padding: 7px 12px; text-decoration: none;
}
.nav-chip:hover { border-color: var(--line-strong); color: var(--ink); }
.hero { padding: 32px 0 8px; animation: rise 700ms var(--ease) both; position: relative; z-index: 1; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px; font-family: var(--font-mono);
  font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--primary);
  background: var(--primary-soft); border: 1px solid oklch(0.72 0.09 210 / 0.28);
  border-radius: 999px; padding: 6px 12px; margin-bottom: 18px;
}
.pulse { width: 7px; height: 7px; border-radius: 50%; background: var(--primary); display: inline-block; animation: pulse 2.2s ease-out infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 var(--primary-glow); } 70% { box-shadow: 0 0 0 10px transparent; } 100% { box-shadow: 0 0 0 0 transparent; } }
.hero h1 { margin: 0 0 12px; font-size: clamp(2rem, 4.5vw, 3.2rem); line-height: 1.05; letter-spacing: -0.035em; font-weight: 650; }
.hero h1 span {
  background: linear-gradient(120deg, var(--primary), oklch(0.85 0.08 190), var(--accent), var(--primary));
  background-size: 300% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shimmer-text 7s ease-in-out infinite;
}
@keyframes shimmer-text { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
.hero-copy { margin: 0 0 20px; max-width: 60ch; color: var(--muted); font-size: 1rem; }
.hero-meta { display: flex; flex-wrap: wrap; gap: 8px 16px; font-family: var(--font-mono); font-size: 0.75rem; color: var(--faint); margin-bottom: 4px; }
.hero-meta a { color: var(--accent); text-decoration: none; }
.st-key-step0, .st-key-step1, .st-key-step2 {
  background: linear-gradient(180deg, oklch(0.165 0.018 210), var(--surface)) !important;
  border: 1px solid var(--line) !important; border-radius: 14px !important; box-shadow: var(--shadow) !important;
  padding: 4px 22px 18px !important; margin-top: 18px !important; position: relative; overflow: hidden;
}
.st-key-step0::before, .st-key-step1::before, .st-key-step2::before {
  content: ""; position: absolute; left: 0; right: 0; top: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--primary), var(--accent), transparent);
  background-size: 200% 100%; opacity: 0.55; animation: shimmer-line 5s linear infinite; z-index: 2;
}
@keyframes shimmer-line { 0% { background-position: 0% 50%; } 100% { background-position: -200% 50%; } }
.step-label { font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--primary); margin: 18px 0 4px; }
.step-title { margin: 0; font-size: 1.2rem; letter-spacing: -0.02em; font-weight: 600; color: var(--ink); }
.step-desc { margin: 6px 0 2px; color: var(--muted); font-size: 0.9rem; max-width: 60ch; }
[data-testid="stTextInput"] input {
  background: oklch(0.1 0.01 210) !important; border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
  color: var(--ink) !important; font-family: var(--font-mono) !important; font-size: 0.88rem !important; padding: 12px 14px !important;
}
[data-testid="stTextInput"] input:focus { border-color: oklch(0.72 0.09 210 / 0.55) !important; box-shadow: 0 0 0 4px var(--primary-soft) !important; }
[data-testid="stTextInput"] input::placeholder { color: var(--faint) !important; }
[data-testid="stButton"] button {
  background: linear-gradient(180deg, oklch(0.78 0.09 210), var(--primary)) !important;
  color: oklch(0.14 0.02 210) !important; border: none !important; border-radius: var(--radius) !important;
  font-weight: 600 !important; font-size: 0.9rem !important; letter-spacing: -0.01em !important;
  padding: 0.6rem 1.4rem !important; box-shadow: 0 8px 28px var(--primary-glow) !important;
  transition: transform 140ms var(--ease), box-shadow 220ms var(--ease) !important;
}
[data-testid="stButton"] button:hover { transform: translateY(-1px); box-shadow: 0 10px 34px var(--primary-glow) !important; }
.status { margin: 10px 0; font-family: var(--font-mono); font-size: 0.8rem; padding: 12px 14px; border-radius: 6px; border: 1px solid transparent; }
.status-ok { color: var(--success); background: var(--success-soft); border-color: oklch(0.78 0.12 155 / 0.25); }
.status-err { color: var(--danger); background: var(--danger-soft); border-color: oklch(0.68 0.16 25 / 0.28); }
.metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }
.metric { background: oklch(0.1 0.01 210); border: 1px solid var(--line); border-radius: var(--radius); padding: 14px 16px; }
.metric-label { font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--faint); margin-bottom: 8px; }
.metric-value { font-size: 1.4rem; font-weight: 650; letter-spacing: -0.03em; color: var(--primary); }
.metric-value.sm { font-size: 0.92rem; color: var(--ink); line-height: 1.35; }
.module-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px; }
.pill {
  font-family: var(--font-mono); font-size: 0.75rem; color: var(--accent); background: var(--accent-soft);
  border: 1px solid oklch(0.78 0.12 72 / 0.28); border-radius: 999px; padding: 5px 10px;
}
.examples { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 4px; }
.example-chip {
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted); background: transparent;
  border: 1px dashed var(--line); border-radius: 999px; padding: 6px 10px;
}
.answer {
  margin: 12px 0 4px; background: oklch(0.1 0.01 210); border: 1px solid var(--line); border-left: 3px solid var(--primary);
  border-radius: var(--radius); padding: 18px 20px; font-family: var(--font-mono); font-size: 0.86rem;
  line-height: 1.75; color: oklch(0.86 0.015 210); white-space: pre-wrap;
}
.footer-bar {
  margin-top: 32px; padding-top: 20px; border-top: 1px solid var(--line); display: flex; flex-wrap: wrap;
  justify-content: space-between; gap: 10px; font-family: var(--font-mono); font-size: 0.72rem;
  color: var(--faint); letter-spacing: 0.04em; position: relative; z-index: 1;
}
.footer-bar a { color: var(--accent); text-decoration: none; }
@keyframes rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
</style>
"""

st.markdown(re.sub(r"\n\s*\n+", "\n", _STYLE_BLOCK), unsafe_allow_html=True)

# ── Top nav ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="topnav">
  <div class="brand">
    <div class="brand-mark"></div>
    <div class="brand-text">
      <span class="brand-name">AI Codebase Analyzer</span>
      <span class="brand-sub">RAG · FAISS · Groq</span>
    </div>
  </div>
  <a class="nav-chip" href="https://github.com/Yadu080" target="_blank" rel="noreferrer">github.com/Yadu080</a>
</div>
""",
    unsafe_allow_html=True,
)

# ── Hero ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="hero-badge"><span class="pulse"></span> Retrieval-Augmented Generation</div>
  <h1>Read any repo with <span>semantic precision</span></h1>
  <p class="hero-copy">
    Clone a public GitHub repository, embed code into a FAISS index, and ask
    questions answered from real retrieved snippets — not guesswork.
  </p>
  <div class="hero-meta">
    <span>Built by Yadunandan M Nimbalkar</span>
    <span>·</span>
    <a href="https://github.com/Yadu080" target="_blank" rel="noreferrer">Portfolio on GitHub</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
if "session" not in st.session_state:
    st.session_state.session = None

# ── Step 00: Groq API key ───────────────────────────────────────────────
with st.container(border=True, key="step0"):
    st.markdown(
        """
<div class="step-label">Step 00</div>
<h2 class="step-title">Add your Groq API key</h2>
<p class="step-desc">This demo doesn't ship a shared key — bring your own, free, from
<a href="https://console.groq.com/keys" target="_blank" rel="noreferrer" style="color:var(--accent)">console.groq.com/keys</a>.
It's kept only in your browser session, never written to disk.</p>
""",
        unsafe_allow_html=True,
    )
    key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
    )
    if key_input != st.session_state.groq_api_key:
        st.session_state.groq_api_key = key_input

    if st.session_state.groq_api_key:
        st.markdown('<div class="status status-ok">✓ Key set for this session</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status status-err">No key yet — Ask AI is disabled until one is set</div>', unsafe_allow_html=True)

# ── Step 01: Analyze ─────────────────────────────────────────────────────
with st.container(border=True, key="step1"):
    st.markdown(
        """
<div class="step-label">Step 01</div>
<h2 class="step-title">Analyze repository</h2>
<p class="step-desc">Clone, filter source files, chunk at 500 characters, embed with MiniLM, and index with FAISS IndexFlatL2.</p>
""",
        unsafe_allow_html=True,
    )

    repo_url = st.text_input(
        "GitHub repository URL",
        value="https://github.com/pallets/flask",
        placeholder="https://github.com/owner/repo",
        label_visibility="collapsed",
    )

    if st.button("Analyze repository", key="analyze_btn"):
        if not repo_url.strip():
            st.markdown('<div class="status status-err">Enter a GitHub repository URL.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Cloning, chunking, and building the vector index… large repos take a few minutes."):
                try:
                    repo_path = clone_repository(repo_url.strip())
                    key = index_cache.cache_key(repo_path, get_head_sha(repo_path), "all-MiniLM-L6-v2")
                    cached = index_cache.load(key)

                    if cached is not None:
                        index, chunks, summary = cached
                        from_cache = True
                    else:
                        files = load_code_files(repo_path)
                        chunks = chunk_code(files, max_chunks=MAX_CHUNKS)

                        if not chunks:
                            st.markdown(
                                '<div class="status status-err">No supported source files found '
                                "(.py, .js, .ts, .java, .cpp, .c).</div>",
                                unsafe_allow_html=True,
                            )
                            st.stop()

                        embeddings = embed_chunks(chunks)
                        index = build_index(embeddings)
                        summary = generate_repo_summary(repo_path, chunks)
                        index_cache.save(key, index, chunks, summary)
                        from_cache = False

                    st.session_state.session = {"index": index, "chunks": chunks, "summary": summary}
                    msg = "Loaded cached index" if from_cache else "Repository indexed successfully"
                    st.markdown(f'<div class="status status-ok">✓ {msg} · {len(chunks)} chunks indexed</div>', unsafe_allow_html=True)

                except InvalidRepoUrlError as exc:
                    st.markdown(f'<div class="status status-err">✗ {html.escape(str(exc))}</div>', unsafe_allow_html=True)
                except CloneTimeoutError as exc:
                    st.markdown(f'<div class="status status-err">✗ {html.escape(str(exc))}</div>', unsafe_allow_html=True)
                except Exception as exc:
                    st.markdown(f'<div class="status status-err">✗ Analyze failed: {html.escape(str(exc))}</div>', unsafe_allow_html=True)

    if st.session_state.session:
        summary = st.session_state.session["summary"]
        st.markdown(
            f"""
<div class="metrics">
  <div class="metric"><div class="metric-label">Total files</div><div class="metric-value">{summary['total_files']}</div></div>
  <div class="metric"><div class="metric-label">Total chunks</div><div class="metric-value">{summary['total_chunks']}</div></div>
  <div class="metric"><div class="metric-label">Languages</div><div class="metric-value sm">{', '.join(summary['languages']) or '—'}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        if summary.get("main_modules"):
            pills = "".join(
                f'<span class="pill">{html.escape(m)}</span>' for m in summary["main_modules"]
            )
            st.markdown(f'<div class="module-row">{pills}</div>', unsafe_allow_html=True)

# ── Step 02: Ask ─────────────────────────────────────────────────────────
with st.container(border=True, key="step2"):
    st.markdown(
        """
<div class="step-label">Step 02</div>
<h2 class="step-title">Ask the AI</h2>
<p class="step-desc">Retrieve the top-5 nearest code chunks, then generate an explanation with Groq Llama 3.3 70B.</p>
""",
        unsafe_allow_html=True,
    )

    question = st.text_input(
        "Question about the codebase",
        placeholder="How does routing work in this project?",
        label_visibility="collapsed",
    )

    chips = "".join(f'<span class="example-chip">{q}</span>' for q in EXAMPLE_QUESTIONS)
    st.markdown(f'<div class="examples">{chips}</div>', unsafe_allow_html=True)

    if st.button("Ask AI", key="ask_btn"):
        if not st.session_state.groq_api_key:
            st.markdown('<div class="status status-err">✗ Add your Groq API key in Step 00 first.</div>', unsafe_allow_html=True)
        elif not st.session_state.session:
            st.markdown('<div class="status status-err">✗ Analyze a repository first so the index exists.</div>', unsafe_allow_html=True)
        elif not question.strip():
            st.markdown('<div class="status status-err">✗ Type a question about the codebase.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Retrieving relevant chunks and generating an answer…"):
                try:
                    session = st.session_state.session
                    query_embedding = embed_query(question.strip())
                    retrieved = retrieve(session["index"], query_embedding, session["chunks"])
                    answer = generate_answer(
                        question.strip(), retrieved, api_key=st.session_state.groq_api_key
                    )
                    st.markdown('<div class="status status-ok">✓ Answer grounded in retrieved code.</div>', unsafe_allow_html=True)
                    # Escaped because the answer routinely quotes real code —
                    # unescaped "<Component>"-style snippets would otherwise
                    # be interpreted as literal (unknown) HTML tags and swallowed.
                    st.markdown(f'<div class="answer">{html.escape(answer)}</div>', unsafe_allow_html=True)
                except RuntimeError as exc:
                    st.markdown(f'<div class="status status-err">✗ {html.escape(str(exc))}</div>', unsafe_allow_html=True)
                except Exception as exc:
                    st.markdown(f'<div class="status status-err">✗ Ask failed: {html.escape(str(exc))}</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer-bar">
  <span>SentenceTransformers · FAISS · Groq · Deployed on Streamlit Cloud</span>
  <a href="https://github.com/Yadu080" target="_blank" rel="noreferrer">Yadunandan M Nimbalkar</a>
</div>
""",
    unsafe_allow_html=True,
)
