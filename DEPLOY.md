# Deploying this demo (free)

This is a **self-contained Streamlit app** — one process, no separate frontend/backend, no CORS, no cross-origin env vars to keep in sync. That sidesteps essentially every failure mode from the Vercel+Render split (CORS mismatches, `NEXT_PUBLIC_API_URL` build-time gotchas, Render's 512MB cap). It also gets more headroom: Streamlit Community Cloud's free tier is historically around 1GB RAM, roughly double Render's free cap, which comfortably fits the original SentenceTransformers + FAISS pipeline unmodified.

## 1. Get a free Groq API key

https://console.groq.com/keys

## 2. Push this folder to its own new GitHub repo

This must be a **separate** repo from your main project — don't push this into your existing `AI-Codebase-Analyzer` repo.

```bash
cd streamlit-demo
git init
git add -A
git commit -m "Streamlit demo of AI Codebase Analyzer"
git remote add origin https://github.com/<you>/ai-codebase-analyzer-demo.git
git push -u origin main
```

(Create the empty repo on GitHub first via github.com/new.)

## 3. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io → **New app**
2. Pick the repo you just pushed, branch `main`, main file `app.py`
3. Before deploying, open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
4. Click **Deploy**

First load downloads the ~90MB embedding model once (cached after that). You'll get a public URL like `https://your-app.streamlit.app`.

## Local run (to test before deploying)

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit in your real key
venv/bin/streamlit run app.py
```

## Notes

- Indexes are cached to disk (`data/.index_cache`) — re-analyzing the same repo skips re-embedding.
- `MAX_CHUNKS` defaults to 3000 as a safety margin for the free tier. Override it in Secrets (`MAX_CHUNKS = 0` for no limit) if you have a paid/self-hosted Streamlit instance with more RAM.
- Same RAG pipeline as the main project: SentenceTransformers `all-MiniLM-L6-v2` → FAISS `IndexFlatL2` → Groq `llama-3.3-70b-versatile`. Nothing about the retrieval/generation logic was changed for this demo, only how it's packaged for deployment.
