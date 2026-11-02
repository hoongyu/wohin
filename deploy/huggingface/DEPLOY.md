# Deploying Wohin to a Hugging Face Space (owner)

The app needs only the committed files: `app.py`, `wohin/`, `rules/`, `artifacts/model/` and
`requirements.txt`. The OpenCLIP weights (~600 MB) are downloaded from the OpenCLIP GitHub
release on the first request, so the first prediction takes a while.

1. Create a new Space on huggingface.co: SDK **Streamlit**, hardware **CPU basic** (free).
2. Clone the Space repository, then copy into it:
   `app.py`, `requirements.txt`, `wohin/`, `rules/`, `artifacts/model/`
   and use `deploy/huggingface/README.md` as the Space's `README.md` (it carries the Space metadata block).
3. `git add . && git commit -m "Deploy Wohin" && git push`.
4. Wait for the build, open the Space, try the text search first (no model download), then a photo.
5. Put the Space URL into the main `README.md` (section "Live demo") and into the Devpost submission.

Notes
- `requirements.txt` installs CPU-only torch from download.pytorch.org to keep the image small.
- scikit-learn is pinned to the version that wrote `artifacts/model/head.joblib`; keep it pinned.
- Streamlit Cloud also works: point it at `app.py`; it reads `requirements.txt`.
