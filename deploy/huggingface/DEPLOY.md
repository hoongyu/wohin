# Deploying Wohin to a Hugging Face Space (owner)

Wohin runs as a **Docker** Space:

- Hugging Face retired the built-in Streamlit SDK on 2025-04-30; Streamlit apps now use the Docker SDK.
- A Space without a Dockerfile defaults to Python 3.10, but scikit-learn 1.9.1 and Streamlit 1.66.1 need 3.11 or newer. The `Dockerfile` here pins Python 3.11.
- The image bakes in the OpenCLIP weights (~600 MB), so the first photo does not wait for a download.

1. On https://huggingface.co/new-space: any name (for example `wohin`), SDK **Docker**, template **Blank**, hardware **CPU basic** (free), visibility **Public**.
2. Clone the Space and copy the app files into it (run from the root of this repository):
   ```bash
   git clone https://huggingface.co/spaces/<your-hf-username>/wohin ../wohin-space
   python deploy/huggingface/make_space.py ../wohin-space
   ```
   This copies `app.py`, `requirements.txt`, `wohin/`, `rules/`, `artifacts/model/`, the `Dockerfile` and this folder's `README.md`, which carries the Space settings (`sdk: docker`, `app_port: 8501`).
3. Push. When git asks for a password, use a Hugging Face access token with write permission (huggingface.co, Settings, Access Tokens).
   ```bash
   cd ../wohin-space
   git add -A
   git commit -m "Deploy Wohin"
   git push
   ```
   Without git you can also upload the same files on the Space page (Files, Add file, Upload files).
4. Wait for the build; the Space page shows the logs. Then open the Space, try the text search, then upload a photo.
5. Put the Space URL into `README.md` (the "Live demo" line at the top) and into the Devpost submission.

Notes

- Photo upload needs `--server.enableXsrfProtection false` on Spaces, because the embedded page cannot use Streamlit's XSRF cookie (https://huggingface.co/docs/hub/spaces-cookie-limitations). The Dockerfile sets it; local runs keep Streamlit's default.
- `requirements.txt` installs CPU-only torch from download.pytorch.org to keep the image small.
- scikit-learn is pinned to the version that wrote `artifacts/model/head.joblib`; keep it pinned.
- Checked locally on 2026-11-02: the image builds from the folder `make_space.py` writes, the app answers on port 8501, and a photo is classified with no network access (`docs/build-log.md`, "Local Docker check").
