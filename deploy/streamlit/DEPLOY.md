# Deploying Wohin to Streamlit Community Cloud (owner)

The live demo runs on Streamlit Community Cloud, which is free. On 2026-11-03 Hugging Face refused to create the Docker Space from a free account (`402 Payment Required`: Gradio and Docker Spaces on the free CPU tier now need a PRO subscription), so `deploy/huggingface/` is kept for PRO accounts.

Why this folder exists: Community Cloud installs the first dependency file it finds, looking next to the entry point before the repository root (https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). The root has `uv.lock`, which pins the CUDA build of torch on Linux (several GB). Here, `requirements.txt` installs CPU-only torch, and `app.py` runs the app in the repository root.

1. Sign in at https://share.streamlit.io with GitHub. A private repository needs GitHub's extra `repo` permission, and Community Cloud allows one private app at a time; a public repository needs only the default permissions.
2. Create an app from GitHub: repository `hoongyu/wohin`, branch `main`, main file path `deploy/streamlit/app.py`. Custom subdomain: `wohin-berlin` (if it is taken, pick another and update the links in `README.md` and `docs/submission/neural-storm.md`).
3. Advanced settings: Python **3.11**, the version the tests and the Docker image use. No secrets are needed.
4. Deploy and wait for the build. Then try the text search and upload a photo. The first photo downloads the OpenCLIP weights (~600 MB) from GitHub.
5. An app deployed from a private repository is private. After the repository is public, make the app public in its settings.

Checked locally on 2026-11-03: `uv pip compile deploy/streamlit/requirements.txt` for Python 3.11 on Linux x86_64 resolves 60 packages with torch 2.15.0+cpu and no CUDA packages, and `tests/test_deploy.py` runs this entry point with Streamlit's AppTest.
