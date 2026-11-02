# Hugging Face Docker Space for Wohin. Build context: the folder written by make_space.py.
# Python 3.11: scikit-learn 1.9.1 and Streamlit 1.66.1 need 3.11 or newer.
FROM python:3.11-slim

# Spaces run the container as user ID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py .
COPY --chown=user wohin ./wohin
COPY --chown=user rules ./rules
COPY --chown=user artifacts/model ./artifacts/model

# Bake the pinned OpenCLIP weights (~600 MB) into the image, so the first photo does not wait for a download.
RUN python -c "from wohin.embed import _fetch_weights; _fetch_weights('clip')"

EXPOSE 8501
# File uploads on Spaces need XSRF protection off: https://huggingface.co/docs/hub/spaces-cookie-limitations
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true", "--server.enableXsrfProtection", "false"]
