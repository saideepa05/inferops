FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

WORKDIR /app

# CPU wheel is ~200 MB vs ~2.5 GB for CUDA. GPU images should start from a
# CUDA base (or the official vllm/vllm-openai image) instead.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY inferops ./inferops
COPY bench ./bench
RUN pip install ".[hf]"

EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "inferops.server:app", "--host", "0.0.0.0", "--port", "8000"]
