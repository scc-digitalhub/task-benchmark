FROM python:3.12-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN python -m pip install --upgrade pip
RUN python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install "transformers>=4.42,<6" "scipy>=1.10,<2"

RUN useradd --create-home --uid 1000 app \
    && install -d -o app -g app /home/app/.cache/huggingface
USER app
WORKDIR /home/app
COPY --chown=app:app server.py /home/app/server.py

EXPOSE 9080
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:9080/v2/health/ready', timeout=3).close()"

CMD ["python", "/home/app/server.py"]