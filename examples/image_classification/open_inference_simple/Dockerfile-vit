FROM python:3.12.16-slim-trixie

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
	"torch==2.2.2" \
	"transformers==4.42.1" \
	"pillow>=10.0.0"

WORKDIR /workspace
COPY server.py /workspace/server.py

CMD ["python", "/workspace/server.py"]