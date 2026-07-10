FROM python:3.13-slim

WORKDIR /code

# install dependencies first so this layer is cached across code-only changes
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src/ src/
COPY app/ app/

ENV PYTHONUNBUFFERED=1
# 7860 is a common convention (Hugging Face Spaces' expected Docker port);
# hosts like Render override this at runtime via their own $PORT env var
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
