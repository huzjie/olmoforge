FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir . 2>/dev/null || true

ENV PYTHONUNBUFFERED=1

EXPOSE 8899

CMD ["python", "-m", "olmoforge", "doctor"]
