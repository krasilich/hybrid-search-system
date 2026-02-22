FROM python:3.11-slim

ARG UID=1000
ARG GID=1000

RUN groupadd -g "${GID}" appuser \
    && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" appuser

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER appuser

COPY --chown=appuser:appuser . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]