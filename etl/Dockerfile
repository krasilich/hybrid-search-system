FROM python:3.11-slim

ARG UID=1000
ARG GID=1000

RUN groupadd -g "${GID}" appuser \
    && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" appuser

WORKDIR /code/etl

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER appuser
COPY --chown=appuser:appuser . .

CMD ["python", "loader.py"]