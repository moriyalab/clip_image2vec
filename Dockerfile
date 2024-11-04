FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git curl python3-pip
RUN git config --global --add safe.directory /app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install poetry \
  && poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

COPY download_weight.py ./
RUN python3 download_weight.py
