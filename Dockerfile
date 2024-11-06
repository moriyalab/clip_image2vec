FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt update && apt install -y git

WORKDIR /app

RUN python3 -m pip install git+https://github.com/moriyalab/CLIP.git

COPY download_weight.py ./
RUN python3 download_weight.py
