FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ubuntu

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /video_transformation
WORKDIR /video_transformation

RUN python setup.py develop easy_install video_transformation[tensorrt] \
    && python -m video_transformation.tools.install-tensorrt

WORKDIR /home/ubuntu/video_transformation

