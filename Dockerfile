FROM nvcr.io/nvidia/pytorch:22.04-py3
RUN pip install Pillow==7.1.2 tensorboardX

WORKDIR /workspace