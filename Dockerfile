FROM nvcr.io/nvidia/pytorch:22.04-py3

ENV DEBIAN_FRONTEND=noninteractive 

RUN pip install Pillow==7.1.2 tensorboardX opencv-contrib-python 
RUN apt-get update
# python -c "import cv2; print(cv2.getBuildInformation())" | grep -i ffmpeg
RUN apt-get -y install ffmpeg libsm6 libxext6
WORKDIR /workspace