FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN pip install scipy

RUN apt install unzip