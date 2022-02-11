FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl wget openssh-client -y
RUN mkdir app
WORKDIR /app

RUN python3 -m pip install --upgrade pip
RUN pip3 install \
    opencv-python \
    optuna==2.0.0 \
    pandas \
    matplotlib \
    torch \
    numpy \
    Pillow \
    bs4 \
    scikit-learn \
    torchvision \
    pytorchtools \
    joblib\
    scikit-image \
    pathlib \
    seaborn \
    scikit-plot \
    torch-summary

