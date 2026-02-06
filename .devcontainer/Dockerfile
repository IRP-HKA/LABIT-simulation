FROM mcr.microsoft.com/devcontainers/base:jammy

##############################################################################
##                                 Global Dependecies                       ##
##############################################################################

# Install libraries for rendering
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update \
    && apt install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    mesa-utils \
    libgl1-mesa-glx \
    libglu1-mesa-dev \
    freeglut3-dev \
    python3-tk \
    mesa-common-dev &&\
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt install -y -qq --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt install -y -qq --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python3 -m pip install --upgrade pip setuptools wheel

##############################################################################
##                               Libraries                                  ##
##############################################################################
WORKDIR /opt/devcon_lib

# Track IK
RUN git clone https://github.com/mjd3/tracikpy.git \
    && DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt install -y -qq --no-install-recommends \
    libeigen3-dev \
    liborocos-kdl-dev \
    libkdl-parser-dev \
    liburdfdom-dev \
    libnlopt-dev \
    libnlopt-cxx-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*
    
RUN cd /opt/devcon_lib/tracikpy \
    && pip3 install .

COPY .devcontainer/requirements.txt /tmp
RUN pip3 --no-cache-dir install -r /tmp/requirements.txt
