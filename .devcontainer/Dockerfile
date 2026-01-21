FROM mcr.microsoft.com/devcontainers/base:focal

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
    mesa-common-dev &&\
    rm -rf /var/lib/apt/lists/*

# Install pip and config python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get update \
    && apt install -y -qq --no-install-recommends \
    python3-pip python3-dev


##############################################################################
##                               Libraries                                  ##
##############################################################################
WORKDIR /opt/devcon_lib
# ToTg
RUN git clone https://github.com/djumpstre/py-totg.git \
    && pip3 install pybind11[global] \
    && DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt install -y -qq --no-install-recommends \
    libeigen3-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/devcon_lib/py-totg \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && pip3 install ..

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


##############################################################################
##                               Pybullet                                   ##
##############################################################################
COPY requirements.txt /tmp
RUN pip3 --no-cache-dir install -r /tmp/requirements.txt
