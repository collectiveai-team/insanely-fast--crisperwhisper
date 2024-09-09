# syntax=docker/dockerfile:1.3
# FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
# FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
ENV EDITOR=vim
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

ARG PYTHON_VERSION=python3.12
ARG NODEJS_VERSION=16

RUN sed -i 's/# deb-src/deb-src/' /etc/apt/sources.list \
    && apt update --fix-missing \
    && apt install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

# Install dev dependences & tools
RUN apt update && apt install -y \
    build-essential \
    htop \
    vim \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    ssh \
    rsync \
    kmod \
    unzip \
    zsh \
    git \
    make \
    sudo \
    python3-setuptools \
    ${PYTHON_VERSION} \
    ${PYTHON_VERSION}-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/* \
    && rm -rf /var/lib/apt/lists/*


ARG USER_NAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USER_NAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME \
    && sudo usermod -aG sudo $USER_NAME \
    && sudo passwd -d $USER_NAME

ENV SHELL=/usr/bin/zsh
RUN chsh $USER_NAME -s $SHELL


# Set default python version
# RUN rm /usr/lib/${PYTHON_VERSION}/EXTERNALLY-MANAGED
RUN update-alternatives --install /usr/bin/python python /usr/bin/${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/${PYTHON_VERSION} 1


WORKDIR /tmp

# Install pip
ARG PIP_VERSION=24.1.2
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py

RUN curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)" \
    && chmod +x /usr/local/bin/cog

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY .devcontainer/requirements.txt /home/$USER_NAME/
RUN --mount=type=cache,target=/root/.cache,id=pip \
    pip install -r /home/$USER_NAME/requirements.txt \
    && rm -rf /home/$USER_NAME/requirements.txt



COPY cog.yaml .devcontainer/install_cog_requirements.py /tmp/
RUN --mount=type=cache,target=/root/.cache,id=pip \
    python /tmp/install_cog_requirements.py /tmp/cog.yaml /tmp/cog-requirements.txt \
    && pip install cog -r /tmp/cog-requirements.txt


USER $USER_NAME

ENV PYTHONPATH=$PYTHONPATH:/workspace/
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
