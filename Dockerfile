# syntax=docker/dockerfile:1.6
#
# The image the kd-session pods on NGT run this repository in. The environment is the one
# poetry.lock pins: Python 3.11 (the floor pyproject.toml sets), torch 2.9.1+cu126 from the
# PyTorch index the lock records, lightning 2.6. The CUDA runtime comes with the torch wheels,
# so a plain python base is enough; the driver is the pod's. The repository itself is not
# copied in: each pod clones or syncs it (scripts/cluster/setup_pod.sh), so the image only changes when
# the lock does.
#
#   docker build --platform linux/amd64 -t registry.cern.ch/adl1t/distilling_symmetries:dev .
#   docker push registry.cern.ch/adl1t/distilling_symmetries:dev

ARG PYTHON_IMAGE=python:3.11-slim-bookworm
# The version that wrote poetry.lock; an older one may refuse the lock's format.
ARG POETRY_VERSION=2.5.1

############################
# builder
############################
FROM ${PYTHON_IMAGE} AS builder
ARG POETRY_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
      openssh-client \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
 && pip install --upgrade pip setuptools wheel \
 && pip install "poetry==${POETRY_VERSION}"

WORKDIR /deps
COPY pyproject.toml poetry.lock /deps/

# package-mode is false, so there is no project to install, only its dependencies; the dev
# group (pytest, pre-commit) is included so the tests can run on a pod.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-ansi --no-root

############################
# runtime
############################
FROM ${PYTHON_IMAGE} AS dev

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      bash \
      vim \
      less \
      procps \
      htop \
      tmux \
      openssh-client \
      git \
      rsync \
      curl \
      wget \
      tini \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
CMD ["bash"]
