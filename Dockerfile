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
      build-essential \
 && rm -rf /var/lib/apt/lists/*

# Poetry in a venv of its own: installed into /opt/venv, it would have its own dependencies
# (distlib among them) replaced under it by the lock's versions mid-install.
RUN python -m venv /opt/poetry \
 && /opt/poetry/bin/pip install --upgrade pip \
 && /opt/poetry/bin/pip install "poetry==${POETRY_VERSION}"

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel

WORKDIR /deps
COPY pyproject.toml poetry.lock /deps/

# package-mode is false, so there is no project to install, only its dependencies; the dev
# group (pytest, pre-commit) is included so the tests can run on a pod, and the optional hgq
# group (HGQ2, Keras, JAX, alkaid) so the quantised student trains and exports there.
# build-essential above serves any of them that builds from source.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    /opt/poetry/bin/poetry install --no-ansi --no-root --with hgq

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
