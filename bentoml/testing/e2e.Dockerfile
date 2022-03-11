# syntax=docker/dockerfile:1.4

ARG PYTHON_VERSION

ARG TEST_ENVIRONMENT=local

FROM --platform=$BUILDPLATFORM bentoml/bento-server:devel-python$PYTHON_VERSION-debian as base

ARG BUILDPLATFORM

FROM base as base-ci

WORKDIR /bentoml

VOLUME [ "/bentoml" ]

RUN git fetch origin --depth 2 origin "${GITHUB_REF#refs/heads/}" && \
    git --work-tree=/bentoml checkout "${GITHUB_REF#refs/heads/}" -- .

FROM base as base-local

WORKDIR /bentoml

VOLUME [ "/bentoml" ]

COPY . .

FROM base-${TEST_ENVIRONMENT} as environment

FROM environment

EXPOSE 3000/tcp
EXPOSE 3000/udp

RUN pip install -U -e .

