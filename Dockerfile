# syntax=docker/dockerfile:1
FROM node:24-bookworm-slim AS node
FROM python:3.13-slim-bookworm

ARG CODEX_VERSION=latest
ARG TRAIL_VERSION=latest
ARG KNOWLEDGE_CATALOG_REF=main
ARG OKF_REF=main
# CUDA 13 supports both RTX 3060 Ti (SM 8.6) and RTX 6000 Ada (SM 8.9).
ARG JAX_PACKAGE=jax[cuda13]

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/jax-env \
    PATH=/opt/jax-env/bin:/usr/local/bin:$PATH

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Native tree-sitter builds, Python extensions, and general development tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git openssh-client ripgrep less nano \
    build-essential cmake ninja-build pkg-config libopenblas-dev libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g "@openai/codex@${CODEX_VERSION}" "@trailhq/trail-code@${TRAIL_VERSION}" \
    && npm cache clean --force

# Knowledge Catalog is a collection of samples, not one installable package.
# Keep the requested repository and install its maintained OKF successor.
RUN git clone --depth 1 --branch "${KNOWLEDGE_CATALOG_REF}" \
      https://github.com/GoogleCloudPlatform/knowledge-catalog.git /opt/knowledge-catalog \
    && git clone --depth 1 --branch "${OKF_REF}" \
      https://github.com/GoogleCloudPlatform/open-knowledge-format.git /opt/open-knowledge-format \
    && python -m venv /opt/okf-env \
    && /opt/okf-env/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/okf-env/bin/python -m pip install --no-cache-dir '/opt/open-knowledge-format[dev]' \
    && ln -s /opt/okf-env/bin/reference-agent /usr/local/bin/reference-agent

# Separate environments prevent Google SDK dependencies from constraining JAX.
RUN python -m venv /opt/jax-env \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "${JAX_PACKAGE}" \
      flax optax chex numpy scipy pandas matplotlib jupyterlab ipykernel pytest ruff \
    && python -m pip check \
    && /opt/okf-env/bin/python -m pip check

RUN useradd --create-home --shell /bin/bash developer \
    && mkdir -p /workspace /home/developer/.codex \
    && chown -R developer:developer /workspace /home/developer

# Recent npm releases require explicit permission for native install scripts.
# The non-root development user owns the agent packages and may replace their
# command links, so `trail upgrade` and npm-based Codex upgrades work in-place.
RUN npm rebuild -g --allow-scripts=@trailhq/trail-code,tree-sitter,tree-sitter-go,tree-sitter-java,tree-sitter-kotlin,tree-sitter-php,tree-sitter-python,@davisvaughan/tree-sitter-r,tree-sitter-typescript,tree-sitter-javascript \
    && chown -R developer:developer \
      /usr/local/lib/node_modules/@trailhq \
      /usr/local/lib/node_modules/@openai \
    && chgrp developer /usr/local/bin \
    && chmod g+w /usr/local/bin

USER developer
# Persist npm's native-script allowlist for future `trail upgrade` runs.
RUN npm config set \
      allow-scripts="@trailhq/trail-code,tree-sitter,tree-sitter-go,tree-sitter-java,tree-sitter-kotlin,tree-sitter-php,tree-sitter-python,@davisvaughan/tree-sitter-r,tree-sitter-typescript,tree-sitter-javascript" \
      --location=user
WORKDIR /workspace
# Fail the build on broken CLI installs or a nonfunctional CPU JAX backend.
RUN codex --version && trail --version && reference-agent --help \
    && JAX_PLATFORMS=cpu python -c "import jax, jax.numpy as jnp; assert int(jax.jit(lambda x: x + 1)(jnp.array(1))) == 2; print(jax.devices())"

# Grow allocations on demand on the 8 GB development GPU. Training can override.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
