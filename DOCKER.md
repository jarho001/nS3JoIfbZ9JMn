# Container development

Use Docker Desktop in Linux-container mode. This launches a separate Codex CLI
session inside Docker; creating the image does not move the current Desktop chat
into the container.

## Build and open (PowerShell)

```powershell
Set-Location 'K:\Pokemon Codex'
docker build -t pokemon-codex .
docker run --rm -it --gpus all --shm-size=2g --mount "type=bind,source=K:\Pokemon Codex,target=/workspace" --mount "type=volume,source=pokemon-codex-auth,target=/home/developer/.codex" pokemon-codex
```

The workspace mount saves edits directly into the Windows project folder. The
named volume retains Codex login and configuration between containers.

Inside the container, initialize Trail once for this project, select Codex, and
sign in to Codex:

```bash
trail init
codex login --device-auth
codex
```

Follow the displayed device-login URL in your Windows browser. If your account
requires enabling device-code authentication, follow the login instructions.
Credentials are supplied at runtime, never embedded in the Dockerfile.

## Included tools

- Node.js 24, npm, Codex CLI and Trail-Code.
  The `developer` user can run `trail upgrade` and npm-based Codex upgrades
  without `sudo`.
- Python 3.13, C/C++ compilers, Make, CMake, Ninja, OpenBLAS, Git and ripgrep.
- CUDA 13 GPU JAX, Flax, Optax, Chex, NumPy, SciPy, pandas, matplotlib, JupyterLab,
  pytest and Ruff in the default environment `/opt/jax-env`.
- Requested Knowledge Catalog repository at `/opt/knowledge-catalog`.
  It has independent samples, so there is no single root package to install.
- Maintained Open Knowledge Format repository at `/opt/open-knowledge-format`,
  installed in `/opt/okf-env`; invoke it with `reference-agent`.

Google Cloud sample execution still requires the relevant project, permissions
and credentials. Google Cloud CLI is not included. An existing Application
Default Credentials JSON can be mounted read-only and referenced using
`GOOGLE_APPLICATION_CREDENTIALS`. Supply `GEMINI_API_KEY` at runtime when using
Gemini's API. Reading or writing local OKF Markdown requires neither service.

The preinstalled environments are read-only for the development user. Add
enduring dependencies to the Dockerfile and rebuild, or create a separate
project virtual environment for experiments. Container-local changes disappear
when the container is removed unless stored in a mounted directory or volume.
The same applies to an in-container Trail or Codex upgrade: it works for the
life of that container, while rebuilding the image is how you make it durable.

## NVIDIA GPU defaults

The default image installs `jax[cuda13]`, including CUDA user-space libraries
and NCCL. Both the RTX 3060 Ti (compute capability 8.6) and RTX 6000 Ada (8.9)
meet JAX CUDA 13's hardware requirement. The host needs a compatible NVIDIA
CUDA 13 driver (Linux minimum 580) and Docker GPU access. On Windows, use Docker
Desktop's WSL2 backend with an up-to-date Windows NVIDIA driver. On a Linux
training host, configure NVIDIA Container Toolkit for Docker. A separate host
CUDA toolkit is not required by the pip-based installation.

Always include `--gpus all`: a Dockerfile cannot grant host GPU access. The
PowerShell command above exposes the single development card. On the training
host, the same option exposes both RTX 6000 Ada cards. `--shm-size=2g` gives
multi-GPU communication more shared memory than Docker's small default.

From your project directory on a Linux training host:

```bash
docker build -t pokemon-codex .
docker run --rm -it --gpus all --shm-size=2g \
  --mount "type=bind,source=$(pwd),target=/workspace" \
  --mount type=volume,source=pokemon-codex-auth,target=/home/developer/.codex \
  pokemon-codex
```

Verify GPU execution inside either container (this fails instead of silently
using CPU if CUDA is unavailable):

```bash
JAX_PLATFORMS=cuda python -c "import jax, jax.numpy as jnp; print(jax.devices()); y = jax.jit(lambda x: x @ x)(jnp.ones((128, 128))); y.block_until_ready(); print(y.devices())"
```

Expect one CUDA device locally and two on the training host. Visibility of two
cards alone does not distribute a training job: training code must use JAX
sharding/data parallelism. GPU memory is per device, not an automatic shared pool.

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is the image default, so JAX allocates on
demand instead of reserving most of the 3060 Ti's 8 GB at startup. This does not
cap memory usage. For dedicated training, optionally pass
`-e XLA_PYTHON_CLIENT_PREALLOCATE=true -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`
to `docker run` after measuring the training workload's memory needs.

The build-time JAX check deliberately uses CPU; Docker builds need no GPU.
To explicitly build a CPU-only variant:

```powershell
docker build --build-arg JAX_PACKAGE=jax -t pokemon-codex-cpu .
```

## Versions and checks

Build arguments `CODEX_VERSION` and `TRAIL_VERSION` accept npm versions;
`KNOWLEDGE_CATALOG_REF` and `OKF_REF` accept branches or tags. Defaults track
upstream releases/main. Python dependencies and image tags are not locked, so
future clean builds can resolve different versions. Pin them for reproducible
training or CI. Use `docker build --pull --no-cache -t pokemon-codex .` to refresh.

Build checks cover both Python dependency environments, CLI entry points, and a
JAX JIT computation. Login, cloud services and GPU operation require runtime
verification.

## Sources

- https://github.com/trailhq/Trail-Code
- https://github.com/GoogleCloudPlatform/knowledge-catalog
- https://github.com/GoogleCloudPlatform/open-knowledge-format
- https://docs.jax.dev/en/latest/installation.html
- https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- https://developer.nvidia.com/cuda/gpus
- https://developers.openai.com/codex/cli
- https://developers.openai.com/codex/auth
