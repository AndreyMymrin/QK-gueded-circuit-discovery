# GPU Server Docker Runbook (Jupyter)

This runbook implements a reproducible Docker workflow for this repository on a shared GPU server.

## 1. Preconditions

- Docker and NVIDIA Container Toolkit are installed on the server.
- A single GPU ID is agreed in advance (do not use `--gpus all`).
- Project data is stored under `/mnt/data`, not in `/home/user`.

## 2. Naming convention

- Image: `<username>/prop-logic-transformer:<date>`
- Container: `<username>-logic-gpu<GPU_ID>`

Example:

- Image: `alice/prop-logic-transformer:20260316`
- Container: `alice-logic-gpu1`

## 3. Build image

```bash
chmod +x scripts/docker/*.sh
./scripts/docker/build_image.sh <username> <date>
```

Example:

```bash
./scripts/docker/build_image.sh alice 20260316
```

## 4. Validate GPU access in container

```bash
./scripts/docker/check_gpu.sh <username> <GPU_ID> <date>
```

Expected output includes:

- `cuda_available= True`
- `gpu_name= <your GPU model>`

## 5. Start Jupyter in Docker

By default, the script uses:

- Project mount: `/mnt/data/<username>/prop-logic-transformer-circuit -> /workspace/project`
- HF cache mount: `/mnt/data/<username>/hf-cache -> /workspace/.cache/huggingface`
- Results mount: `/mnt/data/<username>/logic-results -> /workspace/results`
- Port mapping: `127.0.0.1:8888 -> 8888`

Run:

```bash
./scripts/docker/run_jupyter.sh <username> <GPU_ID> <date>
```

Optional environment variables before run:

```bash
export HF_TOKEN=hf_xxx
export JUPYTER_TOKEN=my_token
export HOST_JUPYTER_PORT=8888
```

Open notebook:

- `analysis_walkthrough/LLM Analysis Part 1 Circuit search, interpretation, and verification.ipynb`

## 6. Runtime policy checklist

- If downloading large files/models in daytime, keep network usage within 50 Mbit/s.
- If more network is needed, request admin approval first.
- Announce heavy or long GPU usage in your server RocketChat channel.
- Keep large datasets/caches in `/mnt/data`.
- Call `torch.cuda.empty_cache()` after heavy sections when applicable.

## 7. Cleanup after work

Remove exited containers and dangling Docker artifacts:

```bash
./scripts/docker/cleanup.sh <username>
```

Also remove old, unused user images periodically:

```bash
docker image ls "<username>/prop-logic-transformer"
docker rmi <username>/prop-logic-transformer:<old_tag>
```
