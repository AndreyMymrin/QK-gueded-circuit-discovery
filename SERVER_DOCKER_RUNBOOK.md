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

## 8. ZHORES-2 adaptation (Slurm + Apptainer)

ZHORES-2 compute nodes use Slurm and Apptainer, so Docker containers should be converted before execution.

### 8.1 Build and publish image from your workstation

Build the image with this repository Dockerfile:

```bash
./scripts/docker/build_image.sh <dockerhub_user> <date>
```

Push it to a registry reachable from ZHORES:

```bash
docker push <dockerhub_user>/prop-logic-transformer:<date>
```

### 8.2 SSH to login node

```bash
ssh <username>@cdise.login
# or ssh <username>@10.5.1.1
# or ssh <username>@10.5.1.2
```

### 8.3 Pull image as Apptainer SIF on ZHORES

```bash
module load apptainer
mkdir -p /gpfs/gpfs0/$USER/containers
apptainer pull \
  /gpfs/gpfs0/$USER/containers/prop-logic-transformer-<date>.sif \
  docker://docker.io/<dockerhub_user>/prop-logic-transformer:<date>
```

If you see `no space left on device` during `apptainer build/pull`, move Apptainer temp/cache away from `/tmp`:

```bash
mkdir -p /gpfs/gpfs0/$USER/apptainer-tmp /gpfs/gpfs0/$USER/apptainer-cache
export APPTAINER_TMPDIR=/gpfs/gpfs0/$USER/apptainer-tmp
export APPTAINER_CACHEDIR=/gpfs/gpfs0/$USER/apptainer-cache
export TMPDIR=/gpfs/gpfs0/$USER/apptainer-tmp
```

Then repeat `apptainer pull` or `apptainer build`.

Note: building containers is heavy; use a Slurm CPU job instead of login node:

```bash
srun -p ais-cpu --time=02:00:00 --cpus-per-task=4 --mem=32G --pty bash
```

### 8.4 Submit interactive Jupyter job (GPU queue)

Use template: `scripts/hpc/zhores2_jupyter.sbatch`

```bash
cd /gpfs/gpfs0/$USER/prop-logic-transformer-circuit
chmod +x scripts/hpc/*.sbatch
export IMAGE_SIF=/gpfs/gpfs0/$USER/containers/prop-logic-transformer-<date>.sif
export HF_TOKEN=hf_xxx
export JUPYTER_TOKEN=my_token
sbatch scripts/hpc/zhores2_jupyter.sbatch
```

Then inspect job logs to get the compute node:

```bash
squeue -u $USER
tail -f logic-jupyter-<job_id>.out
```

From your laptop, create SSH tunnel:

```bash
ssh -N -L 8888:<compute_node_hostname>:8888 <username>@cdise.login
```

Open:

```text
http://127.0.0.1:8888/lab?token=my_token
```

### 8.5 Submit notebook execution as a queued batch job

Use template: `scripts/hpc/zhores2_run_notebook.sbatch`

```bash
cd /gpfs/gpfs0/$USER/prop-logic-transformer-circuit
export IMAGE_SIF=/gpfs/gpfs0/$USER/containers/prop-logic-transformer-<date>.sif
export HF_TOKEN=hf_xxx
sbatch scripts/hpc/zhores2_run_notebook.sbatch
```

Output notebook is written to:

- `/gpfs/gpfs0/$USER/logic-results/LLM_Analysis_Part1_executed_<job_id>.ipynb`

## 9. Queue and policy tips for ZHORES-2

- Use `ais-gpu` for GPU jobs and always set realistic `--time`.
- Prefer 1 GPU for this notebook first (`--gres=gpu:1`) and scale only if utilization justifies it.
- Do not run heavy compute from login nodes (`an01`, `an02`); submit via Slurm.
- Keep project, model cache, and outputs under `/gpfs/gpfs0/$USER` to avoid home quota pressure.
