# DGX full-training handoff

Use `scripts/dgx_train.sbatch` to run the reproducible, full-data training
job on a single DGX node. It requests one GPU, 32 CPU cores, 128 GB RAM, and a
12-hour wall-time limit. The trainer itself is single-process; request more
GPUs only after adding distributed training support.

## What can be verified without a DGX

Run the hardware-independent preflight on Windows, macOS, or Linux:

```powershell
.\.venv\Scripts\python.exe scripts\dgx_preflight.py --mode static `
  --output results\dgx_preflight_local.json
```

This is a repeatable delivery check. It validates the split manifest, all raw
source files, cleaned-cache provenance columns, required Python packages,
Slurm resource directives, manifest forwarding, and the GPU parameters that
will be passed to XGBoost, CatBoost, and the stacking XGBoost estimator. A pass
proves that the repository is internally consistent; it does **not** prove
that a particular DGX driver/CUDA/library combination can execute the job.

The repository also includes `.github/workflows/delivery-preflight.yml`.
GitHub Actions runs `bash -n scripts/dgx_train.sbatch` on Ubuntu for every
push and pull request, covering Bash syntax even when the development machine
is Windows without WSL. A green workflow proves Linux parsing only; it is
still separate from CUDA/DGX acceptance.

The final hardware acceptance is intentionally small and runs automatically
inside the batch job before full training. It calls `nvidia-smi` and fits tiny
XGBoost and CatBoost models on CUDA. It normally finishes in under a minute;
if either installed library lacks a compatible GPU backend, the Slurm job
stops before spending hours on full training.

On any Linux workstation with an NVIDIA GPU, the same acceptance can be run
manually:

```bash
python scripts/dgx_preflight.py --mode gpu \
  --output results/dgx_preflight_gpu.json
```

For submission, provide `results/dgx_preflight_local.json` as configuration
evidence and state explicitly that DGX execution is pending. Ask the DGX
operator or instructor to retain the generated
`results/<run-name>_dgx_preflight.json` and Slurm `.out`/`.err` files as the
hardware acceptance evidence.

## Before submitting

1. Copy the repository and the permitted CICIDS2017/CSE-CIC-IDS2018 CSV files
   to the DGX filesystem. Raw files must be under `data/raw/`; do not commit
   them to git.
2. Start in the repository root and create the scheduler log directory before
   submission (Slurm opens its `--output` file before the batch script runs):

   ```bash
   mkdir -p logs/dgx
   ```

3. Verify the GPU allocation and split policy:

   ```bash
   nvidia-smi -L
   python3 main.py --stage audit
   ```

The batch job creates an isolated `.venv-dgx` and installs the project from
`requirements.txt`; a module/conda Python can be selected with `PYTHON_BIN`.
If the DGX uses an internal Python package mirror, configure `PIP_INDEX_URL`
before `sbatch` rather than editing the script.

The job runs `scripts/dgx_preflight.py --mode gpu` after dependency
installation and before training. A failed CUDA smoke fit is a hard failure;
there is no silent CPU fallback.

## Submit an experiment

```bash
mkdir -p logs/dgx
RUN_LABEL=thesis_full_v1 sbatch scripts/dgx_train.sbatch
```

For a clone outside the submission directory or a non-default manifest:

```bash
PROJECT_ROOT=/workspace/ai-classification-cyber-attack \
SPLIT_MANIFEST=/workspace/ai-classification-cyber-attack/configs/splits/source_holdout_v2_70_30.json \
RUN_LABEL=thesis_full_v1 \
sbatch /workspace/ai-classification-cyber-attack/scripts/dgx_train.sbatch
```

The submitted command is deliberately explicit:

```bash
python main.py --stage train --run-name dgx_<label>_<job>_<utc> \
  --preset full --profile overnight \
  --split-manifest configs/splits/source_holdout_v2_70_30.json \
  --model all --accelerator gpu --gpu-devices 0 \
  --imbalance-strategy class_weight --target-max-fpr 0.02
```

`--preset full` keeps all cleaned source-held-out training data. Slurm makes
the assigned GPU visible through `CUDA_VISIBLE_DEVICES`, and the script records
`nvidia-smi` before training. `--accelerator gpu` explicitly enables CUDA for
XGBoost and GPU mode for CatBoost; LightGBM/RF/MLP remain CPU unless their
implementation is changed. Ensure installed XGBoost and CatBoost have the
desired GPU support; their errors are intentionally not hidden as CPU fallback.

## Outputs, monitoring, and resume

Each submission uses a unique output directory:

```
results/dgx_<label>_<slurm-job-id>_<UTC timestamp>/
```

Scheduler output is in `logs/dgx/cyberml-full-<job-id>.out` and `.err`; the
same combined stream is also saved as `logs/dgx/dgx_<...>.log`. Monitor with:

```bash
squeue -j <job-id>
tail -f logs/dgx/cyberml-full-<job-id>.out
```

## DGX acceptance checklist

A DGX run is accepted only when all of the following are retained:

1. `<run-name>_dgx_preflight.json` has `passed: true`, including
   `nvidia_smi`, `xgboost_cuda_fit`, and `catboost_gpu_fit`.
2. Slurm `.out` records the allocated job/GPU and ends with the `complete`
   line; `.err` has no traceback or CUDA out-of-memory error.
3. `results/<run-name>/bundle_manifest.json` verifies successfully.
4. All requested model checkpoints have phase `complete`.
5. `metrics.json` parses as strict JSON and records `accelerator: gpu` plus
   the expected split protocol.
6. The run directory and corresponding scheduler logs are copied back
   together; do not submit metrics without their provenance evidence.

The training pipeline keeps completed model artefacts in a run directory. To
resume after a timeout or preemption, reuse the exact directory name reported
in the log and submit it as `RUN_NAME` (without `--force`); completed models
are skipped:

```bash
mkdir -p logs/dgx
RUN_NAME=dgx_thesis_full_v1_<old-job>_<utc> sbatch scripts/dgx_train.sbatch
```

Use `--force` only to intentionally retrain an artefact.

Before sharing results, retain the run directory, its metrics/manifest files,
and the corresponding Slurm log. Do not treat rare single-source Heartbleed or
PortScan results as independent generalization evidence.
