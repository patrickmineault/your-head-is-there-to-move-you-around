# V-JEPA 2.1 + Midway brain-alignment extension

Adds two modern video self-supervised ViTs as frozen feature extractors and fits
them to single-neuron data with the existing ridge pipeline.

## Models (vendored under `../third_party/`, checkpoints under `--ckpt_root`)
| `--features`        | arch              | checkpoint                          | layers hooked                | path     |
|---------------------|-------------------|-------------------------------------|------------------------------|----------|
| `vjepa2_1_vitl`     | V-JEPA 2.1 ViT-L/16 384 (3-D, RoPE) | `vjepa2_1_vitl_dist_vitG_384.pt` | blocks 0,4,8,12,16,20,23 (7/24) | 3-D (`threed=True`) |
| `midway_bdd_vitb`   | Midway ViT-B/16 224 (2-D per-frame) | `midway-bdd-vit-b-ep300.pth` (teacher backbone) | blocks 1,3,5,7,9,11 (6/12) | 2-D (`threed=False`) |

Both integrate via `modelzoo/transformer_models.py`: an `InputAdapter`
(`--input_adapt resize|pad`, reflect/mirror-pad in x/y and optional t via
`--vjepa_pad_t`) guarantees patch divisibility, and grid-reshaping forward hooks
turn the token sequences back into `(B, D, T', H', W')` feature maps so the
existing `Downsampler`/`Averager` aggregators work unchanged.

## Data + checkpoints in GCS
`gs://xcorr-dev-motion-model/{data_derived,checkpoints,features,results}/`. Stage with:
```
scripts/sync_datasets_to_gcs.sh gs://xcorr-dev-motion-model
```

## Running jobs (configurable parallelism)
`cloud/launch_jobs.py` caps simultaneous workers at `--max-parallel N` (hard cap 8).

```
# Local smoke test (CPU), one neuron:
python cloud/launch_jobs.py --backend local --max-parallel 2 --device cpu \
    --datasets pvc4 --models midway_bdd_vitb --max-cells 1 \
    --data_root ../data_derived --ckpt_root ../checkpoints --cache_root /tmp/mm_cache --no_wandb

# Full run on a GPU VM with 6 concurrent processes:
python cloud/launch_jobs.py --backend local --max-parallel 6 --device cuda \
    --data_root /data/data_derived --ckpt_root /data/checkpoints

# GCP Batch, <=4 simultaneous GPU VMs, NO Docker (recommended):
#   a script runnable on a Deep Learning VM image pulls cloud/bootstrap.sh from the
#   repo, which git-clones the repo + model sources and runs the work unit.
python cloud/launch_jobs.py --backend batch --max-parallel 4 --repo-ref vjepa-midway-extension

# (optional) container path -- the image bakes deps + model repos but still git-clones
# the repo at runtime (cloud/Dockerfile); pass --image to use it:
python cloud/launch_jobs.py --backend batch --max-parallel 4 \
    --image us-central1-docker.pkg.dev/xcorr-dev/motion/worker:latest
```

**No Docker required.** The default Batch backend runs a script on a CUDA Deep
Learning VM and pulls the repo at runtime (`cloud/bootstrap.sh`); nothing is baked.
The container path is optional and, per best practice, still clones the analysis
repo at runtime (the image only bakes dependencies + the bootstrap launcher).

Each `fit` job runs one ViT forward pass over its neuron's clips (extract → cache →
ridge over all hooked layers). **A GPU is required**: V-JEPA ViT-L (~300M params)
over thousands of clips on CPU is impractical (see the timed smoke test). The
per-cell-stimulus datasets (pvc1/pvc4) extract per neuron; the standalone
`--stage extract` only helps shared-stimulus reuse.

Results are written via offline wandb (`WANDB_MODE=offline`) and synced to
`gs://.../results/`; assemble with `notebooks/Compare results_physiology.ipynb`.

## Neuron counts (from run_remote.sh)
pvc1-repeats 22 · pvc4 24 · mt1_norm_neutralbg 83 · mt2 43 · mst_norm_neutralbg 35
(≈ 207 neurons × 2 models = ~414 fit jobs.)
