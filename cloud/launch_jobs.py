#!/usr/bin/env python3
"""Launch V-JEPA 2.1 / Midway brain-alignment jobs with a configurable cap on the
number of simultaneous workers (the user has < 8 GPU VMs available).

Work is split into two stages:
  * extract  -- one GPU job per (dataset, model): build the feature cache. [10 jobs]
  * fit      -- one job per (dataset, model, neuron): ridge-fit from the cache.

Backends:
  * local  -- run up to N `train_convex.py` subprocesses concurrently on THIS
              machine (a single multi-GPU / CPU VM). Fully self-contained; the
              validated path. `--max-parallel N` is the hard concurrency cap.
  * batch  -- submit a GCP Batch job whose taskGroup parallelism == N, so at most
              N GPU VMs run at once. Robust to whatever GPU quota you have.

Examples:
  # Prove the pipeline on one dataset, 2 neurons, locally on CPU:
  python cloud/launch_jobs.py --backend local --max-parallel 2 --device cpu \
      --datasets pvc4 --max-cells 1 --data_root ../data_derived --stage both

  # Full run on a GPU VM, 6 workers:
  python cloud/launch_jobs.py --backend local --max-parallel 6 --device cuda \
      --data_root /data/data_derived --stage both

  # Submit to GCP Batch, <=4 simultaneous GPU VMs:
  python cloud/launch_jobs.py --backend batch --max-parallel 4 \
      --image us-central1-docker.pkg.dev/xcorr-dev/motion/worker:latest --stage extract
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

# dataset train-name -> number of neurons (max subset index, inclusive), from run_remote.sh
DATASETS = {
    "pvc1-repeats": 22,
    "pvc4": 24,
    "mt1_norm_neutralbg": 83,
    "mt2": 43,
    "mst_norm_neutralbg": 35,
}
MODELS = ["vjepa2_1_vitl", "midway_bdd_vitb"]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def build_manifest(stage, datasets, models, max_cells_override):
    units = []
    if stage in ("extract", "both"):
        for ds in datasets:
            for mdl in models:
                units.append(("extract", ds, mdl, 0))
    if stage in ("fit", "both"):
        for ds in datasets:
            n = DATASETS[ds] if max_cells_override is None else max_cells_override
            for mdl in models:
                for s in range(n + 1):
                    units.append(("fit", ds, mdl, s))
    return units


def local_cmd(unit, a):
    mode, ds, mdl, subset = unit
    cmd = [
        sys.executable, os.path.join(REPO, "train_convex.py"),
        "--exp_name", a.exp_name, "--dataset", ds, "--features", mdl,
        "--data_root", a.data_root, "--ckpt_root", a.ckpt_root,
        "--cache_root", a.cache_root, "--aggregator", a.aggregator,
        "--aggregator_sz", "8", "--pca", str(a.pca), "--method", "ridge",
        "--resize", "112", "--device", a.device, "--input_adapt", a.input_adapt,
        "--vjepa_pad_t", str(a.vjepa_pad_t), "--batch_size", str(a.batch_size),
    ]
    if mode == "extract":
        cmd += ["--subset", "0", "--extract_only"]
    else:
        cmd += ["--subset", str(subset), "--no_wandb" if a.no_wandb else "--save_predictions"]
    return cmd


def run_local(units, a):
    """Run units through a bounded thread pool (cap = a.max_parallel). Extraction
    units are run to completion before fit units (a fit needs its cache)."""
    extracts = [u for u in units if u[0] == "extract"]
    fits = [u for u in units if u[0] == "fit"]
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    def run_one(unit):
        cmd = local_cmd(unit, a)
        tag = ":".join(map(str, unit))
        if a.dry_run:
            print("DRY", " ".join(cmd))
            return tag, 0
        p = subprocess.run(cmd, cwd=REPO, env=env)
        return tag, p.returncode

    for phase, group in (("extract", extracts), ("fit", fits)):
        if not group:
            continue
        print(f"=== {phase}: {len(group)} units, max_parallel={a.max_parallel} ===")
        with cf.ThreadPoolExecutor(max_workers=a.max_parallel) as ex:
            for tag, rc in ex.map(run_one, group):
                status = "ok" if rc == 0 else f"FAIL({rc})"
                print(f"  [{status}] {tag}")


def _raw_url(repo_url, ref, path):
    """github.com/owner/name(.git) -> raw.githubusercontent.com/owner/name/ref/path"""
    slug = repo_url.replace("https://github.com/", "").replace(".git", "").strip("/")
    return f"https://raw.githubusercontent.com/{slug}/{ref}/{path}"


def run_batch(units, a):
    """Submit a GCP Batch job with parallelism == max_parallel (<= N GPU VMs).

    Default: a *script* runnable on a Deep Learning VM image -- no Docker, no image
    registry. The script pulls cloud/bootstrap.sh from the repo and runs it, which
    git-clones the repo + model sources and runs the work unit. Pass --image to use
    a container runnable instead (the image's bootstrap still clones the repo)."""
    manifest = "\n".join(f"{m}\t{d}\t{f}\t{s}" for (m, d, f, s) in units) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        fh.write(manifest)
        local_manifest = fh.name
    gcs_manifest = f"{a.bucket}/manifests/{a.exp_name}_{a.stage}.tsv"
    subprocess.run(["gsutil", "cp", local_manifest, gcs_manifest], check=not a.dry_run)

    env_vars = {
        "MANIFEST_GCS": gcs_manifest, "BUCKET": a.bucket, "DEVICE": "cuda",
        "INPUT_ADAPT": a.input_adapt, "EXP_NAME": a.exp_name,
        "REPO_URL": a.repo_url, "REPO_REF": a.repo_ref,
    }
    if a.image:
        # Batch's default host needs GPU drivers installed for a container runnable.
        runnable = {"container": {"imageUri": a.image}, "environment": {"variables": env_vars}}
        instance_policy = {"machineType": a.machine_type,
                           "accelerators": [{"type": a.gpu_type, "count": 1}]}
        install_drivers = True
    else:
        # No Docker: pull bootstrap.sh from the repo and run it on a DL VM image,
        # which already has CUDA + GPU drivers (so don't reinstall them).
        bootstrap_url = _raw_url(a.repo_url, a.repo_ref, "cloud/bootstrap.sh")
        script = f'curl -fsSL "{bootstrap_url}" -o /tmp/bootstrap.sh && bash /tmp/bootstrap.sh'
        runnable = {"script": {"text": script}, "environment": {"variables": env_vars}}
        instance_policy = {"machineType": a.machine_type,
                           "accelerators": [{"type": a.gpu_type, "count": 1}],
                           "bootDisk": {"image": a.boot_image, "sizeGb": 200}}
        install_drivers = False

    if a.spot:
        instance_policy["provisioningModel"] = "SPOT"

    job = {
        "taskGroups": [{
            "taskCount": len(units),
            "parallelism": a.max_parallel,  # <= N simultaneous VMs
            "taskSpec": {
                "computeResource": {"cpuMilli": 8000, "memoryMib": 32000},
                "maxRetryCount": 1,
                "runnables": [runnable],
            },
        }],
        "allocationPolicy": {"instances": [{"installGpuDrivers": install_drivers, "policy": instance_policy}]},
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
    job_file = local_manifest.replace(".tsv", ".json")
    with open(job_file, "w") as fh:
        json.dump(job, fh, indent=2)
    print(f"Batch job spec ({len(units)} tasks, parallelism {a.max_parallel}) -> {job_file}")
    cmd = ["gcloud", "batch", "jobs", "submit", f"{a.exp_name}-{a.stage}",
           "--location", a.region, "--config", job_file]
    if a.dry_run:
        print("DRY", " ".join(cmd))
        print(json.dumps(job, indent=2))
    else:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["local", "batch"], default="local")
    ap.add_argument("--max-parallel", dest="max_parallel", type=int, default=4,
                    help="Max simultaneous workers / GPU VMs (hard cap 8).")
    # Default to "fit": each fit job extracts its own cell's features (one ViT pass)
    # then ridge-fits. The standalone "extract" stage only pays off for datasets
    # whose stimuli are shared across neurons (cache reuse); the per-cell datasets
    # (pvc1/pvc4) need per-neuron extraction, so each fit job runs the ViT itself.
    ap.add_argument("--stage", choices=["extract", "fit", "both"], default="fit")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--max-cells", dest="max_cells", type=int, default=None,
                    help="Override neuron count per dataset (for quick tests).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input_adapt", default="resize", choices=["resize", "pad"])
    # V-JEPA: mirror-pad 10 -> 16 frames so T'=8 is divisible by the aggregator's
    # nt=4 (10 -> T'=5 is not) and to reduce RoPE temporal interpolation. Midway (2-D)
    # ignores this flag.
    ap.add_argument("--vjepa_pad_t", type=int, default=16)
    # Global token mean-pool (+4 time points) keeps the feature cache ~1 GB/cell;
    # `downsample` preserves 8x8 spatial structure but balloons the cache to ~40 GB/cell.
    ap.add_argument("--aggregator", default="average", choices=["average", "downsample", "downsample_t"])
    ap.add_argument("--pca", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--exp_name", default="vjepa_midway_fit")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--data_root", default="/data/data_derived")
    ap.add_argument("--ckpt_root", default="/data/checkpoints")
    ap.add_argument("--cache_root", default="/cache")
    ap.add_argument("--bucket", default="gs://xcorr-dev-motion-model")
    ap.add_argument("--image", default=None,
                    help="Optional container image URI for the batch backend. If omitted, "
                         "a no-Docker script runnable pulls cloud/bootstrap.sh from the repo.")
    ap.add_argument("--repo-url", dest="repo_url",
                    default="https://github.com/patrickmineault/your-head-is-there-to-move-you-around",
                    help="Repo the worker git-clones at runtime (code is pulled, never baked).")
    ap.add_argument("--repo-ref", dest="repo_ref", default="vjepa-midway-extension",
                    help="Branch/tag/commit of the repo to clone.")
    ap.add_argument("--boot-image", dest="boot_image",
                    default="projects/deeplearning-platform-release/global/images/family/common-cu129-ubuntu-2204-nvidia-580",
                    help="Boot-disk image for the no-Docker script runnable (has CUDA+python+driver).")
    ap.add_argument("--machine_type", default="g2-standard-8")
    ap.add_argument("--gpu_type", default="nvidia-l4")
    ap.add_argument("--spot", action="store_true",
                    help="Use Spot (preemptible) VMs -- cheaper and often the only "
                         "GPU capacity available; Batch retries preempted tasks.")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    a = ap.parse_args()

    if a.max_parallel > 8:
        sys.exit("refusing --max-parallel > 8 (you said you have < 8 GPU VMs)")

    units = build_manifest(a.stage, a.datasets, a.models, a.max_cells)
    print(f"{len(units)} work units across {len(a.datasets)} datasets x {len(a.models)} models")
    (run_batch if a.backend == "batch" else run_local)(units, a)


if __name__ == "__main__":
    main()
