import os

RAW_DATA = "/mnt/e/data_derived/"
DERIVED_DATA = "/mnt/e/data_derived/"
CHECKPOINTS = "/mnt/d/Documents/dorsalnet/pretrained/"
CPC_DPC = "/mnt/e/Documents/ventral-dorsal-model/Models/CPC/dpc"
CPC_BACKBONE = "/mnt/e/Documents/ventral-dorsal-model/Models/CPC/backbone"

# Vendored source for the modern video ViTs (V-JEPA 2.1, Midway Network).
# Override with env vars on cloud workers where the repos live elsewhere.
_REPO_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VJEPA2_ROOT = os.environ.get("VJEPA2_ROOT", os.path.join(_REPO_PARENT, "third_party", "vjepa2"))
MIDWAY_ROOT = os.environ.get("MIDWAY_ROOT", os.path.join(_REPO_PARENT, "third_party", "midway-network"))

# Checkpoint filenames (resolved under train_convex.py --ckpt_root).
VJEPA2_1_VITL_CKPT = "vjepa2_1_vitl_dist_vitG_384.pt"
MIDWAY_BDD_VITB_CKPT = "midway-bdd-vit-b-ep300.pth"
