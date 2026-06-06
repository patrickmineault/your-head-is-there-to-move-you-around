"""Wrappers for modern video self-supervised ViTs (V-JEPA 2.1, Midway Network).

These models are Vision Transformers whose hidden states are token sequences
``(B, N, D)`` rather than the ``(B, C, T, H, W)`` conv feature maps the rest of the
pipeline (``Downsampler`` / ``Averager`` aggregators) expects. The wrappers here:

  * load the pretrained encoders from their vendored source repos + checkpoints,
  * adapt the input clip to the encoder's patch geometry (resize, or mirror/reflect
    pad in x/y and optionally t -- see ``InputAdapter``),
  * run the encoder while forward-hooks on the selected transformer blocks reshape
    each block's tokens back to a spatial grid so the existing aggregators work.

Both vendored repos ship a top-level ``src`` package, so only ONE of them may be on
``sys.path`` per process. ``get_feature_model`` only ever builds one model per run,
so this is fine; the loaders below insert the right root and (for Midway) shim the
``src.utils`` module to avoid pulling in its heavy ``cv2`` dependency.
"""
import collections
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import paths


# --------------------------------------------------------------------------- #
# Input adapter: guarantee patch divisibility before the encoder runs.
# --------------------------------------------------------------------------- #
class InputAdapter(nn.Module):
    """Resize or mirror-pad a clip to the encoder's native spatial size, and
    optionally mirror-pad the temporal dimension to ``target_t`` frames.

    Accepts either 4-D ``(B, C, H, W)`` (per-frame / 2-D models) or 5-D
    ``(B, C, T, H, W)`` (3-D / tubelet models) tensors.
    """

    def __init__(self, size, mode="resize", target_t=None):
        super().__init__()
        self.size = size            # target H == W (divisible by patch size)
        self.mode = mode            # "resize" | "pad"
        self.target_t = target_t    # if set, mirror-pad T up to this many frames

    def _pad_once(self, x, dim, left, right):
        # F.pad with reflect/replicate only pads the *trailing* dims, with a pad
        # vector of length 2 (last dim), 4 (4D, last 2), or 6 (5D, last 3).
        ndim = x.dim()
        d = ndim - 1 - dim  # distance of `dim` from the last axis
        window = 3 if ndim == 5 else (2 if ndim == 4 else 1)
        if d >= window:
            raise ValueError(f"cannot reflect-pad dim {dim} of a {ndim}D tensor")
        pad = [0] * (2 * window)
        pad[2 * d] = left
        pad[2 * d + 1] = right
        cur = x.shape[dim]
        mode = "reflect" if cur > 1 and left < cur and right < cur else "replicate"
        return F.pad(x, pad, mode=mode)

    def _pad_to(self, x, dim, target):
        """Mirror-pad ``x`` along ``dim`` up to length ``target`` (centered).

        Reflect padding can extend a dim by at most ``size-1`` per call, so for
        large extensions (e.g. 112 -> 384) we apply reflect repeatedly (a valid
        mirror tiling), overshoot, then center-crop to ``target``.
        """
        cur = x.shape[dim]
        if cur >= target:
            return x
        while x.shape[dim] < target:
            cur = x.shape[dim]
            step = min(cur - 1, target - cur) if cur > 1 else (target - cur)
            x = self._pad_once(x, dim, step, step)
        # center-crop any overshoot
        cur = x.shape[dim]
        if cur > target:
            start = (cur - target) // 2
            x = x.narrow(dim, start, target)
        return x

    def forward(self, x):
        is5d = x.dim() == 5
        # --- temporal mirror-pad (5-D only) ---
        if is5d and self.target_t is not None:
            x = self._pad_to(x, dim=2, target=self.target_t)

        # --- spatial sizing ---
        H = x.shape[-1]
        if H == self.size:
            return x
        if self.mode == "resize":
            if is5d:
                T = x.shape[2]
                x = F.interpolate(
                    x, size=(T, self.size, self.size),
                    mode="trilinear", align_corners=False,
                )
            else:
                x = F.interpolate(
                    x, size=(self.size, self.size),
                    mode="bilinear", align_corners=False,
                )
        elif self.mode == "pad":
            x = self._pad_to(x, dim=x.dim() - 1, target=self.size)  # W
            x = self._pad_to(x, dim=x.dim() - 2, target=self.size)  # H
        else:
            raise ValueError(f"Unknown input_adapt mode {self.mode}")
        return x


# --------------------------------------------------------------------------- #
# Grid-reshaping forward hook.
# --------------------------------------------------------------------------- #
def make_grid_hook(name, activations, wrapper):
    """Forward hook that reshapes a block's token output to a feature-map grid.

    * 3-D (V-JEPA): ``(B, T'*Hp*Wp, D)`` -> ``(B, D, T', Hp, Wp)``.
    * 2-D (Midway): ``(B*T, n_prefix + Hp*Wp, D)`` -> ``(B*T, D, Hp, Wp)`` (the
      leading CLS / register tokens are dropped).
    """

    def hook(module, inp, out):
        x = out[0] if isinstance(out, tuple) else out  # vjepa blocks return (x, attn)
        B = x.shape[0]
        D = x.shape[-1]
        Hp, Wp = wrapper.grid_hw
        spatial = Hp * Wp
        if wrapper.is_video:
            Tp = wrapper.grid_t
            assert x.shape[1] == Tp * spatial, (
                f"{name}: tokens {x.shape[1]} != T'*Hp*Wp {Tp*spatial}"
            )
            x = x.reshape(B, Tp, Hp, Wp, D).permute(0, 4, 1, 2, 3).contiguous()
        else:
            n_prefix = x.shape[1] - spatial          # CLS (+ registers)
            x = x[:, n_prefix:, :]                    # keep patch tokens only
            x = x.reshape(B, Hp, Wp, D).permute(0, 3, 1, 2).contiguous()
        activations[name] = x

    return hook


# --------------------------------------------------------------------------- #
# Wrapper modules.
# --------------------------------------------------------------------------- #
class VJEPA2Wrapper(nn.Module):
    """V-JEPA 2.1 ViT encoder (3-D tubelet). Expects ``(B, C, T, H, W)``."""

    is_video = True

    def __init__(self, encoder, adapter, patch_size, tubelet_size):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_hw = None
        self.grid_t = None

    def forward(self, x):
        x = self.adapter(x)                       # (B, C, T, H, W)
        _, _, T, H, W = x.shape
        self.grid_t = T // self.tubelet_size
        self.grid_hw = (H // self.patch_size, W // self.patch_size)
        return self.encoder(x)


class MidwayWrapper(nn.Module):
    """Midway ViT-B encoder (2-D per-frame). Expects ``(B*T, C, H, W)``."""

    is_video = False

    def __init__(self, encoder, adapter, patch_size):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.patch_size = patch_size
        self.grid_hw = None
        self.grid_t = None

    def forward(self, x):
        x = self.adapter(x)                       # (B*T, C, H, W)
        _, _, H, W = x.shape
        self.grid_hw = (H // self.patch_size, W // self.patch_size)
        # feature_levels=[] -> encoder returns (cls, []); hooks capture the blocks.
        return self.encoder(x, feature_levels=[])


# --------------------------------------------------------------------------- #
# Encoder loaders.
# --------------------------------------------------------------------------- #
def _clean_backbone_key(state_dict):
    out = {}
    for k, v in state_dict.items():
        out[k.replace("module.", "").replace("backbone.", "")] = v
    return out


def load_vjepa2_1_vitl(ckpt_path, img_size=384):
    """Build the V-JEPA 2.1 ViT-L/16-384 encoder and load ``ema_encoder`` weights."""
    if paths.VJEPA2_ROOT not in sys.path:
        sys.path.insert(0, paths.VJEPA2_ROOT)
    from app.vjepa_2_1.models import vision_transformer as vjepa_vit

    encoder = vjepa_vit.vit_large(
        patch_size=16, img_size=(img_size, img_size), num_frames=64, tubelet_size=2,
        use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
        use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(_clean_backbone_key(ckpt["ema_encoder"]), strict=True)
    encoder.eval()
    return encoder


def _install_midway_src_utils_shim():
    """Provide a minimal ``src.utils`` exposing only ``trunc_normal_`` so that
    ``src.vision_transformer`` imports without pulling in cv2 / decord."""
    if "src.utils" in sys.modules and hasattr(sys.modules["src.utils"], "trunc_normal_"):
        return
    shim = types.ModuleType("src.utils")
    shim.trunc_normal_ = nn.init.trunc_normal_
    # Ensure a parent ``src`` package exists pointing at the midway repo.
    if "src" not in sys.modules:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [f"{paths.MIDWAY_ROOT}/src"]
        sys.modules["src"] = src_pkg
    sys.modules["src.utils"] = shim


def load_midway_vitb(ckpt_path, img_size=224):
    """Build the Midway ViT-B/16 encoder and load the EMA ``teacher.backbone`` weights."""
    if paths.MIDWAY_ROOT not in sys.path:
        sys.path.insert(0, paths.MIDWAY_ROOT)
    _install_midway_src_utils_shim()
    from src import vision_transformer as midway_vit

    encoder = midway_vit.vit_base(
        patch_size=16, img_size=[img_size, img_size], num_register_tokens=0,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    teacher = ckpt["teacher"]
    backbone = {
        k.replace("module.", "").replace("backbone.", "", 1): v
        for k, v in teacher.items()
        if "backbone." in k
    }
    encoder.load_state_dict(backbone, strict=True)
    encoder.eval()
    return encoder


# --------------------------------------------------------------------------- #
# Public entry points used by models.get_feature_model.
# --------------------------------------------------------------------------- #
def build_vjepa2_1_vitl(ckpt_path, activations, layer_idxs, input_adapt="resize",
                        target_t=None, img_size=384):
    encoder = load_vjepa2_1_vitl(ckpt_path, img_size=img_size)
    adapter = InputAdapter(size=img_size, mode=input_adapt, target_t=target_t)
    wrapper = VJEPA2Wrapper(encoder, adapter, patch_size=16, tubelet_size=2)
    layers = collections.OrderedDict(
        (f"layer{i:02}", encoder.blocks[i]) for i in layer_idxs
    )
    for name, blk in layers.items():
        blk.register_forward_hook(make_grid_hook(name, activations, wrapper))
    metadata = {"sz": img_size, "threed": True}
    return wrapper, layers, metadata


def build_midway_vitb(ckpt_path, activations, layer_idxs, input_adapt="resize",
                      img_size=224):
    encoder = load_midway_vitb(ckpt_path, img_size=img_size)
    adapter = InputAdapter(size=img_size, mode=input_adapt, target_t=None)
    wrapper = MidwayWrapper(encoder, adapter, patch_size=16)
    layers = collections.OrderedDict(
        (f"layer{i:02}", encoder.blocks[i]) for i in layer_idxs
    )
    for name, blk in layers.items():
        blk.register_forward_hook(make_grid_hook(name, activations, wrapper))
    metadata = {"sz": img_size, "threed": False}
    return wrapper, layers, metadata
