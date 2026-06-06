"""Shape-probe for the transformer feature extractors.

Builds one model (V-JEPA 2.1 or Midway) with real pretrained weights, pushes a
synthetic clip of the same shape the loaders produce ((B, 3, 10, 112, 112)) through
the exact code path `preprocess_data` uses, and prints, for each hooked layer, the
raw grid shape and the aggregated feature-vector shape. Run ONE model per process
(the two vendored repos both ship a top-level `src` package).

Usage:
    python tools/probe_models.py vjepa2_1_vitl --input_adapt resize --vjepa_pad_t 0
    python tools/probe_models.py midway_bdd_vitb --input_adapt pad
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_feature_model, get_aggregator, resize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", choices=["vjepa2_1_vitl", "midway_bdd_vitb"])
    ap.add_argument("--input_adapt", default="resize", choices=["resize", "pad"])
    ap.add_argument("--vjepa_pad_t", default=0, type=int)
    ap.add_argument("--aggregator", default="downsample_t")
    ap.add_argument("--aggregator_sz", default=8, type=int)
    ap.add_argument("--ckpt_root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "checkpoints"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", default=2, type=int)
    ap.add_argument("--ntau", default=10, type=int)
    cli = ap.parse_args()

    args = SimpleNamespace(
        features=cli.features,
        ckpt_root=cli.ckpt_root,
        input_adapt=cli.input_adapt,
        vjepa_pad_t=cli.vjepa_pad_t,
        subsample_layers=True,
        aggregator=cli.aggregator,
        aggregator_sz=cli.aggregator_sz,
        device=cli.device,
        resize=112,
    )

    model, activations, metadata = get_feature_model(args)
    metadata["sz"] = args.resize  # mirror train_convex.main override
    model.to(device=args.device)
    aggregator = get_aggregator(metadata, args)

    B, T = cli.batch, cli.ntau
    X = torch.randn(B, 3, T, 112, 112, device=args.device)
    print(f"\n=== {cli.features} | input_adapt={cli.input_adapt} | vjepa_pad_t={cli.vjepa_pad_t} ===")
    print(f"raw clip: {tuple(X.shape)}  threed={metadata['threed']}  model sz={metadata['sz']}")

    with torch.no_grad():
        X = resize(X, metadata["sz"])
        if metadata["threed"]:
            _ = model(X)
            for layer in activations:
                al = activations[layer]
                agg = aggregator(al)
                print(f"  {layer}: grid {tuple(al.shape)} -> agg {tuple(agg.shape)}")
        else:
            xr = X.permute(0, 2, 1, 3, 4).reshape(-1, X.shape[1], X.shape[3], X.shape[4])
            _ = model(xr)
            for layer in activations:
                fit = activations[layer]
                fit = fit.reshape(X.shape[0], X.shape[2], *fit.shape[1:])
                fit = fit.permute(0, 2, 1, 3, 4)
                agg = aggregator(fit)
                print(f"  {layer}: per-frame {tuple(activations[layer].shape)} "
                      f"-> grid {tuple(fit.shape)} -> agg {tuple(agg.shape)}")
    print("OK")


if __name__ == "__main__":
    main()
