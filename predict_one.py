"""Predict a single floor plan image using Raster2Seq.

This is a thin, single-image wrapper around Raster2Seq's `predict.py` (which
only takes a directory). The handler in `handler.py` writes the uploaded
image to a temp dir, runs this script, and parses the resulting
`pred_polygons.json`.

Why a script and not in-process?
- Keeps the handler < 100 LOC (handler is just IO plumbing).
- This script can be smoke-tested locally with `python predict_one.py
  /path/to/image.png cubicasa5k` without needing runpod.
- Lets us reuse Raster2Seq's `engine.generate` and `DiscreteTokenizer`
  with the exact same flags the upstream `tools/predict_cc5k.sh` uses.

Output: writes a single JSON file to the path given in --output:
    {
        "checkpoint": "cubicasa5k",
        "image_sha256": "<hex>",
        "polygons": [
            {"label": "kitchen", "vertices": [[0.1, 0.2], [0.3, 0.4], ...]},
            ...
        ],
        "openings": [
            {"type": "door", "polygon_idx": 0, "edge_idx": 2, "width_m": 0.9},
            ...
        ]
    }

This is the same schema as `floorplan_to_3d.r2s_parser.cache.PolygonSeq`,
so the postprocess on the client side is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Allow imports of the vendored Raster2Seq repo.
R2S_REPO = Path(__file__).parent / "vendor" / "Raster2Seq"
sys.path.insert(0, str(R2S_REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from detectron2.data import transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from raster2seq_hub import resolve_checkpoint_path  # noqa: E402

from datasets.transforms import ResizeAndPad  # noqa: E402
from datasets.discrete_tokenizer import DiscreteTokenizer  # noqa: E402
from engine import generate  # noqa: E402
from models import build_model  # noqa: E402
from util.plot_utils import CC5K_LABEL, S3D_LABEL  # noqa: E402

# Mirrors r2s_parser/config.py:CheckpointAlias. Keep in sync.
CHECKPOINT_TO_DATASET = {
    "cubicasa5k":      {"dataset": "cubicasa", "semantic_classes": 12, "door_window_index": [10, 9],  "label_map": CC5K_LABEL},
    "s3d-bw":          {"dataset": "stru3d",   "semantic_classes": 19, "door_window_index": [16, 17], "label_map": S3D_LABEL},
    "raster2graph":    {"dataset": "stru3d",   "semantic_classes": -1, "door_window_index": [],      "label_map": None},
    "raster2graph-512": {"dataset": "stru3d",  "semantic_classes": -1, "door_window_index": [],      "label_map": None},
    "s3d-density":     {"dataset": "stru3d",   "semantic_classes": 19, "door_window_index": [16, 17], "label_map": S3D_LABEL},
}

# Mirrors tools/predict_cc5k.sh + the upstream predict.py argparser defaults.
# The upstream script gets these for free because it goes through
# `parser.parse_args()`. Our predict_one.py builds the Namespace directly
# from this dict, so we have to enumerate every default the upstream
# argparser would have provided.
#
# History of attributes that surfaced as AttributeError on `Namespace`:
#   - `position_embedding`    (commit 5bf5ec4 — position_encoding.py:101)
#   - `lr_backbone`           (post-5bf5ec4 build — backbone.py:128)
# Rather than wait for each next one, this is the full set of upstream
# defaults copied verbatim. Boolean `store_true` flags are stored as
# `False` here (their default before --flag is passed). Values that
# tools/predict_cc5k.sh explicitly sets override these defaults and
# are still correct (poly2seq=True, dec_attn_concat_src=True, etc.).
CC5K_PREDICT_FLAGS = dict(
    # top-level
    batch_size=10,
    debug=False,
    input_channels=3,
    image_norm=False,
    eval_every_epoch=20,
    ckpt_every_epoch=20,
    label_smoothing=0.0,
    ignore_index=-1,
    image_size=256,
    ema4eval=True,
    measure_time=False,
    disable_sampling_cache=False,
    use_anchor=True,
    drop_wd=False,
    plot_text=False,
    image_scale=2,
    one_color=False,
    crop_white_space=False,
    # refinement
    refinement=False,
    refinement_threshold=0.5,
    # raster2seq
    poly2seq=True,
    seq_len=512,
    num_bins=32,
    pre_decoder_pos_embed=False,
    learnable_dec_pe=False,
    dec_qkv_proj=False,
    dec_attn_concat_src=True,
    per_token_sem_loss=True,
    # add_cls_token=False to match the upstream predict_cc5k.sh invocation
    # — the cubicasa5k checkpoint was trained without the <cls> token, so
    # `class_embed` outputs 3 classes (<coord>/<sep>/<eos>). Setting this to
    # True creates a 4-class head that the checkpoint cannot load.
    add_cls_token=False,
    # backbone
    backbone="resnet50",
    lr_backbone=0,
    dilation=False,
    position_embedding="sine",
    position_embedding_scale=2 * np.pi,
    num_feature_levels=4,
    # Transformer
    enc_layers=6,
    dec_layers=6,
    dim_feedforward=1024,
    hidden_dim=256,
    dropout=0.1,
    nheads=8,
    num_queries=800,
    num_polys=20,
    dec_n_points=4,
    enc_n_points=4,
    query_pos_type="sine",
    # with_poly_refine=False to match the upstream predict_cc5k.sh
    # invocation — the script passes --disable_poly_refine, which flips
    # the argparse default of True to False. The cc5k checkpoint was
    # trained without iterative polygon refinement, so its state_dict
    # lacks the per-layer clones (with_poly_refine=True would create
    # `_get_clones`-style duplicate class_embed/coords_embed heads and
    # the checkpoint's `class_embed.X` keys would not line up).
    with_poly_refine=False,
    masked_attn=False,
    # NOTE: `semantic_classes`, `dataset`, `dataset_name`, and
    # `label_map` are NOT set here — they come from
    # CHECKPOINT_TO_DATASET[args.checkpoint] and get splatted into
    # the Namespace at line ~210 alongside the predict_flags dict.
    # If we set them here, the splat raises
    # `TypeError: got multiple values for keyword argument`.
    disable_poly_refine=True,
    # aux
    aux_loss=True,  # `no_aux_loss` is the store_true that flips it
    # dataset parameters
    dataset_name="cubicasa",  # overridden by cfg["dataset"] for cubicasa5k
    dataset_root="",
    eval_set="test",
    # misc
    device="cuda",
    num_workers=2,
    seed=42,
    checkpoint="",
    output_dir="",
    # visualization
    plot_pred=True,
    plot_density=True,
    plot_gt=False,
    save_pred=False,
)

IMAGE_SCALE = 2  # matches predict_cc5k.sh — output is 512x512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image Raster2Seq inference")
    parser.add_argument("image", type=Path, help="Input PNG/JPG image")
    parser.add_argument(
        "checkpoint",
        choices=list(CHECKPOINT_TO_DATASET.keys()),
        help="Raster2Seq checkpoint alias",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the polygon JSON"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2

    cfg = CHECKPOINT_TO_DATASET[args.checkpoint]
    predict_flags = CC5K_PREDICT_FLAGS  # only cc5k supported in Week 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available, this will be very slow", file=sys.stderr)

    # ---- Load and transform the single image -------------------------------
    t0 = time.perf_counter()
    pil = Image.open(args.image).convert("RGB")
    arr = np.array(pil)
    aug = T.AugmentationList([ResizeAndPad((predict_flags["image_size"], predict_flags["image_size"]), pad_value=255)])
    aug_input = T.AugInput(arr)
    _ = aug(aug_input)
    # explicit dtype=float32 — torch.as_tensor on numpy 2.x with a
    # uint8 array (and especially a uint8 scalar from detectron2's
    # AugInput.image) hits `Could not infer dtype of numpy.uint8`.
    # See https://github.com/pytorch/pytorch/issues/120616 and
    # the numpy 2.0 NPY_NEP 50 changes that affect torch 2.1.
    img_t = torch.as_tensor(aug_input.image.transpose((2, 0, 1)), dtype=torch.float32)[None] / 255.0  # (1, C, H, W)
    img_t = img_t.to(device)

    # ---- Build the model and load the checkpoint --------------------------
    tokenizer = DiscreteTokenizer(
        predict_flags["num_bins"], predict_flags["seq_len"], add_cls=predict_flags["add_cls_token"]
    )
    # Convert predict_flags to a Namespace for build_model's signature.
    model_args = argparse.Namespace(**predict_flags, **cfg, vocab_size=len(tokenizer))
    model = build_model(model_args, train=False, tokenizer=tokenizer).to(device)
    ckpt_path = resolve_checkpoint_path(f"hf:{args.checkpoint}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("ema", checkpoint["model"])
    # strip "module." prefix if present
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARN: missing keys ({len(missing)}): {missing[:3]}…", file=sys.stderr)
    if unexpected:
        print(f"WARN: unexpected keys ({len(unexpected)}): {unexpected[:3]}…", file=sys.stderr)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ---- Run inference ----------------------------------------------------
    t_load = time.perf_counter()
    with torch.inference_mode():
        outputs = generate(
            model,
            img_t,
            semantic_rich=cfg["semantic_classes"] > 0,
            use_cache=True,
            per_token_sem_loss=predict_flags["per_token_sem_loss"],
            drop_wd=False,
            poly2seq=True,
        )

    # ---- Convert the model outputs to our polygon-JSON schema -------------
    polygons_out: list[dict] = []
    openings_out: list[dict] = []
    pred_rooms = outputs["room"]   # list[list[polygon]], one per image
    pred_labels = outputs["labels"]  # list[list[int]] or None

    img_h_px, img_w_px = pil.size[1], pil.size[0]  # (W, H) → use H, W for y/x
    # The model emits polygons in [0, 1] normalized coords (the image_scale
    # is only used for visualization, not for the polygon coords themselves).
    # NOTE: when the upstream `predict.py` uses --crop_white_space, the
    # polygons are cropped too. We don't crop in the single-image path
    # because the user controls what they upload.

    for room_polys, room_labels in zip(pred_rooms, pred_labels):
        if room_labels is None:
            room_labels = [-1] * len(room_polys)
        for poly, cls_idx in zip(room_polys, room_labels):
            if poly is None or len(poly) < 3:
                continue
            # `poly` is a (N, 2) array of (x, y) normalized coords.
            verts = [[float(x), float(y)] for x, y in poly]
            # Map class index → semantic label. CC5K_LABEL is a dict.
            label = cfg["label_map"].get(int(cls_idx), "room") if cfg["label_map"] else "room"
            polygons_out.append({"label": str(label), "vertices": verts})

    # NOTE: the v1.0 predict.py we wrap does NOT emit openings. They are
    # encoded implicitly as polygon edges with semantic class = door/window.
    # We extract them here by walking the model outputs again — but the
    # current `engine.generate` does not return openings separately. For
    # Week 1 we emit polygons only and rely on the existing v3 CV parser
    # for door/window detection on top. This is documented in README.
    # TODO(v4-week-2): wire the openings extraction once we have a
    # cloud GPU box to iterate on.

    image_sha256 = hashlib.sha256(args.image.read_bytes()).hexdigest()
    result = {
        "checkpoint": args.checkpoint,
        "image_sha256": image_sha256,
        "polygons": polygons_out,
        "openings": openings_out,  # always [] in v1.0
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    elapsed = time.perf_counter() - t0
    print(
        f"OK: {len(polygons_out)} polygons, {len(openings_out)} openings "
        f"(load {t_load - t0:.1f}s, infer {time.perf_counter() - t_load:.1f}s, total {elapsed:.1f}s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
