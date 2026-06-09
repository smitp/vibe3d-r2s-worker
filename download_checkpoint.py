"""Pre-download the cubicasa5k Raster2Seq checkpoint at build time.

The Raster2Seq cubicasa5k weights are public on the HF Hub (Cornell's
`haopt/Raster2Seq` mirror).  We download them at image-build time so
the first cold-start inference call doesn't pay the 4 GB download.

HF_TOKEN is read from the build env (set it in the RunPod template's
`Env Vars` if you want to bypass HF rate limits; otherwise the download
is anonymous).
"""

from __future__ import annotations

import os

from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
ckpt = hf_hub_download(
    repo_id="haopt/Raster2Seq",
    filename="cubicasa5k.pth",
    cache_dir="/opt/hf_cache",
    token=token,
)
print("downloaded to", ckpt)
