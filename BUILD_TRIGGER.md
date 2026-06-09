# Build triggers

Each new commit to `main` produces a new image tag in the form
`smitp-vibe3d-r2s-worker-main-dockerfile:<first 9 hex chars of SHA>`.
The RunPod Serverless endpoint `n3j6lepsb4ym6e` is configured to pull
from this image stream.

**Why this file exists**

RunPod's GitHub integration does not always auto-pick up new commits
on a bound template. The `saveTemplate` GraphQL mutation (see the
project's `ARCHITECTURE_v4.md` §3) can re-bind the template to a
new image tag, but the **build itself** is also keyed on the commit
SHA — so a fresh commit is the trigger, not a comment-only change.

If a new commit doesn't immediately result in a healthy worker:

1. The `saveTemplate` mutation may need to be re-run (see
   `docs/runpod-rebuild.md` for the exact curl).
2. The RunPod console's **Logs** tab for the endpoint will show the
   actual build / runtime error — the API does not expose worker logs.
3. As a last resort, switch to a Pod with `serve.py` (see
   `README.md` §"Deploy to a RunPod Pod").

**Last-known-good image**

| Commit  | Image tag                          | Status        |
|---------|------------------------------------|---------------|
| `9eb3899` | `…-main-dockerfile:9eb38998f`    | ran, missing descartes |
| `790ea25` | `…-main-dockerfile:790ea256`     | build stuck, unhealthy  |
