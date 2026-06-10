# Build triggers

Each new commit to `main` starts an automated build on RunPod. The
resulting image is pushed to
`registry.runpod.net/smitp-vibe3d-r2s-worker-main-dockerfile:<first 9 hex chars of SHA>`.
The RunPod Serverless endpoint `n3j6lepsb4ym6e` is configured to pull
from this image stream — but the template must be **rebound** to the
new SHA so workers actually pull it.

**The standard deploy flow**

1. Commit + push to `main` on `smitp/vibe3d-r2s-worker`.
2. Wait for RunPod to build the image (typically 5–15 min; can be
   longer on a cold cache). The build runs on RunPod's own
   infrastructure, not via GitHub Actions — there is no workflow
   file in this repo.
3. Re-bind the template to the new SHA via the `saveTemplate`
   GraphQL mutation:
   ```bash
   SHA=$(git log --format=%H -1 | head -c 9)
   IMAGE="registry.runpod.net/smitp-vibe3d-r2s-worker-main-dockerfile:${SHA}"
   curl -X POST "https://api.runpod.io/graphql" \
     -H "Authorization: Bearer $RUNPOD_API_KEY" \
     -H "Content-Type: application/json" \
     -d "$(cat <<EOF
   { "query": "mutation { saveTemplate(input: { id: \"88j9olsbiw\", imageName: \"${IMAGE}\", containerDiskInGb: 20, dockerArgs: \"\", env: [], name: \"vibe3d-r2s-worker\", volumeInGb: 0 }) { id imageName } }" }
   EOF
   )"
   ```
4. Restart the worker pool so a fresh worker pulls the new image:
   `workersMax: 0`, sleep 10s, `workersMax: 1`. Without this step,
   the existing worker keeps running the previous image.

**If a new commit doesn't result in a healthy worker**

1. Check the RunPod console's **Serverless → Endpoint → Builds** tab
   to see if the build is queued, in progress, or failed. The API
   does not expose build status.
2. If the build is in progress, wait — the previous worker will
   keep failing until the new image lands. `unhealthy: 1, ready: 0`
   is the expected state during a build.
3. As a last resort, switch to a Pod with `serve.py` (see
   `README.md` §"Deploy to a RunPod Pod").

**Why this file exists**

RunPod's GitHub integration auto-builds on each commit but does not
auto-rebind the template to the new image — the endpoint keeps
serving the last image it knew about until you call `saveTemplate`.
The endpoint version counter increments on every `saveTemplate`
call, which can be misleading: it tracks template rebinding, not
image builds.

**Last-known-good image**

| Commit     | Image tag                          | Status        |
|------------|------------------------------------|---------------|
| `9eb3899`  | `…-main-dockerfile:9eb38998f`      | ran, missing descartes |
| `790ea25`  | `…-main-dockerfile:790ea256`       | build stuck, unhealthy  |
| `ca2e870`  | `…-main-dockerfile:ca2e870f`       | `TypeError: got multiple values for 'semantic_classes'` in `Namespace(**)` splat |
| `4154e64`  | `…-main-dockerfile:4154e64a`       | new `pos_embed` patch for numpy 2.x |
| `0826b26`  | `…-main-dockerfile:0826b26a0`      | BUILD_TRIGGER.md reverted to match actual workflow |
| `00d1ab2`  | `…-main-dockerfile:00d1ab2e2`      | numpy 2.x pos_embed `TypeError: expected np.ndarray` |
| `e627eb7`  | `…-main-dockerfile:e627eb712`      | `class_embed` size mismatch: ckpt has 3, model had 4 (add_cls_token) |
| `<new>`    | `…-main-dockerfile:<9-hex SHA>`    | set `add_cls_token=False` to match cc5k checkpoint |
