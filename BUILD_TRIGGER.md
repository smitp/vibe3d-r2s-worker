# Build triggers

RunPod's Serverless GitHub integration only builds on **GitHub
Releases**, not on commits. To push a new code change to a running
endpoint you must:

1. Push the commit to `main`.
2. **Create a GitHub release** on the latest commit:
   ```bash
   git tag v0.0.N-short-desc
   git push origin v0.0.N-short-desc
   gh release create v0.0.N-short-desc \
     --repo smitp/vibe3d-r2s-worker \
     --title "v0.0.N — short desc" \
     --notes "..." \
     --target <full SHA>
   ```
3. The release triggers a build in the bound RunPod GitHub
   integration. Build takes ~5–10 min for a fresh image.
4. RunPod pushes the resulting image to
   `registry.runpod.net/smitp-vibe3d-r2s-worker-main-dockerfile:<first 9 hex chars of SHA>`.
5. Re-bind the template to the new image:
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
6. Restart the worker pool (set `workersMax: 0`, sleep 10s, set
   `workersMax: 1`) so a fresh worker pulls the new image.

**Why this file exists**

A new release is the only event that triggers a build on a bound
endpoint. Re-binding the template to a SHA without a release gives
no new image — the worker pulls nothing useful and goes unhealthy.
The RunPod endpoint version counter still increments on every
`saveTemplate` call, which is misleading: it tracks the template
rebinding, not the image build.

**The two-step gotcha**

The release build can take 5–10 min. If you re-bind the template
to a SHA whose image is still being built, the worker will pull
nothing, fail to start, and be marked `unhealthy` after ~60s.
Wait for the build to land in the registry before re-binding.

**Last-known-good image**

| Commit     | Image tag                          | Status        |
|------------|------------------------------------|---------------|
| `9eb3899`  | `…-main-dockerfile:9eb38998f`      | ran, missing descartes |
| `790ea25`  | `…-main-dockerfile:790ea256`       | build stuck, unhealthy  |
| `ca2e870`  | `…-main-dockerfile:ca2e870f`       | `TypeError: got multiple values for 'semantic_classes'` in `Namespace(**)` splat |
| `4154e64`  | `…-main-dockerfile:4154e64a`       | new `pos_embed` patch for numpy 2.x — **release `v0.0.4-numpy2-pos-embed-fix` created on 2026-06-10 to trigger build** |
