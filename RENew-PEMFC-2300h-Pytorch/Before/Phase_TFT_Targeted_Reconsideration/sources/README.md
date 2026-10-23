# Frozen source checkouts

The local source checkouts used for engineering inspection are intentionally not committed as nested Git repositories. `../SOURCE_MANIFEST.yaml` is the authoritative provenance record.

To reconstruct the official xLSTM checkout used for the state-continuity tests, clone the exact tag and verify the detached commit and file hashes recorded in that manifest:

```powershell
git clone --branch v2.0.4 --depth 1 https://github.com/NX-AI/xlstm xlstm_official_v2_0_4
git -C xlstm_official_v2_0_4 rev-parse HEAD
```

The frozen result used commit `add5001330eb62d66867a5eca211050240ca74f5`. The Google Research and PyTorch Forecasting commits in the manifest are implementation references only; neither was executed at runtime.
