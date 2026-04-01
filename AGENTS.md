# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

ComfyUI custom node wrapper for ByteDance's LatentSync 1.5 lip-sync model. Takes video frames + audio and produces lip-synced video output. Built as a fork/reimplementation of ComfyUI-Geeky-LatentSyncWrapper with improvements for long video stability, ComfyUI-standard model paths, and interruptible execution.

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# No test suite exists. Verify by loading in ComfyUI and running the nodes.
# FFmpeg must be on PATH (required at runtime).
```

## Architecture

### Node Registration (ComfyUI entry point)
- `__init__.py` — exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`
- `nodes.py` — defines both ComfyUI nodes and all orchestration logic

### Two ComfyUI Nodes
1. **`LatentSyncNode`** (`LatentSync 1.5`) — main lip-sync node. Accepts images + audio, writes temp video/audio files, invokes the inference pipeline, returns processed frames + audio.
2. **`LatentSyncVideoLengthAdjuster`** (`LatentSync Length Adjuster`) — adjusts video length to match audio via normal trim, pingpong, or loop modes.

### Inference Flow (`nodes.py` → `scripts/inference.py` → pipeline)
`nodes.py:LatentSyncNode.inference()` is the main entry point:
1. Converts ComfyUI IMAGE/AUDIO tensors to temp .mp4/.wav files
2. Delegates runtime execution via `latentsync_refactor_runtime.py` to keep the heavy pipeline isolated and reusable
3. `scripts/inference.py:main()` builds the diffusion pipeline: VAE (sd-vae-ft-mse) + Whisper audio encoder + UNet3D + DDIM scheduler
4. `latentsync/pipelines/lipsync_pipeline.py:LipsyncPipeline.__call__()` runs segmented inference — processes video in chunks of `segment_inferences * 16` frames to control memory

### Model Path Resolution (`latentsync_paths.py`)
Models stored under `ComfyUI/models/latensync1.5/` with legacy fallbacks to `latentsync1.5/` and `LatentSync-1.5/`. The `LATENTSYNC_ROOT_DIR` env var can override. HF and torch caches are redirected under the model root (`hf_cache/`, `torch_cache/`).

Required model files (auto-downloaded from HuggingFace on first run):
- `latentsync_unet.pt` — UNet checkpoint
- `whisper/tiny.pt` — Whisper audio encoder
- `auxiliary/s3fd-e19a316812.pth` — face detector (downloaded from adrianbulat.com)

### Core Library (`latentsync/`)
Vendored/adapted from ByteDance LatentSync. Key subdirectories:
- `models/` — UNet3D, attention, motion modules, SyncNet
- `pipelines/lipsync_pipeline.py` — the diffusion pipeline with segmented inference, face affine transform, and frame restoration
- `whisper/` — bundled Whisper implementation for audio feature extraction
- `utils/` — video I/O, face affine transforms, image processing, mask images (`mask.png` through `mask4.png`)

### Config (`configs/`)
- `unet/stage2.yaml` — primary inference config (256px resolution, 16 frames per chunk, cross_attention_dim=384 → whisper tiny)
- `scheduler/` — DDIM scheduler config
- `audio.yaml`, `syncnet/` — training/eval configs (not used in ComfyUI inference)

## Important Patterns

- **ComfyUI interrupt support**: `throw_if_processing_interrupted()` is called at every major loop/step across `nodes.py`, `scripts/inference.py`, and `lipsync_pipeline.py`. Must be preserved in any new loops.
- **Runtime/cache isolation**: `nodes.py` uses `latentsync_wrapper_`-prefixed temp/cache markers to avoid clashes with other LatentSync wrappers.
- **Temp file management**: Each module load creates a session temp dir under ComfyUI's temp root, cleaned up via `atexit`. Each inference run gets a unique subdirectory.
- **Conflict avoidance**: The code detects coexisting `ComfyUI-LatentSyncWrapper` and uses isolated module names, cache keys, and temp paths prefixed with `latentsync_wrapper_`.
- **`folder_paths` import**: ComfyUI's `folder_paths` module is imported via `__import__("folder_paths")` since it's only available at runtime within ComfyUI.

## Language

Code comments and print statements mix English and Chinese. README is in Chinese.
