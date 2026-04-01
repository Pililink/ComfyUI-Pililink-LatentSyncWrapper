# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature

try:
    from DeepCache import DeepCacheSDHelper
except Exception:
    DeepCacheSDHelper = None


comfy_model_management = None
try:
    comfy_model_management = __import__("comfy.model_management", fromlist=["throw_exception_if_processing_interrupted"])
except Exception:
    comfy_model_management = None


def throw_if_processing_interrupted():
    if comfy_model_management is not None:
        comfy_model_management.throw_exception_if_processing_interrupted()


def is_env_flag_enabled(*names):
    for name in names:
        if os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def maybe_enable_deepcache(pipeline, args, device_str):
    disable_by_env = is_env_flag_enabled("LATENTSYNC_DISABLE_DEEPCACHE")
    if getattr(args, "disable_deepcache", False) or disable_by_env:
        print("DeepCache disabled for this run")
        return None

    if device_str != "cuda" or not torch.cuda.is_available():
        print("DeepCache skipped: CUDA device not available")
        return None

    if DeepCacheSDHelper is None:
        print("DeepCache not installed; continuing without it")
        return None

    cache_interval = max(1, int(getattr(args, "deepcache_cache_interval", 3)))
    cache_branch_id = max(0, int(getattr(args, "deepcache_branch_id", 0)))

    # DeepCache expects a standard `unet` attribute on the pipeline.
    if not hasattr(pipeline, "unet"):
        setattr(pipeline, "unet", pipeline.denoising_unet)

    try:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=cache_interval, cache_branch_id=cache_branch_id)
        helper.enable()
        print(
            f"DeepCache enabled (cache_interval={cache_interval}, "
            f"cache_branch_id={cache_branch_id})"
        )
        return helper
    except Exception as exc:
        print(f"DeepCache initialization failed, continuing without it: {exc}")
        return None


def load_scheduler(scheduler_path, scheduler_type="ddim"):
    resolved_scheduler_type = str(scheduler_type or "ddim").strip().lower()
    if resolved_scheduler_type not in {"ddim", "dpm_solver"}:
        resolved_scheduler_type = "ddim"

    if resolved_scheduler_type == "dpm_solver":
        try:
            return DPMSolverMultistepScheduler.from_pretrained(scheduler_path)
        except Exception:
            return DPMSolverMultistepScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                solver_order=2,
                algorithm_type="dpmsolver++",
            )

    try:
        return DDIMScheduler.from_pretrained(scheduler_path)
    except Exception:
        return DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            skip_prk_steps=True,
        )


def main(config, args):
    throw_if_processing_interrupted()
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    device = getattr(args, "device", "cuda")
    device_str = device.type if hasattr(device, "type") else str(device)

    # Check if the GPU supports float16
    is_fp16_supported = (
        device_str == "cuda"
        and torch.cuda.is_available()
        and int(torch.cuda.get_device_capability()[0]) >= 7
    )
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")
    throw_if_processing_interrupted()

    # Use relative path for scheduler configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scheduler_path = os.path.join(current_dir, "..", "configs", "scheduler")
    
    # Check if scheduler directory exists
    if not os.path.exists(scheduler_path):
        print(f"Creating scheduler directory at {scheduler_path}")
        os.makedirs(scheduler_path, exist_ok=True)
        
        # Create scheduler config file if it doesn't exist
        scheduler_config_file = os.path.join(scheduler_path, "scheduler_config.json")
        config_file = os.path.join(scheduler_path, "config.json")
        
        if not os.path.exists(scheduler_config_file):
            # Default scheduler config
            scheduler_config = {
                "_class_name": "DDIMScheduler",
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "clip_sample": False,
                "num_train_timesteps": 1000,
                "set_alpha_to_one": False,
                "steps_offset": 1,
                "trained_betas": None,
                "skip_prk_steps": True
            }
            
            import json
            with open(scheduler_config_file, 'w') as f:
                json.dump(scheduler_config, f, indent=2)
            
            # Also create a copy as config.json for compatibility
            with open(config_file, 'w') as f:
                json.dump(scheduler_config, f, indent=2)
    
    print(f"Loading scheduler from: {scheduler_path}")
    scheduler_type = str(getattr(args, "scheduler_type", "ddim")).strip().lower()
    try:
        scheduler = load_scheduler(scheduler_path, scheduler_type=scheduler_type)
    except Exception as e:
        print(f"Error loading scheduler ({scheduler_type}): {e}")
        scheduler = load_scheduler(scheduler_path, scheduler_type="ddim")

    # Resolve unified root for all model artifacts.
    latentsync_root = getattr(
        args,
        "latentsync_root",
        os.environ.get("LATENTSYNC_ROOT_DIR", os.path.join(current_dir, "..", "checkpoints")),
    )

    # Use local whisper paths under unified root.
    if config.model.cross_attention_dim == 768:
        whisper_model_path = os.path.join(latentsync_root, "whisper", "small.pt")
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = os.path.join(latentsync_root, "whisper", "tiny.pt")
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device_str,
        audio_embeds_cache_dir=getattr(args, "audio_embeds_cache_dir", None),
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    audio_encoder.model.requires_grad_(False)
    audio_encoder.model.eval()
    throw_if_processing_interrupted()

    vae_model_path = getattr(args, "vae_model_path", "stabilityai/sd-vae-ft-mse")
    vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    vae.requires_grad_(False)
    vae.eval()

    denoising_unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )

    denoising_unet = denoising_unet.to(dtype=dtype)
    denoising_unet.requires_grad_(False)
    denoising_unet.eval()
    throw_if_processing_interrupted()

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
    ).to(device_str)
    throw_if_processing_interrupted()

    deepcache_helper = maybe_enable_deepcache(pipeline, args, device_str)
    throw_if_processing_interrupted()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    try:
        pipeline(
            video_path=args.video_path,
            audio_path=args.audio_path,
            video_out_path=args.video_out_path,
            video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            segment_inferences=getattr(args, "segment_inferences", 48),
            segment_overlap_clips=max(0, int(getattr(args, "segment_overlap_clips", 0))),
            affine_detect_interval=max(1, int(getattr(args, "affine_detect_interval", 1))),
            temp_dir=getattr(args, "temp_dir", "temp"),
            mask_image_path=config.data.mask_image_path,
            skip_video_normalization=getattr(args, "skip_video_normalization", False),
        )
    finally:
        if deepcache_helper is not None and hasattr(deepcache_helper, "disable"):
            try:
                deepcache_helper.disable()
                print("DeepCache disabled after inference")
            except Exception as exc:
                print(f"DeepCache cleanup warning: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--disable_deepcache", action="store_true")
    parser.add_argument("--deepcache_cache_interval", type=int, default=3)
    parser.add_argument("--deepcache_branch_id", type=int, default=0)
    parser.add_argument("--scheduler_type", type=str, default="ddim")
    parser.add_argument("--segment_overlap_clips", type=int, default=0)
    parser.add_argument("--affine_detect_interval", type=int, default=1)
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
