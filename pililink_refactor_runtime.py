import os
import threading

import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
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


_RUNTIME_CACHE = {}
_RUNTIME_CACHE_LOCK = threading.Lock()


def _safe_cache_stamp(path):
    absolute_path = os.path.abspath(path)
    try:
        stat_info = os.stat(absolute_path)
        mtime_ns = getattr(stat_info, "st_mtime_ns", int(stat_info.st_mtime * 1_000_000_000))
        return f"{absolute_path}|{stat_info.st_size}|{mtime_ns}"
    except OSError:
        return f"{absolute_path}|missing"


def _load_scheduler(scheduler_path):
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


def _resolve_whisper_path(latentsync_root, config):
    cross_attention_dim = int(config.model.cross_attention_dim)
    if cross_attention_dim == 768:
        return os.path.join(latentsync_root, "whisper", "small.pt")
    if cross_attention_dim == 384:
        return os.path.join(latentsync_root, "whisper", "tiny.pt")
    raise NotImplementedError("cross_attention_dim must be 768 or 384")


def _maybe_enable_deepcache(pipeline, deepcache, cache_interval, cache_branch_id, device_str):
    if str(deepcache or "on").lower() == "off":
        return None

    disable_by_env = os.environ.get("PILILINK_DISABLE_DEEPCACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if disable_by_env:
        return None

    if device_str != "cuda" or not torch.cuda.is_available() or DeepCacheSDHelper is None:
        return None

    if not hasattr(pipeline, "unet"):
        setattr(pipeline, "unet", pipeline.denoising_unet)

    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(cache_interval=max(1, int(cache_interval)), cache_branch_id=max(0, int(cache_branch_id)))
    helper.enable()
    return helper


def _is_cuda_oom_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "cublas_status_alloc_failed" in message or "out of memory" in message


def _build_clip_batch_candidates(requested_batch_size, auto_oom_fallback):
    requested_batch_size = max(1, int(requested_batch_size))
    if not auto_oom_fallback or requested_batch_size == 1:
        return [requested_batch_size]

    candidates = [requested_batch_size]
    current = requested_batch_size
    while current > 1:
        current = max(1, current // 2)
        if current not in candidates:
            candidates.append(current)
    return candidates


class _RefactorRuntime:
    def __init__(
        self,
        *,
        config,
        scheduler_path,
        inference_ckpt_path,
        whisper_model_path,
        audio_embeds_cache_dir,
        vae_model_path,
        device_str,
        dtype,
    ):
        throw_if_processing_interrupted()

        scheduler = _load_scheduler(scheduler_path)

        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device=device_str,
            audio_embeds_cache_dir=audio_embeds_cache_dir,
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )
        audio_encoder.model.requires_grad_(False)
        audio_encoder.model.eval()

        resolved_vae_model_path = vae_model_path if os.path.exists(vae_model_path) else "stabilityai/sd-vae-ft-mse"
        vae = AutoencoderKL.from_pretrained(resolved_vae_model_path, torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        vae.requires_grad_(False)
        vae.eval()

        denoising_unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            inference_ckpt_path,
            device="cpu",
        )
        denoising_unet = denoising_unet.to(dtype=dtype)
        denoising_unet.requires_grad_(False)
        denoising_unet.eval()

        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        ).to(device_str)

        self.config = config
        self.device_str = device_str
        self.dtype = dtype
        self.pipeline = pipeline
        self.run_lock = threading.Lock()

    def run(
        self,
        *,
        video_path,
        audio_path,
        output_video_path,
        seed,
        inference_steps,
        guidance_scale,
        segment_inferences,
        temp_dir,
        mask_image_path,
        deepcache,
        deepcache_cache_interval,
        deepcache_branch_id,
        skip_video_normalization,
        clip_batch_size,
        auto_oom_fallback,
        quality_mode,
    ):
        with self.run_lock:
            throw_if_processing_interrupted()

            if seed != -1:
                set_seed(seed)
            else:
                torch.seed()

            resolved_quality_mode = str(quality_mode or "balanced").lower()
            if resolved_quality_mode not in {"balanced", "quality_first"}:
                resolved_quality_mode = "balanced"

            target_clip_batch_size = max(1, int(clip_batch_size))
            target_deepcache = deepcache
            if resolved_quality_mode == "quality_first":
                target_clip_batch_size = 1
                target_deepcache = "off"

            clip_batch_candidates = _build_clip_batch_candidates(
                target_clip_batch_size,
                bool(auto_oom_fallback),
            )
            print(
                "Pililink refactor runtime:",
                f"device={self.device_str}",
                f"dtype={self.dtype}",
                f"quality_mode={resolved_quality_mode}",
                f"clip_batch_candidates={clip_batch_candidates}",
            )

            previous_cudnn_benchmark = None
            previous_matmul_tf32 = None
            previous_cudnn_tf32 = None
            try:
                if self.device_str == "cuda" and torch.cuda.is_available():
                    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
                    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
                    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
                    if resolved_quality_mode == "quality_first":
                        torch.backends.cudnn.benchmark = False
                        torch.backends.cuda.matmul.allow_tf32 = False
                        torch.backends.cudnn.allow_tf32 = False

                last_exc = None
                for candidate_index, current_clip_batch_size in enumerate(clip_batch_candidates):
                    deepcache_helper = None
                    try:
                        deepcache_helper = _maybe_enable_deepcache(
                            self.pipeline,
                            target_deepcache,
                            deepcache_cache_interval,
                            deepcache_branch_id,
                            self.device_str,
                        )

                        with torch.inference_mode(False):
                            self.pipeline(
                                video_path=video_path,
                                audio_path=audio_path,
                                video_out_path=output_video_path,
                                video_mask_path=output_video_path.replace(".mp4", "_mask.mp4"),
                                num_frames=self.config.data.num_frames,
                                num_inference_steps=inference_steps,
                                guidance_scale=guidance_scale,
                                weight_dtype=self.dtype,
                                width=self.config.data.resolution,
                                height=self.config.data.resolution,
                                segment_inferences=segment_inferences,
                                temp_dir=temp_dir,
                                mask_image_path=mask_image_path,
                                skip_video_normalization=bool(skip_video_normalization),
                                clear_cuda_cache_per_segment=(
                                    resolved_quality_mode == "quality_first" or current_clip_batch_size <= 1
                                ),
                                clip_batch_size=current_clip_batch_size,
                            )
                        return
                    except Exception as exc:
                        last_exc = exc
                        has_next_candidate = candidate_index < len(clip_batch_candidates) - 1
                        if has_next_candidate and _is_cuda_oom_error(exc):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise
                    finally:
                        if deepcache_helper is not None and hasattr(deepcache_helper, "disable"):
                            try:
                                deepcache_helper.disable()
                            except Exception:
                                pass

                if last_exc is not None:
                    raise last_exc
            finally:
                if self.device_str == "cuda" and torch.cuda.is_available():
                    if previous_cudnn_benchmark is not None:
                        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
                    if previous_matmul_tf32 is not None:
                        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
                    if previous_cudnn_tf32 is not None:
                        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def run_refactor_inference(
    *,
    config_path,
    scheduler_path,
    inference_ckpt_path,
    latentsync_root,
    audio_embeds_cache_dir,
    vae_model_path,
    video_path,
    audio_path,
    output_video_path,
    seed,
    inference_steps,
    guidance_scale,
    segment_inferences,
    temp_dir,
    device,
    dtype,
    mask_image_path,
    deepcache,
    deepcache_cache_interval,
    deepcache_branch_id,
    skip_video_normalization,
    clip_batch_size,
    auto_oom_fallback,
    quality_mode,
):
    throw_if_processing_interrupted()

    config = OmegaConf.load(config_path)
    whisper_model_path = _resolve_whisper_path(latentsync_root, config)

    device_str = device.type if hasattr(device, "type") else str(device)
    if device_str != "cuda":
        dtype = torch.float32

    cache_key = (
        device_str,
        str(dtype),
        _safe_cache_stamp(config_path),
        _safe_cache_stamp(scheduler_path),
        _safe_cache_stamp(inference_ckpt_path),
        _safe_cache_stamp(whisper_model_path),
        _safe_cache_stamp(vae_model_path),
        os.path.abspath(audio_embeds_cache_dir) if audio_embeds_cache_dir else "",
    )

    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(cache_key)
        if runtime is None:
            runtime = _RefactorRuntime(
                config=config,
                scheduler_path=scheduler_path,
                inference_ckpt_path=inference_ckpt_path,
                whisper_model_path=whisper_model_path,
                audio_embeds_cache_dir=audio_embeds_cache_dir,
                vae_model_path=vae_model_path,
                device_str=device_str,
                dtype=dtype,
            )
            _RUNTIME_CACHE[cache_key] = runtime

    runtime.run(
        video_path=video_path,
        audio_path=audio_path,
        output_video_path=output_video_path,
        seed=seed,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        segment_inferences=segment_inferences,
        temp_dir=temp_dir,
        mask_image_path=mask_image_path,
        deepcache=deepcache,
        deepcache_cache_interval=deepcache_cache_interval,
        deepcache_branch_id=deepcache_branch_id,
        skip_video_normalization=skip_video_normalization,
        clip_batch_size=clip_batch_size,
        auto_oom_fallback=auto_oom_fallback,
        quality_mode=quality_mode,
    )
