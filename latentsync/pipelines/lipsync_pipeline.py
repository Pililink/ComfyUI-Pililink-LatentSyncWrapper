# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import (
    check_ffmpeg_installed,
    close_ffmpeg_video_pipe_writer,
    get_ffmpeg_video_encode_args,
    get_preferred_ffmpeg_video_codec,
    get_video_frame_count,
    iter_video_cv2,
    mux_video_audio_stream_copy,
    open_ffmpeg_video_pipe_writer,
    prepare_video_for_processing,
    read_audio,
    write_video,
)
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

comfy_model_management = None
try:
    comfy_model_management = __import__("comfy.model_management", fromlist=["throw_exception_if_processing_interrupted"])
except Exception:
    comfy_model_management = None


def throw_if_processing_interrupted():
    if comfy_model_management is not None:
        comfy_model_management.throw_exception_if_processing_interrupted()


def wait_for_future_with_interrupt(future):
    while not future.done():
        throw_if_processing_interrupted()
        time.sleep(0.1)
    return future.result()

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        denoising_unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(denoising_unet.config, "_diffusers_version") and version.parse(
            version.parse(denoising_unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = (
            hasattr(denoising_unet.config, "sample_size") and denoising_unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(denoising_unet.config)
            new_config["sample_size"] = 64
            denoising_unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.denoising_unet, "_hf_hook"):
            return self.device
        for module in self.denoising_unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        if mask.dim() == 5:
            batch_size, num_frames, _, _, _ = mask.shape
            mask = rearrange(mask, "b f c h w -> (b f) c h w")
            masked_image = rearrange(masked_image, "b f c h w -> (b f) c h w")
        else:
            batch_size = 1
            num_frames = mask.shape[0]

        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        mask = rearrange(mask, "(b f) c h w -> b c f h w", b=batch_size, f=num_frames)
        masked_image_latents = rearrange(
            masked_image_latents,
            "(b f) c h w -> b c f h w",
            b=batch_size,
            f=num_frames,
        )

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        if images.dim() == 5:
            batch_size, num_frames, _, _, _ = images.shape
            images = rearrange(images, "b f c h w -> (b f) c h w")
        else:
            batch_size = 1
            num_frames = images.shape[0]

        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(
            image_latents,
            "(b f) c h w -> b c f h w",
            b=batch_size,
            f=num_frames,
        )
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    @staticmethod
    def temporal_blend_frames(previous_frames: np.ndarray, current_frames: np.ndarray):
        blend_len = min(len(previous_frames), len(current_frames))
        if blend_len <= 0:
            return np.empty((0,), dtype=np.uint8)

        previous = previous_frames[:blend_len].astype(np.float32)
        current = current_frames[:blend_len].astype(np.float32)
        if blend_len == 1:
            alpha = np.array([0.5], dtype=np.float32)
        else:
            alpha = np.linspace(0.0, 1.0, blend_len, dtype=np.float32)
        alpha = alpha.reshape(blend_len, 1, 1, 1)
        blended = previous * (1.0 - alpha) + current * alpha
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def affine_transform_video(self, video_frames: np.ndarray, affine_detect_interval: int = 1):
        affine_detect_interval = max(1, int(affine_detect_interval))
        faces = []
        boxes = []
        affine_matrices = []
        last_good_face = None
        last_good_box = None
        last_good_affine_matrix = None
        failed_indices = []
        reused_indices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for idx, frame in enumerate(tqdm.tqdm(video_frames)):
            throw_if_processing_interrupted()
            should_detect = last_good_affine_matrix is None or (idx % affine_detect_interval == 0)

            if not should_detect:
                try:
                    reused_face, reused_box = self.image_processor.warp_face_with_affine_matrix(
                        frame, last_good_affine_matrix
                    )
                    faces.append(reused_face)
                    boxes.append(reused_box)
                    affine_matrices.append(last_good_affine_matrix.copy())
                    last_good_face = reused_face
                    last_good_box = reused_box
                    reused_indices.append(idx)
                    continue
                except Exception:
                    should_detect = True

            try:
                face, box, affine_matrix = self.image_processor.affine_transform(frame)
                last_good_face = face
                last_good_box = box
                last_good_affine_matrix = affine_matrix.copy()
                faces.append(face)
                boxes.append(box)
                affine_matrices.append(last_good_affine_matrix)
            except RuntimeError as e:
                if "Face not detected" in str(e):
                    failed_indices.append(idx)
                    if last_good_affine_matrix is not None:
                        fallback_face, fallback_box = self.image_processor.warp_face_with_affine_matrix(
                            frame, last_good_affine_matrix
                        )
                        faces.append(fallback_face)
                        boxes.append(fallback_box)
                        affine_matrices.append(last_good_affine_matrix.copy())
                        last_good_face = fallback_face
                        last_good_box = fallback_box
                        reused_indices.append(idx)
                    else:
                        faces.append(None)
                        boxes.append(None)
                        affine_matrices.append(None)
                else:
                    raise

        if failed_indices:
            print(f"[Pililink] Warning: Face not detected in {len(failed_indices)}/{len(video_frames)} frames")
        if reused_indices:
            print(
                f"[Pililink] Reused affine matrix for {len(reused_indices)}/{len(video_frames)} frames "
                f"(interval={affine_detect_interval})"
            )

        # If no face was ever detected, raise a clear error
        if last_good_face is None:
            raise RuntimeError(
                "未检测到人脸 (No face detected in any frame). "
                "请确保输入视频中包含清晰可见的正面人脸。"
                "Please ensure the input video contains a clearly visible frontal face."
            )

        # Back-fill any leading frames that had no face (before the first good detection)
        for i in range(len(faces)):
            if faces[i] is None:
                faces[i] = last_good_face
                boxes[i] = last_good_box
                affine_matrices[i] = last_good_affine_matrix
            else:
                break

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def restore_video(self, faces, video_frames, boxes, affine_matrices):
        video_frames = video_frames[: faces.shape[0]]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            throw_if_processing_interrupted()
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            # face = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        mask_image_path: str = "latentsync/utils/mask.png",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        segment_inferences: int = 48,
        **kwargs,
    ):
        throw_if_processing_interrupted()
        is_train = self.denoising_unet.training
        self.denoising_unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        image_processor_device = device.type if hasattr(device, "type") else str(device)
        self.image_processor = ImageProcessor(height, mask=mask, device=image_processor_device, mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.denoising_unet.config.sample_size * self.vae_scale_factor
        width = width or self.denoising_unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        stream_temp_dir = os.path.join(kwargs.get("temp_dir", "temp"), "stream")
        os.makedirs(stream_temp_dir, exist_ok=True)
        skip_video_normalization = bool(kwargs.get("skip_video_normalization", False))
        clear_cuda_cache_per_segment = bool(kwargs.get("clear_cuda_cache_per_segment", True))
        affine_detect_interval = max(1, int(kwargs.get("affine_detect_interval", 1)))

        _pipeline_t0 = time.time()

        def _log_step(label):
            print(f"[Pililink {time.time() - _pipeline_t0:.2f}s] {label}")

        _log_step("Pipeline start | preparing audio features & video normalization")
        if skip_video_normalization:
            _log_step("Video normalization SKIPPED (already 25fps)")

        executor = ThreadPoolExecutor(max_workers=2)
        try:
            whisper_feature_future = executor.submit(self.audio_encoder.audio2feat, audio_path)
            normalized_video_future = None
            if not skip_video_normalization:
                normalized_video_future = executor.submit(
                    prepare_video_for_processing,
                    video_path,
                    True,
                    stream_temp_dir,
                )
            whisper_feature = wait_for_future_with_interrupt(whisper_feature_future)
            _log_step("Audio feature extraction done")
            if normalized_video_future is None:
                normalized_video_path = video_path
            else:
                normalized_video_path = wait_for_future_with_interrupt(normalized_video_future)
                _log_step("Video normalization done")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        _log_step(f"Whisper chunks ready ({len(whisper_chunks)} chunks)")
        throw_if_processing_interrupted()
        total_frames = get_video_frame_count(normalized_video_path)
        total_aligned_frames = (total_frames // num_frames) * num_frames
        total_inferences = min(total_aligned_frames // num_frames, len(whisper_chunks) // num_frames)

        if total_inferences <= 0:
            raise RuntimeError("No valid frames available for inference after alignment")

        final_video_path = os.path.join(stream_temp_dir, "video_synced.mp4")
        video_writer = None
        processed_frame_count = 0
        global_inference_index = 0

        segment_frames = max(num_frames, segment_inferences * num_frames)
        clip_batch_size = max(1, int(kwargs.get("clip_batch_size", 1)))
        total_output_frames = total_inferences * num_frames
        total_segments = max(1, int(math.ceil(total_inferences / max(1, int(segment_inferences)))))
        processed_segments = 0
        pipeline_start_time = time.time()
        print(
            "Pililink progress init:",
            f"total_clips={total_inferences}",
            f"total_frames={total_output_frames}",
            f"segment_inferences={segment_inferences}",
            f"clip_batch_size={clip_batch_size}",
        )

        try:
            for segment_video_frames in iter_video_cv2(
                normalized_video_path,
                segment_frames,
                interrupt_checker=throw_if_processing_interrupted,
            ):
                throw_if_processing_interrupted()
                remaining_inferences = total_inferences - global_inference_index
                if remaining_inferences <= 0:
                    break

                segment_frames_aligned = (len(segment_video_frames) // num_frames) * num_frames
                segment_inference_count = min(segment_frames_aligned // num_frames, remaining_inferences)
                if segment_inference_count <= 0:
                    continue

                processed_segments += 1
                _seg_t0 = time.time()
                print(
                    f"[Pililink] Segment {processed_segments}/{total_segments} start | "
                    f"clips {global_inference_index}/{total_inferences} | "
                    f"frames {processed_frame_count}/{total_output_frames}"
                )

                segment_video_frames = segment_video_frames[: segment_inference_count * num_frames]
                faces, boxes, affine_matrices = self.affine_transform_video(
                    segment_video_frames, affine_detect_interval=affine_detect_interval
                )
                _log_step(f"Segment {processed_segments} | affine transform done ({time.time() - _seg_t0:.2f}s)")

                num_channels_latents = self.vae.config.latent_channels

                synced_video_frames = []
                for batch_start in tqdm.tqdm(
                    range(0, segment_inference_count, clip_batch_size),
                    desc="Doing batched inference...",
                ):
                    throw_if_processing_interrupted()
                    current_clip_batch = min(clip_batch_size, segment_inference_count - batch_start)

                    frame_start = batch_start * num_frames
                    frame_end = frame_start + current_clip_batch * num_frames
                    inference_faces = faces[frame_start:frame_end]

                    ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                        inference_faces, affine_transform=False
                    )
                    ref_pixel_values = rearrange(
                        ref_pixel_values,
                        "(b f) c h w -> b f c h w",
                        b=current_clip_batch,
                        f=num_frames,
                    )
                    masked_pixel_values = rearrange(
                        masked_pixel_values,
                        "(b f) c h w -> b f c h w",
                        b=current_clip_batch,
                        f=num_frames,
                    )
                    masks = rearrange(
                        masks,
                        "(b f) c h w -> b f c h w",
                        b=current_clip_batch,
                        f=num_frames,
                    )

                    if self.denoising_unet.add_audio_layer:
                        batched_audio_embeds = []
                        for clip_offset in range(current_clip_batch):
                            global_i = global_inference_index + batch_start + clip_offset
                            batched_audio_embeds.append(
                                torch.stack(whisper_chunks[global_i * num_frames : (global_i + 1) * num_frames])
                            )
                        audio_embeds = torch.stack(batched_audio_embeds, dim=0)
                        audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                        if do_classifier_free_guidance:
                            null_audio_embeds = torch.zeros_like(audio_embeds)
                            audio_embeds = torch.cat([null_audio_embeds, audio_embeds], dim=0)
                    else:
                        audio_embeds = None

                    latents = self.prepare_latents(
                        current_clip_batch,
                        num_frames,
                        num_channels_latents,
                        height,
                        width,
                        weight_dtype,
                        device,
                        generator,
                    )

                    mask_latents, masked_image_latents = self.prepare_mask_latents(
                        masks,
                        masked_pixel_values,
                        height,
                        width,
                        weight_dtype,
                        device,
                        generator,
                        do_classifier_free_guidance,
                    )

                    ref_latents = self.prepare_image_latents(
                        ref_pixel_values,
                        device,
                        weight_dtype,
                        generator,
                        do_classifier_free_guidance,
                    )

                    ref_pixel_values_flat = rearrange(ref_pixel_values, "b f c h w -> (b f) c h w")
                    masks_flat = rearrange(masks, "b f c h w -> (b f) c h w")

                    # DPM-Solver schedulers keep internal step state; reset before each batch trajectory.
                    self.scheduler.set_timesteps(num_inference_steps, device=device)
                    timesteps = self.scheduler.timesteps
                    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                    with self.progress_bar(total=num_inference_steps) as progress_bar:
                        for j, t in enumerate(timesteps):
                            throw_if_processing_interrupted()
                            denoising_unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                            denoising_unet_input = self.scheduler.scale_model_input(denoising_unet_input, t)
                            denoising_unet_input = torch.cat(
                                [denoising_unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                            )
                            noise_pred = self.denoising_unet(
                                denoising_unet_input, t, encoder_hidden_states=audio_embeds
                            ).sample

                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                            if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                                progress_bar.update()
                                if callback is not None and j % callback_steps == 0:
                                    callback(j, t, latents)

                    decoded_latents = self.decode_latents(latents)
                    decoded_latents = self.paste_surrounding_pixels_back(
                        decoded_latents, ref_pixel_values_flat, 1 - masks_flat, device, weight_dtype
                    )
                    synced_video_frames.append(decoded_latents)

                    del audio_embeds, inference_faces, latents
                    del ref_pixel_values, masked_pixel_values, masks
                    del ref_pixel_values_flat, masks_flat
                    del mask_latents, masked_image_latents, ref_latents, decoded_latents

                _restore_t0 = time.time()
                restored_segment = self.restore_video(
                    torch.cat(synced_video_frames), segment_video_frames, boxes, affine_matrices
                )
                _log_step(f"Segment {processed_segments} | face restore done ({time.time() - _restore_t0:.2f}s)")

                if video_writer is None:
                    h, w = restored_segment[0].shape[:2]
                    video_writer, _writer_codec = open_ffmpeg_video_pipe_writer(
                        final_video_path, w, h, fps=video_fps
                    )
                    _log_step(f"FFmpeg pipe writer opened ({_writer_codec}, {w}x{h})")

                _write_t0 = time.time()
                for frame in restored_segment:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.stdin.write(bgr.tobytes())
                _log_step(f"Segment {processed_segments} | {restored_segment.shape[0]} frames written ({time.time() - _write_t0:.2f}s)")

                processed_frame_count += restored_segment.shape[0]
                global_inference_index += segment_inference_count
                overall_ratio = global_inference_index / max(total_inferences, 1)
                elapsed = time.time() - pipeline_start_time
                eta_seconds = 0.0
                if overall_ratio > 0:
                    eta_seconds = elapsed * (1.0 / overall_ratio - 1.0)
                print(
                    f"[Pililink] Segment {processed_segments}/{total_segments} done | "
                    f"clips {global_inference_index}/{total_inferences} ({overall_ratio * 100:.1f}%) | "
                    f"elapsed {elapsed:.1f}s | eta {eta_seconds:.1f}s"
                )

                del faces, boxes, affine_matrices, synced_video_frames, restored_segment, segment_video_frames
                if clear_cuda_cache_per_segment and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            if video_writer is not None:
                try:
                    _close_t0 = time.time()
                    close_ffmpeg_video_pipe_writer(video_writer, timeout=60)
                    _log_step(f"FFmpeg pipe writer closed ({time.time() - _close_t0:.2f}s)")
                except Exception as e:
                    print(f"[Pililink] FFmpeg pipe writer cleanup warning: {e}")
                video_writer = None

        if video_writer is not None:
            try:
                _close_t0 = time.time()
                close_ffmpeg_video_pipe_writer(video_writer, timeout=60)
                _log_step(f"FFmpeg pipe writer closed ({time.time() - _close_t0:.2f}s)")
            except Exception as e:
                print(f"[Pililink] FFmpeg pipe writer cleanup warning: {e}")
            video_writer = None

        if is_train:
            self.denoising_unet.train()

        if processed_frame_count <= 0:
            raise RuntimeError("No output frames produced during segmented inference")

        output_duration = processed_frame_count / max(video_fps, 1)
        _log_step(f"Starting audio mux (stream copy, duration={output_duration:.2f}s)")
        throw_if_processing_interrupted()
        _mux_t0 = time.time()
        mux_video_audio_stream_copy(
            final_video_path, audio_path, video_out_path, duration=output_duration
        )
        _log_step(f"Audio mux done ({time.time() - _mux_t0:.2f}s)")
        _log_step(f"Pipeline complete | total {processed_frame_count} frames in {time.time() - _pipeline_t0:.2f}s")
