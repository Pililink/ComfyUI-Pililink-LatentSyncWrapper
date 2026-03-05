# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import os
import shutil
import time
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
    get_video_frame_count,
    iter_video_cv2,
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

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
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

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            throw_if_processing_interrupted()
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

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
        batch_size = 1
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda", mask_image=mask_image)
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

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        throw_if_processing_interrupted()

        stream_temp_dir = os.path.join(kwargs.get("temp_dir", "temp"), "stream")
        os.makedirs(stream_temp_dir, exist_ok=True)

        normalized_video_path = prepare_video_for_processing(video_path, change_fps=True, temp_dir=stream_temp_dir)
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

        try:
            for segment_video_frames in iter_video_cv2(normalized_video_path, segment_frames):
                throw_if_processing_interrupted()
                remaining_inferences = total_inferences - global_inference_index
                if remaining_inferences <= 0:
                    break

                segment_frames_aligned = (len(segment_video_frames) // num_frames) * num_frames
                segment_inference_count = min(segment_frames_aligned // num_frames, remaining_inferences)
                if segment_inference_count <= 0:
                    continue

                segment_video_frames = segment_video_frames[: segment_inference_count * num_frames]
                faces, boxes, affine_matrices = self.affine_transform_video(segment_video_frames)

                num_channels_latents = self.vae.config.latent_channels
                all_latents = self.prepare_latents(
                    batch_size,
                    num_frames * segment_inference_count,
                    num_channels_latents,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                )

                synced_video_frames = []
                for i in tqdm.tqdm(range(segment_inference_count), desc="Doing inference..."):
                    throw_if_processing_interrupted()
                    global_i = global_inference_index + i
                    if self.denoising_unet.add_audio_layer:
                        audio_embeds = torch.stack(whisper_chunks[global_i * num_frames : (global_i + 1) * num_frames])
                        audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                        if do_classifier_free_guidance:
                            null_audio_embeds = torch.zeros_like(audio_embeds)
                            audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
                    else:
                        audio_embeds = None

                    inference_faces = faces[i * num_frames : (i + 1) * num_frames]
                    latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
                    ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                        inference_faces, affine_transform=False
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
                        decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
                    )
                    synced_video_frames.append(decoded_latents)

                    del audio_embeds, inference_faces, latents, ref_pixel_values, masked_pixel_values, masks
                    del mask_latents, masked_image_latents, ref_latents, decoded_latents

                restored_segment = self.restore_video(
                    torch.cat(synced_video_frames), segment_video_frames, boxes, affine_matrices
                )

                if video_writer is None:
                    h, w = restored_segment[0].shape[:2]
                    video_writer = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Failed to open output writer: {final_video_path}")

                for frame in restored_segment:
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                processed_frame_count += restored_segment.shape[0]
                global_inference_index += segment_inference_count

                del faces, boxes, affine_matrices, all_latents, synced_video_frames, restored_segment, segment_video_frames
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            if video_writer is not None:
                video_writer.release()
                video_writer = None

        if video_writer is not None:
            video_writer.release()
            video_writer = None

        if is_train:
            self.denoising_unet.train()

        if processed_frame_count <= 0:
            raise RuntimeError("No output frames produced during segmented inference")

        output_duration = processed_frame_count / max(video_fps, 1)
        command = (
            f"ffmpeg -y -loglevel error -nostdin -i \"{final_video_path}\" -i \"{audio_path}\" "
            f"-t {output_duration:.6f} -c:v libx264 -c:a aac -q:v 0 -q:a 0 \"{video_out_path}\""
        )
        process = None
        try:
            throw_if_processing_interrupted()
            process = subprocess.Popen(command, shell=True)
            while process.poll() is None:
                throw_if_processing_interrupted()
                time.sleep(0.1)
            if process.returncode != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {process.returncode}")
        except Exception:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except Exception:
                    process.kill()
            raise
