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

import os
import imageio
import numpy as np
import json
import queue
import threading
from typing import Union
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from torchvision import transforms

from einops import rearrange
import cv2
from decord import AudioReader, VideoReader
import shutil
import subprocess


# Machine epsilon for a float32 (single precision)
eps = np.finfo(np.float32).eps
_FFMPEG_VIDEO_ENCODERS = None


def read_json(filepath: str):
    with open(filepath) as f:
        json_dict = json.load(f)
    return json_dict


def get_available_ffmpeg_video_encoders():
    global _FFMPEG_VIDEO_ENCODERS
    if _FFMPEG_VIDEO_ENCODERS is not None:
        return _FFMPEG_VIDEO_ENCODERS

    encoders = set()
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = "\n".join([result.stdout or "", result.stderr or ""])
        for line in output.splitlines():
            line = line.strip()
            if not line or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            flags = parts[0]
            if "V" in flags:
                encoders.add(parts[1])
    except Exception:
        pass

    _FFMPEG_VIDEO_ENCODERS = encoders
    return _FFMPEG_VIDEO_ENCODERS


def get_preferred_ffmpeg_video_codec():
    available_encoders = get_available_ffmpeg_video_encoders()
    for codec in ("h264_nvenc", "h264_qsv", "h264_amf", "libx264"):
        if codec == "libx264" or codec in available_encoders:
            return codec
    return "libx264"


def get_ffmpeg_video_encode_args(codec=None):
    codec = codec or get_preferred_ffmpeg_video_codec()
    if codec == "h264_nvenc":
        return ["-c:v", codec, "-preset", "p4", "-cq", "19", "-pix_fmt", "yuv420p"]
    if codec == "h264_qsv":
        return ["-c:v", codec, "-global_quality", "21", "-pix_fmt", "yuv420p"]
    if codec == "h264_amf":
        return ["-c:v", codec, "-quality", "balanced", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]


def open_ffmpeg_video_pipe_writer(output_path, width, height, fps=25):
    """Open an FFmpeg subprocess that accepts raw BGR24 frames on stdin and encodes to h264.

    Returns (process, codec_used).  The caller must:
      1. Write raw frame bytes to ``process.stdin``
      2. Call ``process.stdin.close()`` when done
      3. Call ``process.wait()`` and check ``process.returncode``
    """
    codec = get_preferred_ffmpeg_video_codec()
    codecs_to_try = [codec]
    if codec != "libx264":
        codecs_to_try.append("libx264")

    last_error = None
    for c in codecs_to_try:
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "pipe:0",
            *get_ffmpeg_video_encode_args(c),
            output_path,
        ]
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Quick sanity check — if the process already died the codec is bad.
            if proc.poll() is not None:
                stderr_out = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                raise RuntimeError(f"FFmpeg exited immediately: {stderr_out}")
            return proc, c
        except Exception as e:
            last_error = e
            if c != "libx264":
                print(f"LatentSync util: FFmpeg pipe writer with {c} failed, trying libx264")
            continue

    raise RuntimeError(f"Failed to open FFmpeg pipe writer: {last_error}")


def close_ffmpeg_video_pipe_writer(proc, timeout=60):
    """Gracefully close an FFmpeg pipe writer subprocess.

    Returns True on success, raises on failure.
    """
    if proc is None:
        return True
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait(timeout=timeout)
        if proc.returncode != 0:
            stderr_out = ""
            if proc.stderr:
                stderr_out = proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"FFmpeg pipe writer failed (exit {proc.returncode}): {stderr_out}")
        return True
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        raise


def mux_video_audio_stream_copy(video_path, audio_path, output_path, duration=None):
    """Mux a video (already h264) with audio using stream copy (near-instant).

    Falls back to re-encoding on failure.
    """
    duration_args = ["-t", f"{duration:.6f}"] if duration is not None else []
    # Fast path: stream copy video + encode audio
    mux_command = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", video_path,
        "-i", audio_path,
        *duration_args,
        "-c:v", "copy",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(mux_command, capture_output=True, text=True)
    if result.returncode == 0:
        return

    # Fallback: re-encode video with codec chain
    print(f"[LatentSync] Stream-copy mux failed ({result.stderr.strip()}), falling back to re-encode")
    preferred_codec = get_preferred_ffmpeg_video_codec()
    codecs_to_try = [preferred_codec]
    if preferred_codec != "libx264":
        codecs_to_try.append("libx264")

    last_error = None
    for codec in codecs_to_try:
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
            "-i", video_path,
            "-i", audio_path,
            *duration_args,
            *get_ffmpeg_video_encode_args(codec),
            "-c:a", "aac",
            output_path,
        ]
        re_result = subprocess.run(command, capture_output=True, text=True)
        if re_result.returncode == 0:
            if codec != preferred_codec:
                print(f"[LatentSync] ffmpeg fallback succeeded with {codec}")
            return
        last_error = RuntimeError(
            f"ffmpeg re-encode failed (exit {re_result.returncode}) with {codec}: {re_result.stderr.strip()}"
        )
        if codec != "libx264":
            print(f"[LatentSync] ffmpeg {codec} failed, trying libx264")
            continue
        raise last_error

    if last_error is not None:
        raise last_error


def run_ffmpeg_command(command, error_message):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_message = (result.stderr or result.stdout or "").strip()
        if stderr_message:
            raise RuntimeError(f"{error_message} (exit code {result.returncode}): {stderr_message}")
        raise RuntimeError(f"{error_message} (exit code {result.returncode})")


def run_ffmpeg_video_command_with_fallback(command_builder, error_message):
    preferred_codec = get_preferred_ffmpeg_video_codec()
    codecs_to_try = [preferred_codec]
    if preferred_codec != "libx264":
        codecs_to_try.append("libx264")

    last_error = None
    for codec in codecs_to_try:
        try:
            run_ffmpeg_command(command_builder(codec), error_message)
            if codec != preferred_codec:
                print(f"LatentSync util: ffmpeg video encoder fallback succeeded with {codec}")
            return
        except RuntimeError as exc:
            last_error = exc
            if codec != "libx264":
                print(f"LatentSync util: ffmpeg video encoder {codec} failed, retrying with libx264")
                continue
            raise

    if last_error is not None:
        raise last_error


def read_video(video_path: str, change_fps=True, use_decord=True):
    if change_fps:
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        target_video_path = os.path.join(temp_dir, "video.mp4")
        run_ffmpeg_video_command_with_fallback(
            lambda codec: [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-nostdin",
                "-i",
                video_path,
                "-r",
                "25",
                *get_ffmpeg_video_encode_args(codec),
                target_video_path,
            ],
            f"Failed to normalize video fps: {video_path}",
        )
    else:
        target_video_path = video_path

    if use_decord:
        return read_video_decord(target_video_path)
    else:
        return read_video_cv2(target_video_path)


def prepare_video_for_processing(video_path: str, change_fps=True, temp_dir="temp"):
    """Prepare input video for processing and return path to normalized video."""
    if change_fps:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        target_video_path = os.path.join(temp_dir, "video.mp4")
        run_ffmpeg_video_command_with_fallback(
            lambda codec: [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-nostdin",
                "-i",
                video_path,
                "-r",
                "25",
                *get_ffmpeg_video_encode_args(codec),
                target_video_path,
            ],
            f"Failed to normalize input video: {video_path}",
        )
        return target_video_path
    return video_path


def get_video_frame_count(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for frame count: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def estimate_video_chunk_bytes(video_path: str, chunk_size: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for chunk size estimate: {video_path}")
    try:
        ret, frame = cap.read()
        if not ret:
            return 0
        return int(frame.nbytes * max(chunk_size, 1))
    finally:
        cap.release()


def resolve_video_prefetch_chunks(video_path: str, chunk_size: int, prefetch_chunks: int = None):
    if prefetch_chunks is not None:
        return max(1, int(prefetch_chunks))

    default_prefetch = 2
    try:
        import psutil

        chunk_bytes = estimate_video_chunk_bytes(video_path, chunk_size)
        if chunk_bytes <= 0:
            return 1

        available_bytes = int(psutil.virtual_memory().available)
        ram_budget_bytes = min(int(available_bytes * 0.30), 6 * 1024 * 1024 * 1024)
        if ram_budget_bytes <= chunk_bytes:
            return 1

        adaptive_prefetch = ram_budget_bytes // chunk_bytes
        return max(1, min(8, int(adaptive_prefetch)))
    except Exception:
        return default_prefetch


def iter_video_cv2_sync(video_path: str, chunk_size: int, interrupt_checker=None):
    """Yield RGB frame chunks from a video path using cv2."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for streaming read: {video_path}")

    frames = []
    try:
        while True:
            if interrupt_checker is not None:
                interrupt_checker()
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if len(frames) >= chunk_size:
                yield np.array(frames)
                frames = []
        if frames:
            yield np.array(frames)
    finally:
        cap.release()


def iter_video_cv2(video_path: str, chunk_size: int, prefetch_chunks: int = None, interrupt_checker=None):
    resolved_prefetch_chunks = resolve_video_prefetch_chunks(video_path, chunk_size, prefetch_chunks=prefetch_chunks)
    if resolved_prefetch_chunks <= 1:
        yield from iter_video_cv2_sync(video_path, chunk_size, interrupt_checker=interrupt_checker)
        return

    chunk_queue = queue.Queue(maxsize=resolved_prefetch_chunks)
    stop_event = threading.Event()
    sentinel = object()
    reader_errors = []

    print(f"LatentSync: Video RAM prefetch enabled with {resolved_prefetch_chunks} chunks")

    def put_with_backpressure(item):
        while not stop_event.is_set():
            try:
                chunk_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def reader():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            reader_errors.append(RuntimeError(f"Could not open video for streaming read: {video_path}"))
            put_with_backpressure(sentinel)
            return

        frames = []
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                if len(frames) >= chunk_size:
                    if not put_with_backpressure(np.array(frames)):
                        return
                    frames = []
            if frames:
                put_with_backpressure(np.array(frames))
        except Exception as exc:
            reader_errors.append(exc)
        finally:
            cap.release()
            put_with_backpressure(sentinel)

    reader_thread = threading.Thread(target=reader, name="latentsync_video_prefetch", daemon=True)
    reader_thread.start()

    try:
        while True:
            if interrupt_checker is not None:
                interrupt_checker()
            try:
                item = chunk_queue.get(timeout=0.1)
            except queue.Empty:
                if reader_errors:
                    raise reader_errors[0]
                if not reader_thread.is_alive() and chunk_queue.empty():
                    break
                continue

            if item is sentinel:
                break
            yield item

        if reader_errors:
            raise reader_errors[0]
    finally:
        stop_event.set()
        reader_thread.join(timeout=1.0)


def read_video_decord(video_path: str):
    vr = VideoReader(video_path)
    video_frames = vr[:].asnumpy()
    vr.seek(0)
    return video_frames


def read_video_cv2(video_path: str):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return np.array([])

    frames = []

    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame_rgb)

    # Release the video capture object
    cap.release()

    return np.array(frames)


def read_audio(audio_path: str, audio_sample_rate: int = 16000):
    if audio_path is None:
        raise ValueError("Audio path is required.")
    ar = AudioReader(audio_path, sample_rate=audio_sample_rate, mono=True)

    # To access the audio samples
    audio_samples = torch.from_numpy(ar[:].asnumpy())
    audio_samples = audio_samples.squeeze(0)

    return audio_samples


def write_video(video_output_path: str, video_frames: np.ndarray, fps: int):
    height, width = video_frames[0].shape[:2]
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    # out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"vp09"), fps, (width, height))
    for frame in video_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def init_dist(backend="nccl", **kwargs):
    """Initializes distributed environment."""
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for training.")
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)

    return local_rank


def zero_rank_print(s):
    if dist.is_initialized() and dist.get_rank() == 0:
        print("### " + s)


def zero_rank_log(logger, message: str):
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(message)


def check_video_fps(video_path: str):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        raise ValueError(f"Video FPS is not 25, it is {fps}. Please convert the video to 25 FPS.")


def one_step_sampling(ddim_scheduler, pred_noise, timesteps, x_t):
    # Compute alphas, betas
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timesteps].to(dtype=pred_noise.dtype)
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/abs/2010.02502
    if ddim_scheduler.config.prediction_type == "epsilon":
        beta_prod_t = beta_prod_t[:, None, None, None, None]
        alpha_prod_t = alpha_prod_t[:, None, None, None, None]
        pred_original_sample = (x_t - beta_prod_t ** (0.5) * pred_noise) / alpha_prod_t ** (0.5)
    else:
        raise NotImplementedError("This prediction type is not implemented yet")

    # Clip "predicted x_0"
    if ddim_scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    return pred_original_sample


def plot_loss_chart(save_path: str, *args):
    # Creating the plot
    plt.figure()
    for loss_line in args:
        plt.plot(loss_line[1], loss_line[2], label=loss_line[0])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()

    # Save the figure to a file
    plt.savefig(save_path)

    # Close the figure to free memory
    plt.close()


CRED = "\033[91m"
CEND = "\033[0m"


def red_text(text: str):
    return f"{CRED}{text}{CEND}"


log_loss = nn.BCELoss(reduction="none")


def cosine_loss(vision_embeds, audio_embeds, y):
    sims = nn.functional.cosine_similarity(vision_embeds, audio_embeds)
    # sims[sims!=sims] = 0 # remove nan
    # sims = sims.clamp(0, 1)
    loss = log_loss(sims.unsqueeze(1), y).squeeze()
    return loss


def save_image(image, save_path):
    # input size (C, H, W)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    image = transforms.ToPILImage()(image)
    # Save the image copy
    image.save(save_path)

    # Close the image file
    image.close()


def gather_loss(loss, device):
    # Sum the local loss across all processes
    local_loss = loss.item()
    global_loss = torch.tensor(local_loss, dtype=torch.float32).to(device)
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)

    # Calculate the average loss across all processes
    global_average_loss = global_loss.item() / dist.get_world_size()
    return global_average_loss


def gather_video_paths_recursively(input_dir):
    print(f"Recursively gathering video paths of {input_dir} ...")
    paths = []
    gather_video_paths(input_dir, paths)
    return paths


def gather_video_paths(input_dir, paths):
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".mp4"):
            filepath = os.path.join(input_dir, file)
            paths.append(filepath)
        elif os.path.isdir(os.path.join(input_dir, file)):
            gather_video_paths(os.path.join(input_dir, file), paths)


def count_video_time(video_path):
    video = cv2.VideoCapture(video_path)

    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    return frame_count / fps


def check_ffmpeg_installed():
    # Run the ffmpeg command with the -version argument to check if it's installed
    result = subprocess.run("ffmpeg -version", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if not result.returncode == 0:
        raise FileNotFoundError("ffmpeg not found, please install it by:\n    $ conda install -c conda-forge ffmpeg")
