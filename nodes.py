import atexit
import importlib.util
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid

import cv2
import numpy as np
import requests
import torch
import torchaudio

REQUIRED_PACKAGES = [
    "omegaconf",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "einops",
    "diffusers",
    "ffmpeg-python",
]

folder_paths = __import__("folder_paths")

comfy_utils = None
try:
    comfy_utils = __import__("comfy.utils", fromlist=["ProgressBar"])
except Exception:
    comfy_utils = None

PromptServer = None
try:
    from server import PromptServer
except Exception:
    PromptServer = None

try:
    import torchvision.io as torchvision_io
except Exception:
    torchvision_io = None

av_mod = None

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

try:
    from .latentsync_paths import build_latentsync_path, initialize_latentsync_paths
except Exception:
    from latentsync_paths import build_latentsync_path, initialize_latentsync_paths

_LATENTSYNC_PATHS = initialize_latentsync_paths()
LATENTSYNC_ROOT_DIR = _LATENTSYNC_PATHS["root"]
HF_CACHE_DIR = _LATENTSYNC_PATHS["hf_cache_dir"]

NODE_CATEGORY = "PililinkLatentSync"
NODE_KEY_MAIN = "PililinkLatentSyncNode"
NODE_KEY_ADJUSTER = "PililinkLatentSyncLengthAdjuster"
NODE_KEY_MAIN_PATH = "PililinkLatentSyncVideoPathNode"
NODE_DISPLAY_MAIN = "Pililink LatentSync 1.5"
NODE_DISPLAY_ADJUSTER = "Pililink LatentSync Length Adjuster"
NODE_DISPLAY_MAIN_PATH = "Pililink LatentSync 1.5 (Video Path)"
_FFMPEG_VIDEO_ENCODERS = None
_RUNTIME_SETUP_DONE = False
_RUNTIME_SETUP_LOCK = threading.Lock()

PATH_NODE_DEFAULTS = {
    "seed": 1247,
    "lips_expression": 1.5,
    "inference_steps": 20,
    "vram_usage": "medium",
    "segment_inferences": 8,
    "clip_batch_size": 1,
    "auto_oom_fallback": True,
    "quality_mode": "balanced",
    "deepcache": "on",
    "deepcache_cache_interval": 3,
    "deepcache_branch_id": 0,
    "scheduler_type": "ddim",
    "affine_detect_interval": 1,
    "mode": "normal",
    "silent_padding_sec": 0.5,
    "auto_silent_padding": False,
    "filename_prefix": "LatentSync/Pililink",
}


def _clamp_progress_fraction(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class PililinkProgressReporter:
    def __init__(self, *, total=1000, node_id=None):
        self.total = max(1, int(total))
        self.node_id = str(node_id) if node_id is not None else None
        self.current = 0
        self.last_stage = None
        self.progress_bar = None
        if comfy_utils is not None and hasattr(comfy_utils, "ProgressBar"):
            try:
                self.progress_bar = comfy_utils.ProgressBar(self.total)
            except Exception:
                self.progress_bar = None

    def _send_stage_message(self, stage):
        if (
            not stage
            or stage == self.last_stage
            or PromptServer is None
            or getattr(PromptServer, "instance", None) is None
            or self.node_id is None
        ):
            if stage:
                self.last_stage = stage
            return

        try:
            PromptServer.instance.send_sync(
                "pililink.progress.stage",
                {
                    "node": self.node_id,
                    "stage": stage,
                    "value": self.current,
                    "max": self.total,
                },
            )
        except Exception:
            pass
        self.last_stage = stage

    def update_fraction(self, fraction, stage=None, allow_decrease=False):
        fraction = _clamp_progress_fraction(fraction)
        value = int(round(fraction * self.total))
        if not allow_decrease:
            value = max(self.current, value)
        value = min(self.total, value)
        self.current = value

        if self.progress_bar is not None:
            try:
                self.progress_bar.update_absolute(self.current, self.total)
            except Exception:
                pass

        self._send_stage_message(stage)

    def update_stage_fraction(self, start_fraction, end_fraction, fraction, stage=None):
        start_fraction = _clamp_progress_fraction(start_fraction)
        end_fraction = max(start_fraction, _clamp_progress_fraction(end_fraction))
        mapped_fraction = start_fraction + (end_fraction - start_fraction) * _clamp_progress_fraction(fraction)
        self.update_fraction(mapped_fraction, stage=stage)

    def make_stage_callback(self, start_fraction, end_fraction):
        def _callback(stage, fraction):
            self.update_stage_fraction(start_fraction, end_fraction, fraction, stage=stage)

        return _callback

# Function to check for potential conflicts with other LatentSync implementations
def check_for_conflicts():
    """Check if other LatentSync implementations might conflict"""
    try:
        custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")[0]
        original_path = os.path.join(custom_nodes_dir, "ComfyUI-LatentSyncWrapper")
        
        if os.path.exists(original_path):
            print("[Pililink LatentSync] Detected ComfyUI-LatentSyncWrapper - using isolated paths to avoid conflicts")
            return True
    except:
        pass
    return False

def _normalize_str_path(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode()
        except Exception:
            return None
    if isinstance(value, os.PathLike):
        path_value = os.fspath(value)
        if isinstance(path_value, str):
            return path_value
        if isinstance(path_value, bytes):
            try:
                return path_value.decode()
            except Exception:
                return None
    return None


def get_comfy_temp_root():
    try:
        get_temp_directory = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp_directory):
            comfy_temp = _normalize_str_path(get_temp_directory())
            if comfy_temp:
                return comfy_temp
    except Exception:
        pass

    comfy_temp = _normalize_str_path(getattr(folder_paths, "temp_directory", None))
    if comfy_temp:
        return comfy_temp

    output_dir = _normalize_str_path(getattr(folder_paths, "output_directory", None))
    if output_dir:
        return os.path.join(output_dir, "temp")

    return tempfile.gettempdir()

# Create a module-level function to set up node-local temp directory
def init_temp_directories():
    comfy_temp_root = get_comfy_temp_root()
    os.makedirs(comfy_temp_root, exist_ok=True)

    session_dir = os.path.join(comfy_temp_root, f"pilkilink_latentsync_{uuid.uuid4().hex[:8]}")
    os.makedirs(session_dir, exist_ok=True)

    print(f"Set up ComfyUI temp directory: {session_dir}")
    return session_dir

# Function to clean up everything when the module exits
def module_cleanup():
    """Clean up all resources when the module is unloaded"""
    global MODULE_TEMP_DIR
    
    # Clean up temp directory (node-local session directory)
    if MODULE_TEMP_DIR and os.path.exists(MODULE_TEMP_DIR):
        try:
            shutil.rmtree(MODULE_TEMP_DIR, ignore_errors=True)
            print("Cleaned up node-local temp directory")
        except:
            pass


def ensure_runtime_ready():
    global MODULE_TEMP_DIR, _RUNTIME_SETUP_DONE

    if _RUNTIME_SETUP_DONE:
        return

    with _RUNTIME_SETUP_LOCK:
        if _RUNTIME_SETUP_DONE:
            return

        if not os.path.exists(MODULE_TEMP_DIR):
            os.makedirs(MODULE_TEMP_DIR, exist_ok=True)

        check_and_install_dependencies()
        setup_models()
        _RUNTIME_SETUP_DONE = True

# Call this before anything else
MODULE_TEMP_DIR = init_temp_directories()

# Register the cleanup handler to run when Python exits
atexit.register(module_cleanup)

# Check for conflicts with other implementations
check_for_conflicts()

comfy_model_management = None
try:
    comfy_model_management = __import__("comfy.model_management", fromlist=["throw_exception_if_processing_interrupted"])
except Exception:
    comfy_model_management = None


def throw_if_processing_interrupted():
    if comfy_model_management is not None:
        comfy_model_management.throw_exception_if_processing_interrupted()


def get_latentsync_path(*parts, mkdir_parent=False):
    """Build path under active ComfyUI/models/latensync1.5 root."""
    return build_latentsync_path(LATENTSYNC_ROOT_DIR, *parts, mkdir_parent=mkdir_parent)

def import_refactor_runtime_module():
    try:
        from . import pililink_refactor_runtime as runtime_mod
    except Exception:
        import pililink_refactor_runtime as runtime_mod
    return runtime_mod

def check_ffmpeg():
    try:
        if platform.system() == "Windows":
            # Check if ffmpeg exists in PATH
            ffmpeg_path = shutil.which("ffmpeg.exe")
            if ffmpeg_path is None:
                # Look for ffmpeg in common locations
                possible_paths = [
                    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "ffmpeg", "bin"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin"),
                ]
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        # Add to PATH
                        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                        return True
                print("FFmpeg not found. Please install FFmpeg and add it to PATH")
                return False
            return True
        else:
            subprocess.run(["ffmpeg"], capture_output=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install FFmpeg")
        return False

def check_and_install_dependencies():
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is required but not found")
    
    # Check if we've already run this function successfully
    cache_dir = get_latentsync_path("runtime_cache", "deps")
    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_marker = os.path.join(cache_dir, ".pililink_deps_installed")
    if os.path.exists(cache_marker):
        print("Pililink dependencies already verified, skipping check.")
        return
        
    def is_package_installed(package_name):
        return importlib.util.find_spec(package_name) is not None
        
    def install_package(package):
        python_exe = sys.executable
        try:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', package],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            raise RuntimeError(f"Failed to install required package: {package}")
            
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        if not is_package_installed(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                install_package(package)
            except Exception as e:
                print(f"Warning: Failed to install {package}: {str(e)}")
                raise
    else:
        print("All required packages are already installed.")
    
    # Create marker file
    try:
        with open(cache_marker, 'w') as f:
            f.write(f"Pililink dependencies checked on {time.ctime()}")
    except Exception as e:
        print(f"Warning: Could not create cache marker file: {str(e)}")

def is_probably_25fps_video(video_path):
    """Check if a video file is already at ~25fps, so FFmpeg normalization can be skipped."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
        if fps <= 0.0:
            return False
        return abs(fps - 25.0) < 0.01
    except Exception:
        return False


def download_model(url, save_path):
    """Download a model from a URL and save it to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Check if file already exists
    if os.path.exists(save_path):
        print(f"Model file already exists at {save_path}, skipping download.")
        return
        
    print(f"Downloading {url} to {save_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rDownload progress: {percent:.1f}%", end="")
    print("\nDownload complete")

def pre_download_models():
    """Pre-download all required models."""
    models = {
        "s3fd-e19a316812.pth": "https://www.adrianbulat.com/downloads/python-fan/s3fd-e19a316812.pth",
        # Add other models here
    }

    cache_dir = get_latentsync_path("auxiliary")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if we've already run this function successfully by creating a marker file
    cache_marker = os.path.join(cache_dir, ".pililink_cache_complete")
    if os.path.exists(cache_marker):
        print("Pre-downloaded Pililink models already exist, skipping download.")
        return
    
    for model_name, url in models.items():
        save_path = os.path.join(cache_dir, model_name)
        if not os.path.exists(save_path):
            print(f"Downloading {model_name}...")
            download_model(url, save_path)
        else:
            print(f"{model_name} already exists in Pililink cache.")
    
    # Create marker file to indicate successful completion
    with open(cache_marker, 'w') as f:
        f.write(f"Pililink cache completed on {time.ctime()}")

def setup_models():
    """Setup and pre-download all required models."""
    throw_if_processing_interrupted()
    # Use our global temp directory
    global MODULE_TEMP_DIR
    
    # Pre-download additional models
    pre_download_models()

    # Use a unified model root under ComfyUI/models/LatentSync-1.5
    ckpt_dir = LATENTSYNC_ROOT_DIR
    whisper_dir = os.path.join(ckpt_dir, "whisper")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(whisper_dir, exist_ok=True)

    # Keep Hugging Face cache under LatentSync root for offline packaging.
    temp_downloads = HF_CACHE_DIR
    os.makedirs(temp_downloads, exist_ok=True)
    
    unet_path = os.path.join(ckpt_dir, "latentsync_unet.pt")
    whisper_path = os.path.join(whisper_dir, "tiny.pt")

    # Only download if the files don't already exist
    if os.path.exists(unet_path) and os.path.exists(whisper_path):
        print("Pililink model checkpoints already exist, skipping download.")
        return
        
    print("Downloading required Pililink model checkpoints... This may take a while.")
    try:
        throw_if_processing_interrupted()
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is not available")
        snapshot_download(repo_id="ByteDance/LatentSync-1.5",
                         allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                         local_dir=ckpt_dir, 
                         local_dir_use_symlinks=False,
                         cache_dir=temp_downloads)
        throw_if_processing_interrupted()
        print("Pililink model checkpoints downloaded successfully!")
    except Exception as e:
        print(f"Error downloading Pililink models: {str(e)}")
        print("\nPlease download models manually for Pililink LatentSync:")
        print("1. Visit: https://huggingface.co/chunyu-li/LatentSync")
        print("2. Download: latentsync_unet.pt and whisper/tiny.pt")
        print(f"3. Place them in: {ckpt_dir}")
        print(f"   with whisper/tiny.pt in: {whisper_dir}")
        raise RuntimeError("Pililink model download failed. See instructions above.")

def resolve_user_path(path_value, input_name):
    if path_value is None:
        raise ValueError(f"Pililink: {input_name} is required")

    resolved_path = str(path_value).strip().strip('"').strip("'")
    if not resolved_path:
        raise ValueError(f"Pililink: {input_name} cannot be empty")

    resolved_path = os.path.abspath(os.path.expandvars(os.path.expanduser(resolved_path)))
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Pililink: {input_name} not found: {resolved_path}")

    return resolved_path


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


def get_output_video_path(filename_prefix, output_path=""):
    output_path = str(output_path or "").strip().strip('"').strip("'")
    if output_path:
        resolved_output_path = os.path.abspath(os.path.expandvars(os.path.expanduser(output_path)))
        root, ext = os.path.splitext(resolved_output_path)
        if not ext:
            resolved_output_path = root + ".mp4"
        output_dir = os.path.dirname(resolved_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return resolved_output_path, os.path.basename(resolved_output_path)

    output_dir = folder_paths.get_output_directory()
    try:
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
    except TypeError:
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, 0, 0)

    os.makedirs(full_output_folder, exist_ok=True)
    output_filename = f"{filename}_{counter:05d}.mp4"
    return os.path.join(full_output_folder, output_filename), output_filename


def build_output_file_ui_entry(file_path, *, media_format=None, media_type="output"):
    output_root = os.path.abspath(folder_paths.get_output_directory())
    absolute_path = os.path.abspath(file_path)
    try:
        relative_path = os.path.relpath(absolute_path, output_root)
    except ValueError:
        relative_path = os.path.basename(absolute_path)

    relative_path = str(relative_path).replace("\\", "/")
    if not relative_path or relative_path == ".":
        relative_path = os.path.basename(absolute_path)

    entry = {
        "filename": relative_path,
        "subfolder": "",
        "type": media_type,
        "fullpath": absolute_path,
    }
    if media_format:
        entry["format"] = media_format
    return entry


def save_audio_input(audio, target_audio_path):
    if audio is None:
        raise ValueError("Pililink: audio input is required")

    # ComfyUI may execute nodes under torch.inference_mode().
    # Some audio ops (for example torchaudio resampling) expect normal tensors.
    with torch.inference_mode(False):
        waveform = audio["waveform"].detach().clone().cpu()
        sample_rate = int(audio["sample_rate"])
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        torchaudio.save(target_audio_path, waveform.cpu(), sample_rate)
        return {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": sample_rate,
        }


def run_process_with_interrupt(command, error_message, shell=False):
    process = None
    stdout = ""
    stderr = ""
    try:
        throw_if_processing_interrupted()
        process = subprocess.Popen(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        while process.poll() is None:
            throw_if_processing_interrupted()
            time.sleep(0.1)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            stderr_message = (stderr or stdout or "").strip()
            if stderr_message:
                raise RuntimeError(f"{error_message} (exit code {process.returncode}): {stderr_message}")
            raise RuntimeError(f"{error_message} (exit code {process.returncode})")
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except Exception:
                process.kill()
        raise


def run_ffmpeg_video_command_with_fallback(command_builder, error_message):
    preferred_codec = get_preferred_ffmpeg_video_codec()
    codecs_to_try = [preferred_codec]
    if preferred_codec != "libx264":
        codecs_to_try.append("libx264")

    last_error = None
    for codec in codecs_to_try:
        try:
            run_process_with_interrupt(command_builder(codec), error_message)
            if codec != preferred_codec:
                print(f"Pililink: ffmpeg video encoder fallback succeeded with {codec}")
            return
        except RuntimeError as exc:
            last_error = exc
            if codec != "libx264":
                print(f"Pililink: ffmpeg video encoder {codec} failed, retrying with libx264")
                continue
            raise

    if last_error is not None:
        raise last_error


def extract_audio_from_video(video_path, target_audio_path):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        target_audio_path,
    ]
    run_process_with_interrupt(command, f"Failed to extract audio from video: {video_path}")
    if not os.path.exists(target_audio_path):
        raise RuntimeError(f"Pililink: extracted audio file missing: {target_audio_path}")
    return target_audio_path


def run_small_process_with_interrupt_capture(command, error_message, shell=False):
    process = None
    stdout = ""
    stderr = ""
    try:
        throw_if_processing_interrupted()
        process = subprocess.Popen(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        while process.poll() is None:
            throw_if_processing_interrupted()
            time.sleep(0.1)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            stderr_message = (stderr or "").strip()
            if stderr_message:
                raise RuntimeError(f"{error_message} (exit code {process.returncode}): {stderr_message}")
            raise RuntimeError(f"{error_message} (exit code {process.returncode})")
        return stdout
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except Exception:
                process.kill()
        raise


def probe_media_duration(media_path, media_kind):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        media_path,
    ]
    output = run_small_process_with_interrupt_capture(
        command,
        f"Failed to probe {media_kind} duration: {media_path}",
    )
    try:
        duration = float(str(output).strip())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Pililink: invalid {media_kind} duration reported for {media_path}: {output!r}") from exc
    if duration <= 0:
        raise RuntimeError(f"Pililink: {media_kind} duration must be positive: {media_path}")
    return duration


def adjust_audio_duration(audio_path, target_audio_path, target_duration):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        audio_path,
        "-af",
        "apad",
        "-t",
        f"{target_duration:.6f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        target_audio_path,
    ]
    run_process_with_interrupt(command, f"Failed to adjust audio duration: {audio_path}")
    if not os.path.exists(target_audio_path):
        raise RuntimeError(f"Pililink: adjusted audio file missing: {target_audio_path}")
    return target_audio_path


def trim_video_to_duration(video_path, target_video_path, target_duration):
    def build_command(codec):
        return [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            video_path,
            "-t",
            f"{target_duration:.6f}",
            "-an",
            *get_ffmpeg_video_encode_args(codec),
            target_video_path,
        ]

    run_ffmpeg_video_command_with_fallback(build_command, f"Failed to trim video duration: {video_path}")
    if not os.path.exists(target_video_path):
        raise RuntimeError(f"Pililink: trimmed video file missing: {target_video_path}")
    return target_video_path


def loop_video_to_duration(video_path, target_video_path, target_duration):
    def build_command(codec):
        return [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-nostdin",
            "-stream_loop",
            "-1",
            "-i",
            video_path,
            "-t",
            f"{target_duration:.6f}",
            "-an",
            *get_ffmpeg_video_encode_args(codec),
            target_video_path,
        ]

    run_ffmpeg_video_command_with_fallback(build_command, f"Failed to loop video to target duration: {video_path}")
    if not os.path.exists(target_video_path):
        raise RuntimeError(f"Pililink: looped video file missing: {target_video_path}")
    return target_video_path


def create_pingpong_cycle(video_path, cycle_video_path):
    def build_command(codec):
        return [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            video_path,
            "-filter_complex",
            "[0:v]split[fwd][tmp];[tmp]reverse[rev];[fwd][rev]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-an",
            *get_ffmpeg_video_encode_args(codec),
            cycle_video_path,
        ]

    run_ffmpeg_video_command_with_fallback(build_command, f"Failed to create pingpong video cycle: {video_path}")
    if not os.path.exists(cycle_video_path):
        raise RuntimeError(f"Pililink: pingpong cycle video missing: {cycle_video_path}")
    return cycle_video_path


def match_path_node_lengths(video_path, audio_path, mode, silent_padding_sec, temp_dir, auto_silent_padding=False):
    duration_epsilon = 0.05
    video_duration = probe_media_duration(video_path, "video")
    audio_duration = probe_media_duration(audio_path, "audio")

    resolved_mode = str(mode or "normal")
    if resolved_mode not in {"normal", "pingpong", "loop_to_audio"}:
        raise ValueError(f"Pililink: unsupported length mode: {resolved_mode}")

    resolved_video_path = video_path
    resolved_audio_path = audio_path
    manual_padding_sec = max(0.0, float(silent_padding_sec or 0.0))
    effective_padding_sec = max(0.0, video_duration - audio_duration) if auto_silent_padding else manual_padding_sec

    def needs_adjustment(current_duration, target_duration):
        return abs(current_duration - target_duration) > duration_epsilon

    def build_temp_path(stem):
        return os.path.join(temp_dir, stem)

    if resolved_mode == "normal":
        target_duration = min(video_duration, audio_duration + effective_padding_sec)
        # The pipeline already aligns output length to the shortest valid stream,
        # so avoid an eager FFmpeg trim in normal mode unless the user explicitly
        # asked for additional silence padding.
        if effective_padding_sec > duration_epsilon and needs_adjustment(audio_duration, target_duration):
            resolved_audio_path = adjust_audio_duration(
                audio_path,
                build_temp_path("pililink_length_normal_audio.wav"),
                target_duration,
            )
    elif resolved_mode == "pingpong":
        if audio_duration <= video_duration + duration_epsilon:
            target_duration = video_duration
            if needs_adjustment(audio_duration, target_duration):
                resolved_audio_path = adjust_audio_duration(
                    audio_path,
                    build_temp_path("pililink_length_pingpong_audio.wav"),
                    target_duration,
                )
        else:
            target_duration = audio_duration + effective_padding_sec
            resolved_audio_path = adjust_audio_duration(
                audio_path,
                build_temp_path("pililink_length_pingpong_audio.wav"),
                target_duration,
            )
            pingpong_cycle_path = create_pingpong_cycle(
                video_path,
                build_temp_path("pililink_length_pingpong_cycle.mp4"),
            )
            resolved_video_path = loop_video_to_duration(
                pingpong_cycle_path,
                build_temp_path("pililink_length_pingpong_video.mp4"),
                target_duration,
            )
    else:
        target_duration = audio_duration + effective_padding_sec
        if needs_adjustment(audio_duration, target_duration):
            resolved_audio_path = adjust_audio_duration(
                audio_path,
                build_temp_path("pililink_length_loop_audio.wav"),
                target_duration,
            )
        if target_duration < video_duration - duration_epsilon:
            resolved_video_path = trim_video_to_duration(
                video_path,
                build_temp_path("pililink_length_loop_video.mp4"),
                target_duration,
            )
        elif target_duration > video_duration + duration_epsilon:
            resolved_video_path = loop_video_to_duration(
                video_path,
                build_temp_path("pililink_length_loop_video.mp4"),
                target_duration,
            )

    print(
        "Pililink: Length mode "
        f"{resolved_mode}, video {video_duration:.3f}s, audio {audio_duration:.3f}s, "
        f"silent_padding {effective_padding_sec:.3f}s"
    )
    if resolved_video_path != video_path or resolved_audio_path != audio_path:
        print(
            "Pililink: Prepared aligned inputs "
            f"video={resolved_video_path}, audio={resolved_audio_path}"
        )

    return resolved_video_path, resolved_audio_path


class PililinkLatentSyncBase:
    def __init__(self):
        # Make sure our temp directory is the current one
        global MODULE_TEMP_DIR
        if not os.path.exists(MODULE_TEMP_DIR):
            os.makedirs(MODULE_TEMP_DIR, exist_ok=True)

    def _maybe_cuda_empty_cache(self, execution_settings=None, force=False):
        if not torch.cuda.is_available():
            return
        if force:
            torch.cuda.empty_cache()
            return
        aggressive = True
        if isinstance(execution_settings, dict):
            aggressive = bool(execution_settings.get("aggressive_cuda_cache_clear", True))
        if aggressive:
            torch.cuda.empty_cache()

    @staticmethod
    def _load_audio_only(video_path):
        """Load only the audio track from a video file via FFmpeg, without reading frames."""
        temp_wav = video_path + ".audio_extract.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-nostdin",
                 "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", temp_wav],
                check=True,
            )
            import torchaudio
            waveform, sr = torchaudio.load(temp_wav)
            waveform = waveform.to(torch.float32).cpu().unsqueeze(0)
            return {"waveform": waveform, "sample_rate": sr}
        except Exception as e:
            print(f"[Pililink] Audio extraction failed: {e}")
            return {"waveform": torch.zeros((1, 1, 0), dtype=torch.float32), "sample_rate": 16000}
        finally:
            try:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except Exception:
                pass

    def _prepare_execution_settings(self, inference_steps, vram_usage, quality_mode="balanced"):
        resolved_quality_mode = str(quality_mode or "balanced").lower()
        if resolved_quality_mode not in {"balanced", "quality_first"}:
            resolved_quality_mode = "balanced"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        supports_fp16 = False
        if device.type == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Turing (7.x) and newer GPUs can run fp16 inference efficiently.
            supports_fp16 = int(capability[0]) >= 7

        vram_mode = str(vram_usage or "medium").lower()
        use_mixed_precision = (
            resolved_quality_mode != "quality_first"
            and supports_fp16
            and vram_mode in {"high", "medium"}
        )
        dtype = torch.float16 if use_mixed_precision else torch.float32

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = resolved_quality_mode != "quality_first"
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = resolved_quality_mode != "quality_first"
            torch.backends.cudnn.allow_tf32 = resolved_quality_mode != "quality_first"
            torch.cuda.empty_cache()
        else:
            print("No GPU detected, using CPU execution settings")

        return {
            "device": device,
            "batch_size": 1,
            "use_mixed_precision": use_mixed_precision,
            "dtype": dtype,
            "aggressive_cuda_cache_clear": False,
        }

    def _create_run_temp_dir(self):
        global MODULE_TEMP_DIR
        run_id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_dir = os.path.join(MODULE_TEMP_DIR, f"pililink_run_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        return run_id, temp_dir

    @staticmethod
    def _is_probably_25fps_video(video_path):
        return is_probably_25fps_video(video_path)

    @staticmethod
    def _resolve_mask_image_path():
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        default_mask_path = os.path.join(cur_dir, "latentsync", "utils", "mask.png")
        if os.path.exists(default_mask_path):
            return default_mask_path
        alt_mask_path = os.path.join(cur_dir, "utils", "mask.png")
        if os.path.exists(alt_mask_path):
            return alt_mask_path
        return default_mask_path

    @staticmethod
    def _write_temp_video_with_progress(frames_cpu, target_path, fps, progress=None, start_fraction=0.0, end_fraction=1.0):
        global av_mod

        if torchvision_io is not None:
            try:
                torchvision_io.write_video(target_path, frames_cpu, fps=fps, video_codec="h264")
                if progress is not None:
                    progress.update_fraction(end_fraction, stage="Temporary video ready")
                return
            except (TypeError, RuntimeError, ValueError, AttributeError):
                pass

        if av_mod is None:
            av_mod = __import__("av")

        container = av_mod.open(target_path, mode="w")
        try:
            stream = container.add_stream("h264", rate=fps)
            stream.width = frames_cpu.shape[2]
            stream.height = frames_cpu.shape[1]

            total_frames = int(frames_cpu.shape[0])
            for frame_index, frame in enumerate(frames_cpu):
                throw_if_processing_interrupted()
                video_frame = av_mod.VideoFrame.from_ndarray(frame.numpy(), format="rgb24")
                packet = stream.encode(video_frame)
                container.mux(packet)

                if progress is not None and total_frames > 0:
                    if frame_index + 1 == total_frames or ((frame_index + 1) % 8 == 0):
                        progress.update_stage_fraction(
                            start_fraction,
                            end_fraction,
                            (frame_index + 1) / total_frames,
                            stage="Writing temporary video",
                        )

            packet = stream.encode(None)
            container.mux(packet)
        finally:
            container.close()

    def _run_inference(
        self,
        video_path,
        audio_path,
        output_video_path,
        seed,
        lips_expression,
        inference_steps,
        segment_inferences,
        temp_dir,
        execution_settings,
        deepcache="on",
        deepcache_cache_interval=3,
        deepcache_branch_id=0,
        scheduler_type="ddim",
        affine_detect_interval=1,
        skip_video_normalization=False,
        clip_batch_size=1,
        auto_oom_fallback=True,
        quality_mode="balanced",
        progress_callback=None,
    ):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.dirname(cur_dir)
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        if cur_dir not in sys.path:
            sys.path.insert(0, cur_dir)

        runtime_mod = import_refactor_runtime_module()
        config_path = os.path.join(cur_dir, "configs", "unet", "stage2.yaml")
        scheduler_path = os.path.join(cur_dir, "configs", "scheduler")
        ckpt_path = get_latentsync_path("latentsync_unet.pt")
        mask_image_path = self._resolve_mask_image_path()
        resolved_scheduler_type = str(scheduler_type or "ddim").strip().lower()
        if resolved_scheduler_type not in {"ddim", "dpm_solver"}:
            resolved_scheduler_type = "ddim"

        runtime_mod.run_refactor_inference(
            config_path=config_path,
            scheduler_path=scheduler_path,
            inference_ckpt_path=ckpt_path,
            latentsync_root=LATENTSYNC_ROOT_DIR,
            audio_embeds_cache_dir=get_latentsync_path("runtime_cache", "audio_embeds"),
            vae_model_path=get_latentsync_path("vae", "sd-vae-ft-mse"),
            video_path=video_path,
            audio_path=audio_path,
            output_video_path=output_video_path,
            seed=int(seed),
            inference_steps=max(1, int(inference_steps)),
            guidance_scale=float(lips_expression),
            segment_inferences=max(2, int(segment_inferences)),
            temp_dir=temp_dir,
            device=execution_settings["device"],
            dtype=execution_settings.get("dtype", torch.float32),
            mask_image_path=mask_image_path,
            deepcache=deepcache,
            deepcache_cache_interval=max(1, int(deepcache_cache_interval)),
            deepcache_branch_id=max(0, int(deepcache_branch_id)),
            scheduler_type=resolved_scheduler_type,
            skip_video_normalization=bool(
                skip_video_normalization or self._is_probably_25fps_video(video_path)
            ),
            clip_batch_size=max(1, int(clip_batch_size)),
            auto_oom_fallback=bool(auto_oom_fallback),
            quality_mode=str(quality_mode or "balanced").lower(),
            affine_detect_interval=max(1, int(affine_detect_interval)),
            progress_callback=progress_callback,
        )


class PililinkLatentSyncNode(PililinkLatentSyncBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "images": ("IMAGE",),
                    "audio": ("AUDIO", ),
                    "seed": ("INT", {"default": 1247}),
                "lips_expression": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                "inference_steps": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),
                "vram_usage": (["high", "medium", "low"], {"default": "medium"}),
                "segment_inferences": ("INT", {"default": 16, "min": 2, "max": 256, "step": 2}),
                "clip_batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "auto_oom_fallback": ("BOOLEAN", {"default": True}),
                "quality_mode": (["balanced", "quality_first"], {"default": "balanced"}),
                "deepcache": (["on", "off"], {"default": "on"}),
                "deepcache_cache_interval": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1}),
                "deepcache_branch_id": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1}),
                "scheduler_type": (["ddim", "dpm_solver"], {"default": "ddim"}),
                "affine_detect_interval": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                 },
                "hidden": {"node_id": "UNIQUE_ID"}}

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio") 
    FUNCTION = "inference"

    def inference(
        self,
        images,
        audio,
        seed,
        lips_expression=1.5,
        inference_steps=20,
        vram_usage="medium",
        segment_inferences=16,
        clip_batch_size=1,
        auto_oom_fallback=True,
        quality_mode="balanced",
        deepcache="on",
        deepcache_cache_interval=3,
        deepcache_branch_id=0,
        scheduler_type="ddim",
        affine_detect_interval=1,
        node_id=None,
    ):
        throw_if_processing_interrupted()
        ensure_runtime_ready()
        progress = PililinkProgressReporter(node_id=node_id)
        progress.update_fraction(0.01, stage="Initializing")
        # Add timing information
        start_time = time.time()
        
        # Use our module temp directory
        global MODULE_TEMP_DIR
        
        # Define timing checkpoint function
        def log_timing(step):
            elapsed = time.time() - start_time
            print(f"[Pililink {elapsed:.2f}s] {step}")
        
        log_timing("Starting Pililink inference")
        
        execution_settings = self._prepare_execution_settings(
            inference_steps,
            vram_usage,
            quality_mode=quality_mode,
        )
        run_id, temp_dir = self._create_run_temp_dir()
        progress.update_fraction(0.05, stage="Preparing inputs")

        temp_video_path = None
        output_video_path = None
        audio_path = None

        try:
            global av_mod
            # Create temporary file paths in our system temp directory
            temp_video_path = os.path.join(temp_dir, f"pililink_temp_{run_id}.mp4")
            output_video_path = os.path.join(temp_dir, f"pililink_latentsync_{run_id}_out.mp4")
            audio_path = os.path.join(temp_dir, f"pililink_latentsync_{run_id}_audio.wav")
            
            log_timing("Processing input frames")
            throw_if_processing_interrupted()
            # Process input frames
            if isinstance(images, list):
                frames = torch.stack(images)
            else:
                frames = images
            frames = (frames * 255).byte().cpu()

            # Process audio data to get expected frame count for a single image
            waveform = audio["waveform"].cpu()
            sample_rate = audio["sample_rate"]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
                
            # Check if we have a single image (either as a batch of 1 or a single 3D tensor)
            is_single_image = False
            if len(frames.shape) == 3:  # Single 3D tensor (H,W,C)
                frames = frames.unsqueeze(0)
                is_single_image = True
            elif frames.shape[0] == 1:  # Batch of 1
                is_single_image = True
                
            # If it's a single image, duplicate it to match audio duration
            if is_single_image:
                # Calculate audio duration in seconds
                audio_duration = waveform.shape[1] / sample_rate
                
                # Calculate how many frames we need at 25fps (standard for this model)
                required_frames = math.ceil(audio_duration * 25)
                
                # Duplicate the single frame to match required frame count
                # (minimum 4 frames to avoid tensor stack issues)
                required_frames = max(required_frames, 4)
                single_frame = frames[0]
                duplicated_frames = single_frame.unsqueeze(0).repeat(required_frames, 1, 1, 1)
                frames = duplicated_frames
                print(f"Pililink: Duplicated single image to create {required_frames} frames matching audio duration")
            progress.update_fraction(0.12, stage="Frames ready")

            log_timing("Processing audio")
            throw_if_processing_interrupted()
            log_timing("Saving temporary files")
            throw_if_processing_interrupted()
            resampled_audio = save_audio_input(audio, audio_path)
            progress.update_fraction(0.18, stage="Audio ready")

            # Move frames to CPU for saving to video
            frames_cpu = frames.cpu()
            self._write_temp_video_with_progress(
                frames_cpu,
                temp_video_path,
                fps=25,
                progress=progress,
                start_fraction=0.18,
                end_fraction=0.30,
            )
            
            # Free up memory after saving
            del frames_cpu
            self._maybe_cuda_empty_cache(execution_settings)

            log_timing("Running Pililink inference")
            throw_if_processing_interrupted()
            self._run_inference(
                video_path=temp_video_path,
                audio_path=audio_path,
                output_video_path=output_video_path,
                seed=seed,
                lips_expression=lips_expression,
                inference_steps=inference_steps,
                segment_inferences=segment_inferences,
                temp_dir=temp_dir,
                execution_settings=execution_settings,
                deepcache=deepcache,
                deepcache_cache_interval=deepcache_cache_interval,
                deepcache_branch_id=deepcache_branch_id,
                scheduler_type=scheduler_type,
                affine_detect_interval=affine_detect_interval,
                skip_video_normalization=True,
                clip_batch_size=clip_batch_size,
                auto_oom_fallback=auto_oom_fallback,
                quality_mode=quality_mode,
                progress_callback=progress.make_stage_callback(0.30, 0.92),
            )

            log_timing("Processing output")
            throw_if_processing_interrupted()
            progress.update_fraction(0.94, stage="Loading output video")
            # Clean GPU cache after inference
            self._maybe_cuda_empty_cache(execution_settings)

            # Verify output file exists
            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"Output video not found at: {output_video_path}")
            
            # Read the processed video - ensure it's loaded as CPU tensor
            if torchvision_io is None:
                raise RuntimeError("torchvision.io is required for reading output video")
            processed_frames = torchvision_io.read_video(output_video_path, pts_unit='sec')[0]
            processed_frames = processed_frames.float() / 255.0
            progress.update_fraction(0.99, stage="Finalizing outputs")

            # Ensure audio is on CPU before returning
            if torch.cuda.is_available():
                if isinstance(resampled_audio["waveform"], torch.Tensor) and resampled_audio["waveform"].device.type == 'cuda':
                    resampled_audio["waveform"] = resampled_audio["waveform"].cpu()
                if isinstance(processed_frames, torch.Tensor) and processed_frames.device.type == 'cuda':
                    processed_frames = processed_frames.cpu()

            total_time = time.time() - start_time
            print(f"Pililink total processing time: {total_time:.2f}s")
            progress.update_fraction(1.0, stage="Done")
            
            return (processed_frames, resampled_audio)

        except RuntimeError as e:
            if "Face not detected" in str(e) or "未检测到人脸" in str(e):
                print(f"[Pililink] Face detection failed: {str(e)}")
                raise RuntimeError(
                    "未检测到人脸：输入视频中未能识别到人脸，请检查输入视频是否包含清晰的正面人脸。\n"
                    "Face not detected: No face was found in the input video. "
                    "Please ensure the video contains a clearly visible frontal face."
                ) from e
            print(f"Error during Pililink inference: {str(e)}")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"Error during Pililink inference: {str(e)}")
            traceback.print_exc()
            raise

        finally:
            # Cleanup GPU memory
            self._maybe_cuda_empty_cache(execution_settings)
                
            # Only remove temporary files if successful (keep for debugging if failed)
            try:
                # Clean up temporary files individually
                for path in [temp_video_path, output_video_path, audio_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass  # Ignore cleanup errors

            # Final GPU cache cleanup
            self._maybe_cuda_empty_cache(execution_settings)


class PililinkLatentSyncVideoPathNode(PililinkLatentSyncBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False, "placeholder": "D:/videos/input.mp4"}),
                "seed": ("INT", {"default": PATH_NODE_DEFAULTS["seed"]}),
                "lips_expression": ("FLOAT", {"default": PATH_NODE_DEFAULTS["lips_expression"], "min": 1.0, "max": 3.0, "step": 0.1}),
                "inference_steps": ("INT", {"default": PATH_NODE_DEFAULTS["inference_steps"], "min": 1, "max": 999, "step": 1}),
                "vram_usage": (["high", "medium", "low"], {"default": PATH_NODE_DEFAULTS["vram_usage"]}),
                "segment_inferences": ("INT", {"default": PATH_NODE_DEFAULTS["segment_inferences"], "min": 2, "max": 256, "step": 2}),
                "clip_batch_size": ("INT", {"default": PATH_NODE_DEFAULTS["clip_batch_size"], "min": 1, "max": 16, "step": 1}),
                "auto_oom_fallback": ("BOOLEAN", {"default": PATH_NODE_DEFAULTS["auto_oom_fallback"]}),
                "quality_mode": (["balanced", "quality_first"], {"default": PATH_NODE_DEFAULTS["quality_mode"]}),
                "deepcache": (["on", "off"], {"default": PATH_NODE_DEFAULTS["deepcache"]}),
                "deepcache_cache_interval": ("INT", {"default": PATH_NODE_DEFAULTS["deepcache_cache_interval"], "min": 1, "max": 16, "step": 1}),
                "deepcache_branch_id": ("INT", {"default": PATH_NODE_DEFAULTS["deepcache_branch_id"], "min": 0, "max": 4, "step": 1}),
                "scheduler_type": (["ddim", "dpm_solver"], {"default": PATH_NODE_DEFAULTS["scheduler_type"]}),
                "affine_detect_interval": ("INT", {"default": PATH_NODE_DEFAULTS["affine_detect_interval"], "min": 1, "max": 8, "step": 1}),
                "mode": (["normal", "pingpong", "loop_to_audio"], {"default": PATH_NODE_DEFAULTS["mode"]}),
                "silent_padding_sec": ("FLOAT", {"default": PATH_NODE_DEFAULTS["silent_padding_sec"], "min": 0.0, "max": 3.0, "step": 0.1}),
                "auto_silent_padding": ("BOOLEAN", {"default": PATH_NODE_DEFAULTS["auto_silent_padding"]}),
                "filename_prefix": ("STRING", {"default": PATH_NODE_DEFAULTS["filename_prefix"]}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Leave empty to use audio input or the video's original audio"}),
                "output_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional absolute output .mp4 path"}),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "video_path", "filename")
    FUNCTION = "inference_from_path"
    OUTPUT_NODE = True

    def inference_from_path(
        self,
        video_path,
        seed,
        lips_expression=PATH_NODE_DEFAULTS["lips_expression"],
        inference_steps=PATH_NODE_DEFAULTS["inference_steps"],
        vram_usage=PATH_NODE_DEFAULTS["vram_usage"],
        segment_inferences=PATH_NODE_DEFAULTS["segment_inferences"],
        clip_batch_size=PATH_NODE_DEFAULTS["clip_batch_size"],
        auto_oom_fallback=PATH_NODE_DEFAULTS["auto_oom_fallback"],
        quality_mode=PATH_NODE_DEFAULTS["quality_mode"],
        deepcache=PATH_NODE_DEFAULTS["deepcache"],
        deepcache_cache_interval=PATH_NODE_DEFAULTS["deepcache_cache_interval"],
        deepcache_branch_id=PATH_NODE_DEFAULTS["deepcache_branch_id"],
        scheduler_type=PATH_NODE_DEFAULTS["scheduler_type"],
        affine_detect_interval=PATH_NODE_DEFAULTS["affine_detect_interval"],
        mode=PATH_NODE_DEFAULTS["mode"],
        silent_padding_sec=PATH_NODE_DEFAULTS["silent_padding_sec"],
        auto_silent_padding=PATH_NODE_DEFAULTS["auto_silent_padding"],
        filename_prefix=PATH_NODE_DEFAULTS["filename_prefix"],
        audio=None,
        audio_path="",
        output_path="",
        node_id=None,
    ):
        throw_if_processing_interrupted()
        ensure_runtime_ready()
        progress = PililinkProgressReporter(node_id=node_id)
        progress.update_fraction(0.01, stage="Initializing")
        start_time = time.time()
        resolved_video_path = resolve_user_path(video_path, "video_path")

        execution_settings = self._prepare_execution_settings(
            inference_steps,
            vram_usage,
            quality_mode=quality_mode,
        )
        run_id, temp_dir = self._create_run_temp_dir()
        progress.update_fraction(0.05, stage="Resolving media")

        temp_audio_path = os.path.join(temp_dir, f"pililink_latentsync_{run_id}_audio.wav")
        final_output_path, output_filename = get_output_video_path(filename_prefix, output_path)

        try:
            audio_path_text = str(audio_path or "").strip()
            if audio_path_text:
                resolved_audio_path = resolve_user_path(audio_path_text, "audio_path")
            elif audio is not None:
                save_audio_input(audio, temp_audio_path)
                resolved_audio_path = temp_audio_path
            else:
                print("Pililink: No audio input provided, extracting audio track from source video")
                resolved_audio_path = extract_audio_from_video(resolved_video_path, temp_audio_path)
            progress.update_fraction(0.12, stage="Audio ready")

            aligned_video_path, aligned_audio_path = match_path_node_lengths(
                resolved_video_path,
                resolved_audio_path,
                mode,
                silent_padding_sec,
                temp_dir,
                auto_silent_padding=auto_silent_padding,
            )
            progress.update_fraction(0.20, stage="Media aligned")

            print(f"Pililink: Processing source video path {resolved_video_path}")
            print(f"Pililink: Saving output video to {final_output_path}")

            self._run_inference(
                video_path=aligned_video_path,
                audio_path=aligned_audio_path,
                output_video_path=final_output_path,
                seed=seed,
                lips_expression=lips_expression,
                inference_steps=inference_steps,
                segment_inferences=segment_inferences,
                temp_dir=temp_dir,
                execution_settings=execution_settings,
                deepcache=deepcache,
                deepcache_cache_interval=deepcache_cache_interval,
                deepcache_branch_id=deepcache_branch_id,
                scheduler_type=scheduler_type,
                affine_detect_interval=affine_detect_interval,
                skip_video_normalization=is_probably_25fps_video(aligned_video_path),
                clip_batch_size=clip_batch_size,
                auto_oom_fallback=auto_oom_fallback,
                quality_mode=quality_mode,
                progress_callback=progress.make_stage_callback(0.20, 0.95),
            )

            total_time = time.time() - start_time
            print(f"Pililink path-node total processing time: {total_time:.2f}s")

            # Release GPU memory before building output
            self._maybe_cuda_empty_cache(execution_settings, force=True)

            # Load only audio from the output video (no IMAGE tensor to avoid OOM)
            progress.update_fraction(0.97, stage="Loading output audio")
            output_audio = self._load_audio_only(final_output_path)
            progress.update_fraction(1.0, stage="Done")
            video_preview = build_output_file_ui_entry(
                final_output_path,
                media_format="video/mp4",
                media_type="output",
            )
            return {
                "ui": {
                    "text": [final_output_path],
                    "gifs": [video_preview],
                },
                "result": (
                    output_audio,
                    final_output_path,
                    output_filename,
                ),
            }
        except RuntimeError as e:
            if "Face not detected" in str(e) or "未检测到人脸" in str(e):
                print(f"[Pililink] Face detection failed: {str(e)}")
                raise RuntimeError(
                    "未检测到人脸：输入视频中未能识别到人脸，请检查输入视频是否包含清晰的正面人脸。\n"
                    "Face not detected: No face was found in the input video. "
                    "Please ensure the video contains a clearly visible frontal face."
                ) from e
            print(f"Error during Pililink path inference: {str(e)}")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"Error during Pililink path inference: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            try:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


class PililinkVideoLengthAdjuster:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                "silent_padding_sec": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1}),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "adjust"

    def _check_memory_capacity(self, num_frames, frame_shape):
        """Pre-check if we have enough memory for the target frames"""
        try:
            import psutil
            # Rough estimate: each frame ~1-4MB depending on resolution
            bytes_per_frame = frame_shape[0] * frame_shape[1] * frame_shape[2] * 4  # float32
            required_bytes = num_frames * bytes_per_frame
            required_mb = required_bytes / (1024 * 1024)
            
            # Get available memory
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            threshold_mb = max(1000, available_mb * 0.1)  # Keep 10% free or 1GB minimum
            
            if required_mb > available_mb - threshold_mb:
                print(f"[Pililink WARNING] Estimated memory needed: {required_mb:.0f}MB, Available: {available_mb:.0f}MB")
                print(f"[Pililink WARNING] Processing may be slow or fail. Consider shorter video duration.")
                return False
            return True
        except (ImportError, Exception) as e:
            # If psutil not available, do a basic memory check
            if torch.cuda.is_available():
                try:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    print(f"[Pililink INFO] GPU memory: {gpu_mem:.1f}GB, Processing {num_frames} frames")
                except:
                    pass
            return True

    def _expand_frames_efficient(self, frame_indices, num_frames, total_frames_needed, fps):
        """Efficiently expand frames using indexing instead of list concatenation
        
        Args:
            frame_indices: List of indices to access frames
            num_frames: Original number of frames
            total_frames_needed: Target number of frames
            fps: Frames per second (for progress reporting)
        
        Returns:
            List of indices that efficiently tile the frames
        """
        if num_frames == 0:
            return []
        
        if total_frames_needed <= num_frames:
            return frame_indices[:total_frames_needed]
        
        # Use numpy for efficient indexing
        indices = np.tile(frame_indices, math.ceil(total_frames_needed / num_frames))
        return indices[:total_frames_needed].tolist()

    def _create_pingpong_sequence(self, num_frames):
        """Create a pingpong sequence: forward then backward"""
        if num_frames <= 1:
            return list(range(num_frames))
        
        # Forward sequence
        forward = list(range(num_frames))
        # Backward sequence (excluding first and last to avoid duplication)
        backward = list(range(num_frames - 2, 0, -1))
        
        if len(backward) == 0:
            return forward

        return forward + backward

    def _materialize_frames_by_indices(
        self,
        original_frames,
        final_indices,
        progress=None,
        start_fraction=0.0,
        end_fraction=1.0,
        stage="Assembling frames",
    ):
        adjusted_frames = []
        total = len(final_indices)
        for idx, frame_index in enumerate(final_indices):
            throw_if_processing_interrupted()
            adjusted_frames.append(original_frames[frame_index])
            if progress is not None and total > 0:
                if idx + 1 == total or ((idx + 1) % 8 == 0):
                    progress.update_stage_fraction(
                        start_fraction,
                        end_fraction,
                        (idx + 1) / total,
                        stage=stage,
                    )
        return adjusted_frames

    def adjust(self, images, audio, mode, fps=25.0, silent_padding_sec=0.5, node_id=None):
        throw_if_processing_interrupted()  # Add interruption check at start
        progress = PililinkProgressReporter(node_id=node_id)
        progress.update_fraction(0.02, stage="Preparing inputs")
        
        waveform = audio["waveform"].squeeze(0)
        sample_rate = int(audio["sample_rate"])
        
        # Convert images tensor to list of frames
        if isinstance(images, torch.Tensor):
            original_frames = [images[i] for i in range(images.shape[0])]
        else:
            original_frames = images.copy()
        
        num_original_frames = len(original_frames)
        progress.update_fraction(0.10, stage="Inputs ready")
        
        if num_original_frames == 0:
            raise ValueError("Pililink: Input images cannot be empty")
        
        # Pre-check memory capacity
        frame_shape = original_frames[0].shape if original_frames else images.shape[1:]
        
        if mode == "normal":
            throw_if_processing_interrupted()
            progress.update_fraction(0.25, stage="Adjusting duration")
            # Add silent padding to the audio and then trim video to match
            audio_duration = waveform.shape[1] / sample_rate
            
            # Add silent padding to the audio
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            
            # Calculate required frames based on the padded audio
            padded_audio_duration = (waveform.shape[1] + silence_samples) / sample_rate
            required_frames = int(padded_audio_duration * fps)
            
            if len(original_frames) > required_frames:
                # Trim video frames to match padded audio duration
                adjusted_frames = original_frames[:required_frames]
            else:
                # If video is shorter than padded audio, keep all video frames
                # and trim the audio accordingly
                adjusted_frames = original_frames
                required_samples = int(len(original_frames) / fps * sample_rate)
                padded_audio = padded_audio[:, :required_samples]
            
            throw_if_processing_interrupted()
            progress.update_fraction(1.0, stage="Done")
            return (
                torch.stack(adjusted_frames),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

        elif mode == "pingpong":
            throw_if_processing_interrupted()
            progress.update_fraction(0.25, stage="Adjusting duration")
            video_duration = len(original_frames) / fps
            audio_duration = waveform.shape[1] / sample_rate
            
            if audio_duration <= video_duration:
                throw_if_processing_interrupted()
                required_samples = int(video_duration * sample_rate)
                silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
                adjusted_audio = torch.cat([waveform, silence], dim=1)

                progress.update_fraction(1.0, stage="Done")
                return (
                    torch.stack(original_frames),
                    {"waveform": adjusted_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

            else:
                throw_if_processing_interrupted()
                silence_samples = math.ceil(silent_padding_sec * sample_rate)
                silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
                padded_audio = torch.cat([waveform, silence], dim=1)
                total_duration = (waveform.shape[1] + silence_samples) / sample_rate
                target_frames = math.ceil(total_duration * fps)
                
                # Check memory before processing
                self._check_memory_capacity(target_frames, frame_shape)
                
                # Create pingpong sequence
                pingpong_indices = self._create_pingpong_sequence(num_original_frames)
                pingpong_cycle_len = len(pingpong_indices)
                
                if pingpong_cycle_len == 0:
                    raise ValueError("Pililink: Pingpong sequence is empty")
                
                # Use efficient indexed expansion
                final_indices = self._expand_frames_efficient(
                    pingpong_indices, pingpong_cycle_len, target_frames, fps
                )
                
                adjusted_frames = self._materialize_frames_by_indices(
                    original_frames,
                    final_indices[:target_frames],
                    progress=progress,
                    start_fraction=0.35,
                    end_fraction=0.95,
                    stage="Building pingpong frames",
                )
                
                throw_if_processing_interrupted()
                progress.update_fraction(1.0, stage="Done")
                return (
                    torch.stack(adjusted_frames),
                    {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

        elif mode == "loop_to_audio":
            throw_if_processing_interrupted()
            progress.update_fraction(0.25, stage="Adjusting duration")
            # Add silent padding then simple loop
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            total_duration = (waveform.shape[1] + silence_samples) / sample_rate
            target_frames = math.ceil(total_duration * fps)
            
            # Check memory before processing
            self._check_memory_capacity(target_frames, frame_shape)
            
            # Use efficient indexed expansion instead of list concatenation
            loop_indices = list(range(num_original_frames))
            final_indices = self._expand_frames_efficient(
                loop_indices, num_original_frames, target_frames, fps
            )
            
            adjusted_frames = self._materialize_frames_by_indices(
                original_frames,
                final_indices[:target_frames],
                progress=progress,
                start_fraction=0.35,
                end_fraction=0.95,
                stage="Looping frames",
            )
            
            throw_if_processing_interrupted()
            progress.update_fraction(1.0, stage="Done")
            return (
                torch.stack(adjusted_frames),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )


# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    NODE_KEY_MAIN: PililinkLatentSyncNode,
    NODE_KEY_MAIN_PATH: PililinkLatentSyncVideoPathNode,
    NODE_KEY_ADJUSTER: PililinkVideoLengthAdjuster,
}

# Display Names for ComfyUI - Clear distinction from original
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_KEY_MAIN: NODE_DISPLAY_MAIN,
    NODE_KEY_MAIN_PATH: NODE_DISPLAY_MAIN_PATH,
    NODE_KEY_ADJUSTER: NODE_DISPLAY_ADJUSTER,
}
