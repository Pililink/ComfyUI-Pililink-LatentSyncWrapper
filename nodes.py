import argparse
import atexit
import copy
import importlib.util
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

import cv2
import numpy as np
import requests
import torch
import torchaudio
from omegaconf import OmegaConf

# Global model cache to avoid reloading models - Using unique name to avoid conflicts
_PILILINK_MODEL_CACHE = {}
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
TORCH_CACHE_DIR = _LATENTSYNC_PATHS["torch_cache_dir"]

NODE_CATEGORY = "PililinkLatentSync"
NODE_KEY_MAIN = "PililinkLatentSyncNode"
NODE_KEY_ADJUSTER = "PililinkLatentSyncLengthAdjuster"
NODE_KEY_MAIN_PATH = "PililinkLatentSyncVideoPathNode"
NODE_KEY_MAIN_REFACTOR = "PililinkLatentSyncRefactorNode"
NODE_KEY_MAIN_PATH_REFACTOR = "PililinkLatentSyncRefactorVideoPathNode"
NODE_DISPLAY_MAIN = "Pililink LatentSync 1.5"
NODE_DISPLAY_ADJUSTER = "Pililink LatentSync Length Adjuster"
NODE_DISPLAY_MAIN_PATH = "Pililink LatentSync 1.5 (Video Path)"
NODE_DISPLAY_MAIN_REFACTOR = "Pililink LatentSync 1.5 (Refactor)"
NODE_DISPLAY_MAIN_PATH_REFACTOR = "Pililink LatentSync 1.5 (Video Path, Refactor Legacy)"
_FFMPEG_VIDEO_ENCODERS = None

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

def get_unique_temp_path(suffix=""):
    return os.path.join(get_comfy_temp_root(), f"pilkilink_latentsync_{uuid.uuid4().hex[:8]}{suffix}")


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
    global MODULE_TEMP_DIR, _PILILINK_MODEL_CACHE
    
    # Clear model cache references to free memory
    _PILILINK_MODEL_CACHE.clear()
    
    # Clean up temp directory (node-local session directory)
    if MODULE_TEMP_DIR and os.path.exists(MODULE_TEMP_DIR):
        try:
            shutil.rmtree(MODULE_TEMP_DIR, ignore_errors=True)
            print("Cleaned up node-local temp directory")
        except:
            pass

# Call this before anything else
MODULE_TEMP_DIR = init_temp_directories()

# Register the cleanup handler to run when Python exits
atexit.register(module_cleanup)

# Check for conflicts with other implementations
conflict_detected = check_for_conflicts()

comfy_model_management = None
try:
    comfy_model_management = __import__("comfy.model_management", fromlist=["throw_exception_if_processing_interrupted"])
except Exception:
    comfy_model_management = None


def throw_if_processing_interrupted():
    if comfy_model_management is not None:
        comfy_model_management.throw_exception_if_processing_interrupted()


def get_latentsync_root_dir(mkdir=True):
    """Return unified model root: ComfyUI/models/latensync1.5 (with legacy compatibility)."""
    root = LATENTSYNC_ROOT_DIR
    if mkdir:
        os.makedirs(root, exist_ok=True)
    return root


def get_latentsync_path(*parts, mkdir_parent=False):
    """Build path under active ComfyUI/models/latensync1.5 root."""
    return build_latentsync_path(LATENTSYNC_ROOT_DIR, *parts, mkdir_parent=mkdir_parent)


def get_cached_model(model_path, model_type, device):
    global _PILILINK_MODEL_CACHE
    cache_key = f"pililink_{model_type}_{model_path}"
    
    if cache_key in _PILILINK_MODEL_CACHE:
        # Check if the cached model is on the right device
        cached_model = _PILILINK_MODEL_CACHE[cache_key]
        model_device = next(cached_model.parameters()).device
        if str(model_device) == str(device):
            print(f"Using cached {model_type} model from Pililink cache")
            return cached_model
        else:
            print(f"Moving cached {model_type} model to {device}")
            cached_model = cached_model.to(device)
            return cached_model
    
    print(f"Loading {model_type} model from disk into Pililink cache")
    # Load the model
    model = torch.load(model_path, map_location=device)
    
    # Cache the model
    _PILILINK_MODEL_CACHE[cache_key] = model
    return model

def import_inference_script(script_path):
    """Import a Python file as a module using its file path."""
    if not os.path.exists(script_path):
        raise ImportError(f"Script not found: {script_path}")

    module_name = "pililink_latentsync_inference"  # Unique module name

    # Always reload to avoid stale module cache after code updates.
    if module_name in sys.modules:
        del sys.modules[module_name]
        print("Reloading Pililink inference module")
    
    print(f"Importing Pililink inference script from {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {script_path}")
    if spec.loader is None:
        raise ImportError(f"Module loader not available for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {str(e)}")

    return module


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


def normalize_path(path):
    """Normalize path to handle spaces and special characters"""
    return os.path.normpath(path).replace('\\', '/')

def get_ext_dir(subpath=None, mkdir=False):
    """Get extension directory path, optionally with a subpath"""
    # Get the directory containing this script
    dir = os.path.dirname(os.path.abspath(__file__))
    
    # Special case for temp directories
    if subpath and ("temp" in subpath.lower() or "tmp" in subpath.lower()):
        # Use our global temp directory instead
        global MODULE_TEMP_DIR
        sub_temp = os.path.join(MODULE_TEMP_DIR, subpath)
        if mkdir and not os.path.exists(sub_temp):
            os.makedirs(sub_temp, exist_ok=True)
        return sub_temp
    
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    return dir

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


def resolve_path_node_output_target(filename_prefix, output_path, result_mode, temp_dir, run_id):
    output_path_text = str(output_path or "").strip().strip('"').strip("'")
    resolved_mode = str(result_mode or "memory_only").lower()
    if resolved_mode not in {"memory_only", "both"}:
        raise ValueError(f"Pililink: unsupported result mode: {result_mode}")

    if output_path_text:
        final_output_path, output_filename = get_output_video_path(filename_prefix, output_path_text)
        return final_output_path, output_filename, True

    if resolved_mode == "both":
        final_output_path, output_filename = get_output_video_path(filename_prefix, "")
        return final_output_path, output_filename, True

    final_output_path = os.path.join(temp_dir, f"pililink_latentsync_{run_id}_result.mp4")
    return final_output_path, os.path.basename(final_output_path), False


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
        if target_duration < video_duration - duration_epsilon:
            resolved_video_path = trim_video_to_duration(
                video_path,
                build_temp_path("pililink_length_normal_video.mp4"),
                target_duration,
            )
        if needs_adjustment(audio_duration, target_duration):
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
        
        check_and_install_dependencies()
        setup_models()

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

    def _prepare_execution_settings(self, inference_steps, vram_usage):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_mixed_precision = False

        if torch.cuda.is_available():
            if vram_usage == "high":
                batch_size = min(32, 120 // inference_steps)
                use_mixed_precision = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True if hasattr(torch.backends.cuda, "matmul") else False
                torch.backends.cudnn.allow_tf32 = True
                torch.cuda.set_per_process_memory_fraction(0.95)
                print(f"Using Pililink high VRAM settings with {batch_size} batch size")
            elif vram_usage == "medium":
                batch_size = min(16, 80 // inference_steps)
                use_mixed_precision = True
                torch.backends.cudnn.benchmark = True
                torch.cuda.set_per_process_memory_fraction(0.85)
                print(f"Using Pililink medium VRAM settings with {batch_size} batch size")
            else:
                batch_size = min(8, 40 // inference_steps)
                use_mixed_precision = False
                torch.cuda.set_per_process_memory_fraction(0.75)
                print(f"Using Pililink low VRAM settings with {batch_size} batch size")

            torch.cuda.empty_cache()
        else:
            batch_size = 4
            print("No GPU detected, using CPU with minimal batch size")

        return {
            "device": device,
            "batch_size": batch_size,
            "use_mixed_precision": use_mixed_precision,
        }

    def _create_run_temp_dir(self):
        global MODULE_TEMP_DIR
        run_id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_dir = os.path.join(MODULE_TEMP_DIR, f"pililink_run_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        return run_id, temp_dir

    def _build_inference_args(
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
        segment_overlap_clips=0,
        affine_detect_interval=1,
        skip_video_normalization=False,
    ):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
        config_path = os.path.join(cur_dir, "configs", "unet", "stage2.yaml")
        scheduler_config_path = os.path.join(cur_dir, "configs")
        ckpt_path = get_latentsync_path("latentsync_unet.pt")
        whisper_ckpt_path = get_latentsync_path("whisper", "tiny.pt")

        config = OmegaConf.load(config_path)

        mask_image_path = os.path.join(cur_dir, "latentsync", "utils", "mask.png")
        if not os.path.exists(mask_image_path):
            alt_mask_path = os.path.join(cur_dir, "utils", "mask.png")
            if os.path.exists(alt_mask_path):
                mask_image_path = alt_mask_path
            else:
                print("Warning: Could not find mask image at expected locations")

        if hasattr(config, "data") and hasattr(config.data, "mask_image_path"):
            config.data.mask_image_path = mask_image_path

        args = argparse.Namespace(
            unet_config_path=config_path,
            inference_ckpt_path=ckpt_path,
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output_video_path,
            seed=seed,
            inference_steps=inference_steps,
            guidance_scale=lips_expression,
            scheduler_config_path=scheduler_config_path,
            whisper_ckpt_path=whisper_ckpt_path,
            device=execution_settings["device"],
            batch_size=execution_settings["batch_size"],
            use_mixed_precision=execution_settings["use_mixed_precision"],
            temp_dir=temp_dir,
            segment_inferences=segment_inferences,
            mask_image_path=mask_image_path,
            latentsync_root=LATENTSYNC_ROOT_DIR,
            hf_cache_dir=HF_CACHE_DIR,
            audio_embeds_cache_dir=get_latentsync_path("runtime_cache", "audio_embeds"),
            vae_model_path=get_latentsync_path("vae", "sd-vae-ft-mse"),
            disable_deepcache=str(deepcache or "on").lower() == "off",
            deepcache_cache_interval=max(1, int(deepcache_cache_interval)),
            deepcache_branch_id=max(0, int(deepcache_branch_id)),
            scheduler_type=str(scheduler_type or "ddim").lower(),
            segment_overlap_clips=max(0, int(segment_overlap_clips)),
            affine_detect_interval=max(1, int(affine_detect_interval)),
            skip_video_normalization=bool(skip_video_normalization),
        )

        return cur_dir, inference_script_path, config, args

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
        segment_overlap_clips=0,
        affine_detect_interval=1,
        skip_video_normalization=False,
    ):
        cur_dir, inference_script_path, config, args = self._build_inference_args(
            video_path=video_path,
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
            segment_overlap_clips=segment_overlap_clips,
            affine_detect_interval=affine_detect_interval,
            skip_video_normalization=skip_video_normalization,
        )

        package_root = os.path.dirname(cur_dir)
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        if cur_dir not in sys.path:
            sys.path.insert(0, cur_dir)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_module = import_inference_script(inference_script_path)
        if hasattr(inference_module, "get_temp_dir"):
            setattr(inference_module, "get_temp_dir", lambda *args, **kwargs: temp_dir)

        inference_temp = os.path.join(temp_dir, "temp")
        os.makedirs(inference_temp, exist_ok=True)

        # ComfyUI often wraps node execution in torch.inference_mode().
        # The LatentSync stack and some third-party preprocessing ops are
        # more reliable with normal tensors, so we explicitly disable it here.
        with torch.inference_mode(False):
            inference_module.main(config, args)


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
                "deepcache": (["on", "off"], {"default": "on"}),
                "deepcache_cache_interval": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1}),
                "deepcache_branch_id": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1}),
                "scheduler_type": (["ddim", "dpm_solver"], {"default": "ddim"}),
                "segment_overlap_clips": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "affine_detect_interval": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                 },}

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio") 
    FUNCTION = "inference"

    def process_batch(self, batch, use_mixed_precision=False):
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            processed_batch = batch.float() / 255.0
            if len(processed_batch.shape) == 3:
                processed_batch = processed_batch.unsqueeze(0)
            if processed_batch.shape[0] == 3:
                processed_batch = processed_batch.permute(1, 2, 0)
            if processed_batch.shape[-1] == 4:
                processed_batch = processed_batch[..., :3]
            return processed_batch

    def inference(
        self,
        images,
        audio,
        seed,
        lips_expression=1.5,
        inference_steps=20,
        vram_usage="medium",
        segment_inferences=16,
        deepcache="on",
        deepcache_cache_interval=3,
        deepcache_branch_id=0,
        scheduler_type="ddim",
        segment_overlap_clips=0,
        affine_detect_interval=1,
    ):
        throw_if_processing_interrupted()
        # Add timing information
        start_time = time.time()
        
        # Use our module temp directory
        global MODULE_TEMP_DIR
        
        # Define timing checkpoint function
        def log_timing(step):
            elapsed = time.time() - start_time
            print(f"[Pililink {elapsed:.2f}s] {step}")
        
        log_timing("Starting Pililink inference")
        
        execution_settings = self._prepare_execution_settings(inference_steps, vram_usage)
        run_id, temp_dir = self._create_run_temp_dir()
        
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

            log_timing("Processing audio")
            throw_if_processing_interrupted()
            log_timing("Saving temporary files")
            throw_if_processing_interrupted()
            resampled_audio = save_audio_input(audio, audio_path)

            # Move frames to CPU for saving to video
            frames_cpu = frames.cpu()
            try:
                if torchvision_io is None:
                    raise RuntimeError("torchvision.io is not available")
                torchvision_io.write_video(temp_video_path, frames_cpu, fps=25, video_codec='h264')
            except (TypeError, RuntimeError, ValueError, AttributeError):
                if av_mod is None:
                    av_mod = __import__("av")
                container = av_mod.open(temp_video_path, mode='w')
                stream = container.add_stream('h264', rate=25)
                stream.width = frames_cpu.shape[2]
                stream.height = frames_cpu.shape[1]

                for frame in frames_cpu:
                    frame = av_mod.VideoFrame.from_ndarray(frame.numpy(), format='rgb24')
                    packet = stream.encode(frame)
                    container.mux(packet)

                packet = stream.encode(None)
                container.mux(packet)
                container.close()
            
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
                segment_overlap_clips=segment_overlap_clips,
                affine_detect_interval=affine_detect_interval,
                skip_video_normalization=True,
            )

            log_timing("Processing output")
            throw_if_processing_interrupted()
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

            # Ensure audio is on CPU before returning
            if torch.cuda.is_available():
                if isinstance(resampled_audio["waveform"], torch.Tensor) and resampled_audio["waveform"].device.type == 'cuda':
                    resampled_audio["waveform"] = resampled_audio["waveform"].cpu()
                if isinstance(processed_frames, torch.Tensor) and processed_frames.device.type == 'cuda':
                    processed_frames = processed_frames.cpu()

            total_time = time.time() - start_time
            print(f"Pililink total processing time: {total_time:.2f}s")
            
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
                "seed": ("INT", {"default": 1247}),
                "lips_expression": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                "inference_steps": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),
                "vram_usage": (["high", "medium", "low"], {"default": "medium"}),
                "segment_inferences": ("INT", {"default": 16, "min": 2, "max": 256, "step": 2}),
                "deepcache": (["on", "off"], {"default": "on"}),
                "deepcache_cache_interval": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1}),
                "deepcache_branch_id": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1}),
                "scheduler_type": (["ddim", "dpm_solver"], {"default": "ddim"}),
                "segment_overlap_clips": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "affine_detect_interval": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
                "silent_padding_sec": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1}),
                "auto_silent_padding": ("BOOLEAN", {"default": False}),
                "result_mode": (["both", "memory_only"], {"default": "both"}),
                "filename_prefix": ("STRING", {"default": "LatentSync/Pililink"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Leave empty to use audio input or the video's original audio"}),
                "output_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional absolute output .mp4 path"}),
            },
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
        lips_expression=1.5,
        inference_steps=20,
        vram_usage="medium",
        segment_inferences=16,
        deepcache="on",
        deepcache_cache_interval=3,
        deepcache_branch_id=0,
        scheduler_type="ddim",
        segment_overlap_clips=0,
        affine_detect_interval=1,
        mode="normal",
        silent_padding_sec=0.5,
        auto_silent_padding=False,
        result_mode="memory_only",
        filename_prefix="LatentSync/Pililink",
        audio=None,
        audio_path="",
        output_path="",
    ):
        throw_if_processing_interrupted()
        start_time = time.time()
        resolved_video_path = resolve_user_path(video_path, "video_path")
        execution_settings = self._prepare_execution_settings(inference_steps, vram_usage)
        run_id, temp_dir = self._create_run_temp_dir()

        temp_audio_path = os.path.join(temp_dir, f"pililink_latentsync_{run_id}_audio.wav")
        final_output_path, output_filename, preserve_output_file = resolve_path_node_output_target(
            filename_prefix=filename_prefix,
            output_path=output_path,
            result_mode=result_mode,
            temp_dir=temp_dir,
            run_id=run_id,
        )

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

            aligned_video_path, aligned_audio_path = match_path_node_lengths(
                resolved_video_path,
                resolved_audio_path,
                mode,
                silent_padding_sec,
                temp_dir,
                auto_silent_padding=auto_silent_padding,
            )

            print(f"Pililink: Processing source video path {resolved_video_path}")
            if preserve_output_file:
                print(f"Pililink: Saving output video to {final_output_path}")
            else:
                print(f"Pililink: Writing temporary result video to {final_output_path}")

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
                segment_overlap_clips=segment_overlap_clips,
                affine_detect_interval=affine_detect_interval,
                skip_video_normalization=is_probably_25fps_video(aligned_video_path),
            )

            total_time = time.time() - start_time
            print(f"Pililink path-node total processing time: {total_time:.2f}s")

            # Release GPU memory before building output
            self._maybe_cuda_empty_cache(execution_settings, force=True)

            # Load only audio from the output video (no IMAGE tensor to avoid OOM)
            output_audio = self._load_audio_only(final_output_path)
            returned_video_path = final_output_path if preserve_output_file else ""
            returned_filename = output_filename if preserve_output_file else ""
            return {
                "ui": {"text": [returned_video_path or "[memory_only]"]},
                "result": (
                    output_audio,
                    returned_video_path,
                    returned_filename,
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
            }
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

    def adjust(self, images, audio, mode, fps=25.0, silent_padding_sec=0.5):
        throw_if_processing_interrupted()  # Add interruption check at start
        
        waveform = audio["waveform"].squeeze(0)
        sample_rate = int(audio["sample_rate"])
        
        # Convert images tensor to list of frames
        if isinstance(images, torch.Tensor):
            original_frames = [images[i] for i in range(images.shape[0])]
        else:
            original_frames = images.copy()
        
        num_original_frames = len(original_frames)
        
        if num_original_frames == 0:
            raise ValueError("Pililink: Input images cannot be empty")
        
        # Pre-check memory capacity
        frame_shape = original_frames[0].shape if original_frames else images.shape[1:]
        
        if mode == "normal":
            throw_if_processing_interrupted()
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
            return (
                torch.stack(adjusted_frames),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

        elif mode == "pingpong":
            throw_if_processing_interrupted()
            video_duration = len(original_frames) / fps
            audio_duration = waveform.shape[1] / sample_rate
            
            if audio_duration <= video_duration:
                throw_if_processing_interrupted()
                required_samples = int(video_duration * sample_rate)
                silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
                adjusted_audio = torch.cat([waveform, silence], dim=1)

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
                
                # Create frame list by indexing (no memory duplication)
                adjusted_frames = [original_frames[i] for i in final_indices[:target_frames]]
                
                throw_if_processing_interrupted()
                return (
                    torch.stack(adjusted_frames),
                    {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

        elif mode == "loop_to_audio":
            throw_if_processing_interrupted()
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
            
            # Create frame list by indexing (no memory duplication)
            adjusted_frames = [original_frames[i] for i in final_indices[:target_frames]]
            
            throw_if_processing_interrupted()
            return (
                torch.stack(adjusted_frames),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )


class PililinkLatentSyncRefactorMixin(PililinkLatentSyncBase):
    def _get_refactor_runtime_options(self):
        options = getattr(self, "_pililink_refactor_runtime_options", None)
        if not isinstance(options, dict):
            return {
                "clip_batch_size": 1,
                "auto_oom_fallback": True,
                "quality_mode": "balanced",
                "scheduler_type": "ddim",
                "segment_overlap_clips": 0,
                "affine_detect_interval": 1,
            }
        return {
            "clip_batch_size": max(1, int(options.get("clip_batch_size", 1))),
            "auto_oom_fallback": bool(options.get("auto_oom_fallback", True)),
            "quality_mode": str(options.get("quality_mode", "balanced")).lower(),
            "scheduler_type": str(options.get("scheduler_type", "ddim")).lower(),
            "segment_overlap_clips": max(0, int(options.get("segment_overlap_clips", 0))),
            "affine_detect_interval": max(1, int(options.get("affine_detect_interval", 1))),
        }

    def _prepare_execution_settings(self, inference_steps, vram_usage):
        runtime_options = self._get_refactor_runtime_options()
        quality_mode = runtime_options["quality_mode"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        supports_fp16 = False
        if device.type == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Turing (7.x) and newer GPUs can run fp16 inference efficiently.
            supports_fp16 = int(capability[0]) >= 7
        vram_mode = str(vram_usage or "medium").lower()
        use_mixed_precision = (
            quality_mode != "quality_first"
            and supports_fp16
            and vram_mode in {"high", "medium"}
        )
        dtype = torch.float16 if use_mixed_precision else torch.float32

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = quality_mode != "quality_first"
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = quality_mode != "quality_first"
            torch.backends.cudnn.allow_tf32 = quality_mode != "quality_first"

        return {
            "device": device,
            "batch_size": 1,
            "use_mixed_precision": use_mixed_precision,
            "dtype": dtype,
            "aggressive_cuda_cache_clear": False,
        }

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
        segment_overlap_clips=0,
        affine_detect_interval=1,
        skip_video_normalization=False,
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
        skip_video_normalization = skip_video_normalization or self._is_probably_25fps_video(video_path)
        runtime_options = self._get_refactor_runtime_options()
        resolved_scheduler_type = str(
            scheduler_type or runtime_options.get("scheduler_type", "ddim")
        ).strip().lower()
        if resolved_scheduler_type not in {"ddim", "dpm_solver"}:
            resolved_scheduler_type = "ddim"
        resolved_segment_overlap_clips = max(
            0,
            int(
                segment_overlap_clips
                if segment_overlap_clips is not None
                else runtime_options.get("segment_overlap_clips", 0)
            ),
        )
        resolved_affine_detect_interval = max(
            1,
            int(
                affine_detect_interval
                if affine_detect_interval is not None
                else runtime_options.get("affine_detect_interval", 1)
            ),
        )

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
            skip_video_normalization=skip_video_normalization,
            clip_batch_size=runtime_options["clip_batch_size"],
            auto_oom_fallback=runtime_options["auto_oom_fallback"],
            quality_mode=runtime_options["quality_mode"],
            segment_overlap_clips=resolved_segment_overlap_clips,
            affine_detect_interval=resolved_affine_detect_interval,
        )


class PililinkLatentSyncRefactorNode(PililinkLatentSyncRefactorMixin, PililinkLatentSyncNode):
    @classmethod
    def INPUT_TYPES(cls):
        input_types = copy.deepcopy(PililinkLatentSyncNode.INPUT_TYPES())
        input_types["required"]["clip_batch_size"] = ("INT", {"default": 1, "min": 1, "max": 16, "step": 1})
        input_types["required"]["auto_oom_fallback"] = ("BOOLEAN", {"default": True})
        input_types["required"]["quality_mode"] = (["balanced", "quality_first"], {"default": "balanced"})
        return input_types

    def inference(
        self,
        images,
        audio,
        seed,
        lips_expression=1.5,
        inference_steps=20,
        vram_usage="medium",
        segment_inferences=16,
        deepcache="on",
        deepcache_cache_interval=3,
        deepcache_branch_id=0,
        scheduler_type="ddim",
        segment_overlap_clips=0,
        affine_detect_interval=1,
        clip_batch_size=1,
        auto_oom_fallback=True,
        quality_mode="balanced",
    ):
        self._pililink_refactor_runtime_options = {
            "clip_batch_size": clip_batch_size,
            "auto_oom_fallback": auto_oom_fallback,
            "quality_mode": quality_mode,
            "scheduler_type": scheduler_type,
            "segment_overlap_clips": segment_overlap_clips,
            "affine_detect_interval": affine_detect_interval,
        }
        try:
            return super().inference(
                images=images,
                audio=audio,
                seed=seed,
                lips_expression=lips_expression,
                inference_steps=inference_steps,
                vram_usage=vram_usage,
                segment_inferences=segment_inferences,
                deepcache=deepcache,
                deepcache_cache_interval=deepcache_cache_interval,
                deepcache_branch_id=deepcache_branch_id,
                scheduler_type=scheduler_type,
                segment_overlap_clips=segment_overlap_clips,
                affine_detect_interval=affine_detect_interval,
            )
        finally:
            self._pililink_refactor_runtime_options = None


class PililinkLatentSyncRefactorVideoPathNode(
    PililinkLatentSyncRefactorMixin,
    PililinkLatentSyncVideoPathNode,
):
    CATEGORY = f"{NODE_CATEGORY}/Legacy"
    @classmethod
    def INPUT_TYPES(cls):
        input_types = copy.deepcopy(PililinkLatentSyncVideoPathNode.INPUT_TYPES())
        input_types["required"]["clip_batch_size"] = ("INT", {"default": 1, "min": 1, "max": 16, "step": 1})
        input_types["required"]["auto_oom_fallback"] = ("BOOLEAN", {"default": True})
        input_types["required"]["quality_mode"] = (["balanced", "quality_first"], {"default": "balanced"})
        return input_types

    def inference_from_path(
        self,
        video_path,
        seed,
        lips_expression=1.5,
        inference_steps=20,
        vram_usage="medium",
        segment_inferences=16,
        deepcache="on",
        deepcache_cache_interval=3,
        deepcache_branch_id=0,
        scheduler_type="ddim",
        segment_overlap_clips=0,
        affine_detect_interval=1,
        mode="normal",
        silent_padding_sec=0.5,
        auto_silent_padding=False,
        result_mode="memory_only",
        filename_prefix="LatentSync/Pililink",
        audio=None,
        audio_path="",
        output_path="",
        clip_batch_size=1,
        auto_oom_fallback=True,
        quality_mode="balanced",
    ):
        self._pililink_refactor_runtime_options = {
            "clip_batch_size": clip_batch_size,
            "auto_oom_fallback": auto_oom_fallback,
            "quality_mode": quality_mode,
            "scheduler_type": scheduler_type,
            "segment_overlap_clips": segment_overlap_clips,
            "affine_detect_interval": affine_detect_interval,
        }
        try:
            return super().inference_from_path(
                video_path=video_path,
                seed=seed,
                lips_expression=lips_expression,
                inference_steps=inference_steps,
                vram_usage=vram_usage,
                segment_inferences=segment_inferences,
                deepcache=deepcache,
                deepcache_cache_interval=deepcache_cache_interval,
                deepcache_branch_id=deepcache_branch_id,
                scheduler_type=scheduler_type,
                segment_overlap_clips=segment_overlap_clips,
                affine_detect_interval=affine_detect_interval,
                mode=mode,
                silent_padding_sec=silent_padding_sec,
                auto_silent_padding=auto_silent_padding,
                result_mode=result_mode,
                filename_prefix=filename_prefix,
                audio=audio,
                audio_path=audio_path,
                output_path=output_path,
            )
        finally:
            self._pililink_refactor_runtime_options = None

class PililinkLatentSyncUnifiedVideoPathNode(PililinkLatentSyncRefactorVideoPathNode):
    CATEGORY = NODE_CATEGORY


# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    NODE_KEY_MAIN: PililinkLatentSyncNode,
    NODE_KEY_MAIN_PATH: PililinkLatentSyncUnifiedVideoPathNode,
    NODE_KEY_MAIN_REFACTOR: PililinkLatentSyncRefactorNode,
    NODE_KEY_MAIN_PATH_REFACTOR: PililinkLatentSyncRefactorVideoPathNode,
    NODE_KEY_ADJUSTER: PililinkVideoLengthAdjuster,
}

# Display Names for ComfyUI - Clear distinction from original
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_KEY_MAIN: NODE_DISPLAY_MAIN,
    NODE_KEY_MAIN_PATH: NODE_DISPLAY_MAIN_PATH,
    NODE_KEY_MAIN_REFACTOR: NODE_DISPLAY_MAIN_REFACTOR,
    NODE_KEY_MAIN_PATH_REFACTOR: NODE_DISPLAY_MAIN_PATH_REFACTOR,
    NODE_KEY_ADJUSTER: NODE_DISPLAY_ADJUSTER,
}
