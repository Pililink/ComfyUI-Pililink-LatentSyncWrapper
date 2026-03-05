import argparse
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
import time
import traceback
import uuid

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
NODE_DISPLAY_MAIN = "Pililink LatentSync 1.5"
NODE_DISPLAY_ADJUSTER = "Pililink LatentSync Length Adjuster"

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

class PililinkLatentSyncNode:
    def __init__(self):
        # Make sure our temp directory is the current one
        global MODULE_TEMP_DIR
        if not os.path.exists(MODULE_TEMP_DIR):
            os.makedirs(MODULE_TEMP_DIR, exist_ok=True)
        
        check_and_install_dependencies()
        setup_models()

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
        
        # Get GPU capabilities and memory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_mixed_precision = False
        
        # Set VRAM usage based on user preference
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_mem_gb = gpu_mem / (1024 ** 3)
            
            # Dynamic batch size and settings based on VRAM usage preference
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
            else:  # low
                batch_size = min(8, 40 // inference_steps)
                use_mixed_precision = False
                torch.cuda.set_per_process_memory_fraction(0.75)
                print(f"Using Pililink low VRAM settings with {batch_size} batch size")
                
            # Clear GPU cache before processing
            torch.cuda.empty_cache()
        else:
            # CPU fallback settings
            batch_size = 4
            print("No GPU detected, using CPU with minimal batch size")
        
        # Create a run-specific subdirectory in our temp directory
        run_id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_dir = os.path.join(MODULE_TEMP_DIR, f"pililink_run_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_video_path = None
        output_video_path = None
        audio_path = None

        try:
            global av_mod
            # Create temporary file paths in our system temp directory
            temp_video_path = os.path.join(temp_dir, f"pililink_temp_{run_id}.mp4")
            output_video_path = os.path.join(temp_dir, f"pililink_latentsync_{run_id}_out.mp4")
            audio_path = os.path.join(temp_dir, f"pililink_latentsync_{run_id}_audio.wav")
            
            # Get the extension directory
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            
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
            # Resample audio if needed
            if sample_rate != 16000:
                new_sample_rate = 16000
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=new_sample_rate
                )
                waveform_16k = resampler(waveform)
                waveform, sample_rate = waveform_16k, new_sample_rate

            # Package resampled audio
            resampled_audio = {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": sample_rate
            }
            
            log_timing("Saving temporary files")
            throw_if_processing_interrupted()
            # Move waveform to CPU for saving
            waveform_cpu = waveform.cpu()
            torchaudio.save(audio_path, waveform_cpu, sample_rate)

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
            del frames_cpu, waveform_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_timing("Setting up model paths")
            throw_if_processing_interrupted()
            # Define paths to required files and configs from unified root
            inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
            config_path = os.path.join(cur_dir, "configs", "unet", "stage2.yaml")
            scheduler_config_path = os.path.join(cur_dir, "configs")
            ckpt_path = get_latentsync_path("latentsync_unet.pt")
            whisper_ckpt_path = get_latentsync_path("whisper", "tiny.pt")

            # Create config and args
            config = OmegaConf.load(config_path)

            # Set the correct mask image path
            mask_image_path = os.path.join(cur_dir, "latentsync", "utils", "mask.png")
            # Make sure the mask image exists
            if not os.path.exists(mask_image_path):
                # Try to find it in the utils directory directly
                alt_mask_path = os.path.join(cur_dir, "utils", "mask.png")
                if os.path.exists(alt_mask_path):
                    mask_image_path = alt_mask_path
                else:
                    print(f"Warning: Could not find mask image at expected locations")

            # Set mask path in config
            if hasattr(config, "data") and hasattr(config.data, "mask_image_path"):
                config.data.mask_image_path = mask_image_path

            args = argparse.Namespace(
                unet_config_path=config_path,
                inference_ckpt_path=ckpt_path,
                video_path=temp_video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                inference_steps=inference_steps,
                guidance_scale=lips_expression,  # Using lips_expression for the guidance_scale
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path,
                device=device,
                batch_size=batch_size,
                use_mixed_precision=use_mixed_precision,
                temp_dir=temp_dir,
                segment_inferences=segment_inferences,
                mask_image_path=mask_image_path,
                latentsync_root=LATENTSYNC_ROOT_DIR,
                hf_cache_dir=HF_CACHE_DIR,
                vae_model_path=get_latentsync_path("vae", "sd-vae-ft-mse")
            )

            # Set PYTHONPATH to include our directories 
            package_root = os.path.dirname(cur_dir)
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
            if cur_dir not in sys.path:
                sys.path.insert(0, cur_dir)

            # Clean GPU cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_timing("Importing inference module")
            throw_if_processing_interrupted()
            # Import the inference module
            inference_module = import_inference_script(inference_script_path)
            
            # Monkey patch any temp directory functions in the inference module
            if hasattr(inference_module, 'get_temp_dir'):
                setattr(inference_module, 'get_temp_dir', lambda *args, **kwargs: temp_dir)
                
            # Create subdirectories that the inference module might expect
            inference_temp = os.path.join(temp_dir, "temp")
            os.makedirs(inference_temp, exist_ok=True)
            
            log_timing("Running Pililink inference")
            throw_if_processing_interrupted()
            # Run inference
            inference_module.main(config, args)

            log_timing("Processing output")
            throw_if_processing_interrupted()
            # Clean GPU cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        except Exception as e:
            print(f"Error during Pililink inference: {str(e)}")
            traceback.print_exc()
            raise

        finally:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Only remove temporary files if successful (keep for debugging if failed)
            try:
                # Clean up temporary files individually
                for path in [temp_video_path, output_video_path, audio_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
            except:
                pass  # Ignore cleanup errors

            # Final GPU cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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

    def adjust(self, images, audio, mode, fps=25.0, silent_padding_sec=0.5):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = int(audio["sample_rate"])
        original_frames = [images[i] for i in range(images.shape[0])] if isinstance(images, torch.Tensor) else images.copy()

        if mode == "normal":
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
            
            return (
                torch.stack(adjusted_frames),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

        elif mode == "pingpong":
            video_duration = len(original_frames) / fps
            audio_duration = waveform.shape[1] / sample_rate
            if audio_duration <= video_duration:
                required_samples = int(video_duration * sample_rate)
                silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
                adjusted_audio = torch.cat([waveform, silence], dim=1)

                return (
                    torch.stack(original_frames),
                    {"waveform": adjusted_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

            else:
                silence_samples = math.ceil(silent_padding_sec * sample_rate)
                silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
                padded_audio = torch.cat([waveform, silence], dim=1)
                total_duration = (waveform.shape[1] + silence_samples) / sample_rate
                target_frames = math.ceil(total_duration * fps)
                reversed_frames = original_frames[::-1][1:-1]  # Remove endpoints
                frames = original_frames + reversed_frames
                while len(frames) < target_frames:
                    frames += frames[:target_frames - len(frames)]
                return (
                    torch.stack(frames[:target_frames]),
                    {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

        elif mode == "loop_to_audio":
            # Add silent padding then simple loop
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            total_duration = (waveform.shape[1] + silence_samples) / sample_rate
            target_frames = math.ceil(total_duration * fps)

            frames = original_frames.copy()
            while len(frames) < target_frames:
                frames += original_frames[:target_frames - len(frames)]
            
            return (
                torch.stack(frames[:target_frames]),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    NODE_KEY_MAIN: PililinkLatentSyncNode,
    NODE_KEY_ADJUSTER: PililinkVideoLengthAdjuster,
}

# Display Names for ComfyUI - Clear distinction from original
NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_KEY_MAIN: NODE_DISPLAY_MAIN,
    NODE_KEY_ADJUSTER: NODE_DISPLAY_ADJUSTER,
}
