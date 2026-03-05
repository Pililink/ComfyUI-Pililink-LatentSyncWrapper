import os

folder_paths = __import__("folder_paths")

LATENTSYNC_MODEL_TYPE = "latensync1.5"
LATENTSYNC_ALT_MODEL_TYPE = "latentsync1.5"
LATENTSYNC_LEGACY_DIRNAME = "LatentSync-1.5"


def safe_get_models_dir():
    try:
        return folder_paths.models_dir
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def register_latentsync_model_paths(models_root, env_root):
    new_root = os.path.join(models_root, LATENTSYNC_MODEL_TYPE)
    alt_new_root = os.path.join(models_root, LATENTSYNC_ALT_MODEL_TYPE)
    legacy_root = os.path.join(models_root, LATENTSYNC_LEGACY_DIRNAME)

    ordered_paths = []
    if env_root:
        ordered_paths.append(env_root)
    ordered_paths.extend([new_root, alt_new_root, legacy_root])

    seen = set()
    for path in ordered_paths:
        if path in seen:
            continue
        seen.add(path)
        os.makedirs(path, exist_ok=True)
        is_default = (path == new_root) if not env_root else (path == env_root)
        folder_paths.add_model_folder_path(LATENTSYNC_MODEL_TYPE, path, is_default=is_default)
        folder_paths.add_model_folder_path(LATENTSYNC_ALT_MODEL_TYPE, path, is_default=is_default)

    return new_root, legacy_root, alt_new_root


def has_required_latentsync_files(root):
    if not root:
        return False
    return os.path.exists(os.path.join(root, "latentsync_unet.pt")) and os.path.exists(
        os.path.join(root, "whisper", "tiny.pt")
    )


def resolve_active_latentsync_root(env_root, new_root, legacy_root, alt_new_root):
    if env_root:
        return env_root
    if has_required_latentsync_files(new_root):
        return new_root
    if has_required_latentsync_files(alt_new_root):
        return alt_new_root
    if has_required_latentsync_files(legacy_root):
        print("[Geeky LatentSync] 发现旧目录模型，保持兼容使用 legacy 路径。")
        return legacy_root
    return new_root


def configure_model_cache_env(latentsync_root_dir):
    hf_cache_dir = os.path.join(latentsync_root_dir, "hf_cache")
    torch_cache_dir = os.path.join(latentsync_root_dir, "torch_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(torch_cache_dir, exist_ok=True)

    os.environ["LATENTSYNC_ROOT_DIR"] = latentsync_root_dir
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
    os.environ["TORCH_HOME"] = torch_cache_dir
    return hf_cache_dir, torch_cache_dir


def initialize_latentsync_paths():
    models_root = safe_get_models_dir()
    env_root = os.environ.get("LATENTSYNC_ROOT_DIR")
    new_root, legacy_root, alt_new_root = register_latentsync_model_paths(models_root, env_root)
    active_root = resolve_active_latentsync_root(env_root, new_root, legacy_root, alt_new_root)
    hf_cache_dir, torch_cache_dir = configure_model_cache_env(active_root)
    return {
        "root": active_root,
        "hf_cache_dir": hf_cache_dir,
        "torch_cache_dir": torch_cache_dir,
    }


def build_latentsync_path(latentsync_root_dir, *parts, mkdir_parent=False):
    path = os.path.join(latentsync_root_dir, *parts)
    if mkdir_parent:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
