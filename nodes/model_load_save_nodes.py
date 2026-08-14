import re
import os

import torch
import comfy.sd
import folder_paths
from safetensors.torch import load_file, save_file
from safetensors import safe_open

from comfy_api.latest import io

# ------------------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------------------

# Precision/format keywords stripped from checkpoint filenames.
_NAME_CLEAN_KEYWORDS = (
    "fp32", "fp16", "bf16", "fp8",
    "svd", "scaled", "mxfp8", "nvfp4",
    "int8", "convrot", "hq",
    "e4m3fn", "e4m3", "e5m2fn", "e5m2",
)
_NAME_CLEAN_PATTERN = r'(?:' + '|'.join(_NAME_CLEAN_KEYWORDS) + r')'


def clean_ckpt_name(ckpt_name: str) -> str:
    """Strip extension, precision/format keywords, and trailing '-'/'_' from a checkpoint filename."""
    name = re.sub(r'\.(safetensors|pt|pth)', '', ckpt_name, flags=re.IGNORECASE)
    name = re.sub(_NAME_CLEAN_PATTERN, '', name, flags=re.IGNORECASE)
    name = re.sub(r'[-_]+$', '', name)
    return name


def get_combined_model_list() -> list[str]:
    """Return the sorted union of filenames in 'checkpoints' and 'diffusion_models'."""
    ckpt_list = folder_paths.get_filename_list("checkpoints")
    unet_list = folder_paths.get_filename_list("diffusion_models")
    return sorted(set(ckpt_list) | set(unet_list))


def get_model_full_path(model_name: str) -> str:
    """Resolve model_name against 'checkpoints' then 'diffusion_models'."""
    if model_name in folder_paths.get_filename_list("checkpoints"):
        return folder_paths.get_full_path_or_raise("checkpoints", model_name)
    if model_name in folder_paths.get_filename_list("diffusion_models"):
        return folder_paths.get_full_path_or_raise("diffusion_models", model_name)
    raise FileNotFoundError(f"Model '{model_name}' not found in 'checkpoints' or 'diffusion_models' folders.")


# ------------------------------------------------------------------------------
# Node: Load Checkpoint (with Name)
# ------------------------------------------------------------------------------

class LoadCheckpointWithName(io.ComfyNode):

    NODE_ID_LEGACY = "LoadCheckpointWithName"
    NODE_ID_INPUT_ORDER = ("model_name",)

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_LoadCheckpointWithNameClean",
            display_name="Load Checkpoint (with Name Clean)",
            category="model/loaders",
            description="Loads a diffusion model checkpoint (from 'checkpoints' or 'diffusion_models' "
                         "folders) and also outputs a cleaned name string.",
            search_aliases=["load model", "checkpoint", "model loader", "load checkpoint", "ckpt", "model"],
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=get_combined_model_list(),
                    tooltip="The name of the checkpoint (model) to load. "
                            "Searches both 'checkpoints' and 'diffusion_models' folders.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL", tooltip="The model used for denoising latents."),
                io.Clip.Output(display_name="CLIP", tooltip="The CLIP model used for encoding text prompts."),
                io.Vae.Output(
                    display_name="VAE",
                    tooltip="The VAE model used for encoding and decoding images to and from latent space.",
                ),
                io.String.Output(
                    display_name="model_name_clean",
                    tooltip="Cleaned model name (extension and precision/format keywords removed).",
                ),
            ],
        )

    @classmethod
    def execute(cls, model_name: str) -> io.NodeOutput:
        ckpt_path = get_model_full_path(model_name)
        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return io.NodeOutput(model, clip, vae, clean_ckpt_name(model_name))


# ------------------------------------------------------------------------------
# Node: Load Diffusion Model (with Name)
# ------------------------------------------------------------------------------

_WEIGHT_DTYPE_MAP = {
    "fp8_e4m3fn": {"dtype": torch.float8_e4m3fn},
    "fp8_e4m3fn_fast": {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True},
    "fp8_e5m2": {"dtype": torch.float8_e5m2},
}


class LoadDiffusionModelWithName(io.ComfyNode):

    NODE_ID_LEGACY = "LoadDiffusionModelWithName"
    NODE_ID_INPUT_ORDER = ("model_name", "weight_dtype")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_LoadDiffusionModelWithNameClean",
            display_name="Load Diffusion Model (with Name Clean)",
            category="model/loaders",
            description="Loads a diffusion model (UNET) from 'checkpoints' or 'diffusion_models' "
                         "folders and also outputs a cleaned name string.",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=get_combined_model_list(),
                    tooltip="The name of the diffusion model (UNET) to load. "
                            "Searches both 'checkpoints' and 'diffusion_models' folders.",
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
                io.String.Output(display_name="model_name_clean"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, weight_dtype: str) -> io.NodeOutput:
        model_options = _WEIGHT_DTYPE_MAP.get(weight_dtype, {})
        unet_path = get_model_full_path(model_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return io.NodeOutput(model, clean_ckpt_name(model_name))


# ------------------------------------------------------------------------------
# Load Original Model / Model Save with Original
#
# Solves two problems that arise when editing and re-saving a ComfyUI model:
#
#   1. Key prefix mismatch — ComfyUI may prepend an internal prefix (e.g.
#      "diffusion_model.") to state_dict keys, so they no longer match the
#      original safetensors file's key names. Resolved by brute-forcing
#      dot-separated prefix candidates on both sides and picking the pair
#      with the most matching keys.
#
#   2. Missing tensors — ComfyUI's state_dict only contains tensors it
#      actually used, so some may be dropped on save. Resolved by keeping
#      the full original tensor set and filling in whatever is missing
#      from ComfyUI's state_dict at save time.
# ------------------------------------------------------------------------------

ORIGINAL_MODEL_TYPE = io.Custom("ORIGINAL_MODEL")


def _collect_prefix_candidates(keys: list[str]) -> list[str]:
    """Enumerate dot-separated prefix candidates from a key set (always includes "")."""
    candidates = {""}
    for key in keys:
        parts = key.split(".")
        for i in range(1, len(parts)):
            candidates.add(".".join(parts[:i]) + ".")
    return list(candidates)


def _strip_prefix_from_keys(keys: list[str], prefix: str) -> dict[str, str]:
    """Return {normalized_key: original_key}; keys not matching prefix are left as-is."""
    result = {}
    for k in keys:
        norm = k[len(prefix):] if (prefix and k.startswith(prefix)) else k
        result[norm] = k
    return result


def _find_best_prefix_pair(
    orig_keys: list[str],
    comfy_keys: list[str],
) -> tuple[str, str, dict[str, str], dict[str, str]]:
    """
    Brute-force both sides' prefix candidates and return the (orig_prefix, comfy_prefix)
    pair that maximizes the number of matching normalized keys.

    Returns:
        orig_prefix, comfy_prefix, orig_norm_map, comfy_norm_map
    """
    orig_candidates = _collect_prefix_candidates(orig_keys)
    comfy_candidates = _collect_prefix_candidates(comfy_keys)

    orig_cache: dict[str, dict[str, str]] = {}
    comfy_cache: dict[str, dict[str, str]] = {}

    def get_orig(pfx: str) -> dict[str, str]:
        return orig_cache.setdefault(pfx, _strip_prefix_from_keys(orig_keys, pfx))

    def get_comfy(pfx: str) -> dict[str, str]:
        return comfy_cache.setdefault(pfx, _strip_prefix_from_keys(comfy_keys, pfx))

    best_count = -1
    best_orig_pfx = ""
    best_comfy_pfx = ""

    for op in orig_candidates:
        on_set = set(get_orig(op).keys())
        for cp in comfy_candidates:
            count = len(on_set & set(get_comfy(cp).keys()))
            if count > best_count:
                best_count = count
                best_orig_pfx = op
                best_comfy_pfx = cp

    print(
        f"[ModelSaveWithOriginal] prefix resolution: "
        f"orig={repr(best_orig_pfx)}, comfy={repr(best_comfy_pfx)}, matched_keys={best_count}"
    )

    return best_orig_pfx, best_comfy_pfx, get_orig(best_orig_pfx), get_comfy(best_comfy_pfx)


# ------------------------------------------------------------------------------
# Node: Load Original Model
# ------------------------------------------------------------------------------

class LoadOriginalModel(io.ComfyNode):
    """Reads a safetensors file raw, exposing both the original tensor set
    (original_model) and a normally-loaded ComfyUI MODEL."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        unet_names = folder_paths.get_filename_list("diffusion_models")
        ckpt_names = folder_paths.get_filename_list("checkpoints")
        return io.Schema(
            node_id="EasygoingNodes_LoadOriginalModel",
            display_name="Load Original Model",
            category="loaders",
            inputs=[
                io.Combo.Input("unet_name", options=sorted(set(unet_names + ckpt_names))),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp32", "fp16", "bf16"],
                    default="default",
                ),
            ],
            outputs=[
                ORIGINAL_MODEL_TYPE.Output(display_name="original_model"),
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, unet_name: str, weight_dtype: str) -> io.NodeOutput:
        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if path is None:
            path = folder_paths.get_full_path("checkpoints", unet_name)
        if path is None:
            raise FileNotFoundError(
                f"Model file not found: {unet_name}\n"
                f"Place it in the diffusion_models or checkpoints folder."
            )

        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        load_dtype = dtype_map.get(weight_dtype)

        # Raw-read to preserve original keys and every tensor.
        raw_sd = load_file(path, device="cpu")
        if load_dtype is not None:
            raw_sd = {k: v.to(load_dtype) for k, v in raw_sd.items()}

        metadata = {}
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                metadata = dict(f.metadata()) if f.metadata() else {}
        except Exception:
            pass

        original_model = {"tensors": raw_sd, "metadata": metadata, "source_path": path}

        # Normal ComfyUI MODEL load (equivalent to UNETLoader).
        model_options = {}
        if weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        comfy_model = comfy.sd.load_diffusion_model(path, model_options=model_options)

        return io.NodeOutput(original_model, comfy_model)


# ------------------------------------------------------------------------------
# Node: Model Save with Original
# ------------------------------------------------------------------------------

class ModelSaveWithOriginal(io.ComfyNode):
    """Merges an edited ComfyUI MODEL with its original_model and saves as safetensors.

    Flow: match keys between the two state_dicts via prefix search, take the
    ComfyUI (edited) value for matched tensors, restore missing tensors from
    the original, and save under the original file's key names.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelSaveWithOriginal",
            display_name="Model Save with Original",
            category="model_merging",
            is_output_node=True,
            inputs=[
                ORIGINAL_MODEL_TYPE.Input("original_model"),
                io.Model.Input("model"),
                io.String.Input("filename_prefix", default="model_restored"),
                io.Boolean.Input("save_metadata", default=True, optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        original_model: dict,
        model,
        filename_prefix: str,
        save_metadata: bool = True,
    ) -> io.NodeOutput:
        orig_tensors: dict[str, torch.Tensor] = original_model["tensors"]
        orig_metadata: dict = original_model.get("metadata", {})

        try:
            comfy_sd: dict[str, torch.Tensor] = model.model.state_dict()
        except AttributeError:
            try:
                comfy_sd = model.state_dict()
            except AttributeError:
                raise RuntimeError(
                    "Could not retrieve state_dict from model. "
                    "Check that LoadOriginalModel is connected to ModelSaveWithOriginal."
                )

        orig_keys = list(orig_tensors.keys())
        comfy_keys = list(comfy_sd.keys())
        _, _, orig_norm_map, comfy_norm_map = _find_best_prefix_pair(orig_keys, comfy_keys)

        comfy_norm_tensors: dict[str, torch.Tensor] = {
            norm_k: comfy_sd[orig_k] for norm_k, orig_k in comfy_norm_map.items()
        }

        # Merge: matched tensors take the edited value, missing ones are restored.
        output_sd: dict[str, torch.Tensor] = {}
        matched = 0
        restored = 0
        for norm_key, orig_key in orig_norm_map.items():
            if norm_key in comfy_norm_tensors:
                output_sd[orig_key] = comfy_norm_tensors[norm_key].to(orig_tensors[orig_key].dtype)
                matched += 1
            else:
                output_sd[orig_key] = orig_tensors[orig_key]
                restored += 1

        comfy_only = set(comfy_norm_map.keys()) - set(orig_norm_map.keys())
        print(
            f"[ModelSaveWithOriginal] matched={matched}, restored={restored}, "
            f"comfy_only(skipped)={len(comfy_only)}"
        )
        if comfy_only:
            print(
                "[ModelSaveWithOriginal] comfy_only keys (skipped): "
                + ", ".join(sorted(comfy_only)[:10])
                + ("..." if len(comfy_only) > 10 else "")
            )

        # Resolve output path, avoiding collisions via a numeric suffix.
        output_dir = folder_paths.get_output_directory()
        base_dir = os.path.join(output_dir, os.path.dirname(filename_prefix))
        os.makedirs(base_dir, exist_ok=True)
        base_name = os.path.basename(filename_prefix)

        counter = 1
        while True:
            suffix = f"_{counter:04d}" if counter > 1 else ""
            filename = f"{base_name}{suffix}.safetensors"
            full_path = os.path.join(base_dir, filename)
            if not os.path.exists(full_path):
                break
            counter += 1

        save_meta: dict[str, str] | None = None
        if save_metadata:
            save_meta = {k: str(v) for k, v in orig_metadata.items()} if orig_metadata else {}
            save_meta["restored_by"] = "ModelSaveWithOriginal"
            save_meta["restored_tensors"] = str(restored)
            save_meta["matched_tensors"] = str(matched)

        save_file(output_sd, full_path, metadata=save_meta)

        print(f"[ModelSaveWithOriginal] saved: {full_path}")
        print(
            f"  total tensors: {len(output_sd)} "
            f"(orig: {len(orig_tensors)}, matched: {matched}, restored: {restored})"
        )

        return io.NodeOutput()


NODE_LIST = [
    LoadCheckpointWithName,
    LoadDiffusionModelWithName,
    LoadOriginalModel,
    ModelSaveWithOriginal,
]