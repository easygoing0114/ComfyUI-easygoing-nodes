import json
import logging
import os

import torch

import comfy.sd
import comfy.utils
import folder_paths
from comfy.cli_args import args

from comfy_api.latest import ComfyExtension, io

LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------------------

_SCALE_ARG = dict(default=1.0, min=0.0, max=2.0, step=0.01)
_RATIO_ARG = dict(default=1.0, min=0.0, max=1.0, step=0.01)


def longest_prefix_match(key: str, ratios: dict) -> float:
    """Return the value whose key is the longest prefix of `key`. Default 1.0."""
    best_value = 1.0
    best_len = 0
    for prefix, value in ratios.items():
        if key.startswith(prefix) and len(prefix) > best_len:
            best_value = value
            best_len = len(prefix)
    return best_value


def build_scale_inputs(layer_keys: list[str], node_input: str, node_type=io.Float, **arg) -> list:
    """Build one widget input per layer key, plus the leading model/clip/vae input."""
    scale_arg = arg or _SCALE_ARG
    return [node_type.Input(node_input)] + [
        io.Float.Input(key, **scale_arg) for key in layer_keys
    ]


def single_output(node_type, display_name: str | None = None) -> list:
    """One output socket of the given io type (Model / Clip / Vae, ...)."""
    return [node_type.Output(display_name=display_name)] if display_name else [node_type.Output()]


def restore_layer_keys(kwargs: dict, layer_keys: list[str]) -> dict:
    """ComfyUI may convert '.' to '_' in widget ids; map kwargs back onto layer_keys."""
    dotted_to_underscore = {key.replace(".", "_"): key for key in layer_keys}
    ratios = {}
    for k, v in kwargs.items():
        if k in dotted_to_underscore:
            ratios[dotted_to_underscore[k]] = v
        elif k in layer_keys:
            ratios[k] = v
    return ratios


def scale_model_by_prefix(model, ratios: dict, prefix: str = "diffusion_model."):
    """Clone `model` and multiply each weight by longest_prefix_match(key, ratios)."""
    m = model.clone()
    for key, patch in m.get_key_patches(prefix).items():
        key_inner = key[len(prefix):]
        scale = longest_prefix_match(key_inner, ratios)
        if scale != 1.0:
            # add_patches computes: output = weight * strength_model + patch * strength_patch.
            # Using patch == weight itself: weight * 1.0 + weight * (scale - 1.0) = weight * scale.
            m.add_patches({key: patch}, scale - 1.0, 1.0)
    return m


def scale_vae_sd(vae, ratios: dict, skip_dtypes: set = frozenset()):
    """Return a new VAE with each tensor multiplied by longest_prefix_match(key, ratios)."""
    sd = vae.get_sd()
    new_sd = {}
    for key, tensor in sd.items():
        if tensor.dtype in skip_dtypes:
            new_sd[key] = tensor
            continue
        scale = longest_prefix_match(key, ratios)
        new_sd[key] = tensor * scale if scale != 1.0 else tensor
    return comfy.sd.VAE(sd=new_sd)


def merge_vae_sd(vae1, vae2, ratios: dict, default_ratio: float = 0.5):
    """Blend two VAE state dicts key-by-key: (1-ratio)*vae1 + ratio*vae2."""
    sd1, sd2 = vae1.get_sd(), vae2.get_sd()
    new_sd = {}
    for key, tensor in sd1.items():
        if key not in sd2:
            new_sd[key] = tensor
            continue
        ratio = longest_prefix_match(key, ratios) if ratios else default_ratio
        new_sd[key] = tensor * (1.0 - ratio) + sd2[key] * ratio
    for key, tensor in sd2.items():
        new_sd.setdefault(key, tensor)
    return comfy.sd.VAE(sd=new_sd)


def numbered_keys(prefix: str, count: int, suffix: str = ".", start: int = 0) -> list[str]:
    """e.g. numbered_keys("layers.", 3) -> ["layers.0.", "layers.1.", "layers.2."]"""
    return [f"{prefix}{i}{suffix}" for i in range(start, start + count)]


# ------------------------------------------------------------------------------
# Node: Key Name Inspector
# ------------------------------------------------------------------------------

class KeyNameInspector(io.ComfyNode):
    """Debug node: lists internal weight key names for MODEL / CLIP / VAE inputs,
    passing every input through unchanged."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_KeyNameInspector",
            display_name="Key Name Inspector",
            category="advanced/debug",
            description="Inspect internal weight key names for MODEL, CLIP, and/or VAE inputs. "
                        "Passes all inputs through unchanged; outputs a combined STRING report "
                        "and also prints it to the console.",
            inputs=[
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
                io.String.Output(display_name="keys_report"),
            ],
        )

    @staticmethod
    def _keys_from_model(model) -> list[str]:
        try:
            return sorted(model.get_key_patches("diffusion_model.").keys())
        except Exception as e:
            LOGGER.warning("KeyNameInspector: failed to read MODEL keys: %s", e)
            return []

    @staticmethod
    def _keys_from_clip(clip) -> list[str]:
        try:
            return sorted(clip.get_sd().keys())
        except Exception as e:
            LOGGER.warning("KeyNameInspector: failed to read CLIP keys: %s", e)
            return []

    @staticmethod
    def _keys_from_vae(vae) -> list[str]:
        # VAE objects may expose get_sd() or only first_stage_model.state_dict().
        try:
            if hasattr(vae, "get_sd"):
                return sorted(vae.get_sd().keys())
        except Exception as e:
            LOGGER.warning("KeyNameInspector: VAE.get_sd() failed: %s", e)
        try:
            if hasattr(vae, "first_stage_model"):
                return sorted(vae.first_stage_model.state_dict().keys())
        except Exception as e:
            LOGGER.warning("KeyNameInspector: VAE.first_stage_model.state_dict() failed: %s", e)
        return []

    @staticmethod
    def _format_section(title: str, keys: list[str], max_console_lines: int = 80) -> str:
        print("=" * 80)
        print(f"[KeyNameInspector] {title}: total keys = {len(keys)}")
        for k in keys[:max_console_lines]:
            print(" ", k)
        if len(keys) > max_console_lines:
            print(f"  ... and {len(keys) - max_console_lines} more")
        print("=" * 80)

        lines = [f"### {title} (total keys: {len(keys)}) ###", *keys]
        return "\n".join(lines)

    @classmethod
    def execute(cls, model=None, clip=None, vae=None) -> io.NodeOutput:
        sections = []
        if model is not None:
            sections.append(cls._format_section("MODEL keys (diffusion_model.)", cls._keys_from_model(model)))
        if clip is not None:
            sections.append(cls._format_section("CLIP keys", cls._keys_from_clip(clip)))
        if vae is not None:
            sections.append(cls._format_section("VAE keys", cls._keys_from_vae(vae)))

        if sections:
            report = "\n\n".join(sections)
        else:
            report = "KeyNameInspector: no input connected (model / clip / vae all unconnected)"
            print(report)

        return io.NodeOutput(model, clip, vae, report)


# ------------------------------------------------------------------------------
# Model-specific layer key tables
#
# Each table lists the dot-separated prefixes used for longest-prefix-match
# scaling/merging of a given architecture's diffusion_model weights.
# ------------------------------------------------------------------------------

_SDXL_LAYER_KEYS = [
    "time_embed.", "label_emb.",
    *numbered_keys("input_blocks.", 9, suffix=""),
    *numbered_keys("middle_block.", 3, suffix=""),
    *numbered_keys("output_blocks.", 9, suffix=""),
    "out.",
]

_HIDREAM_LAYER_KEYS = [
    "x_embedder.", "t_embedder.", "caption_projection.",
    *numbered_keys("double_stream_blocks.", 13),
    *numbered_keys("single_stream_blocks.", 32),
]

_QWEN_IMAGE_LAYER_KEYS = [
    "pos_embeds.", "img_in.", "txt_norm.", "txt_in.", "time_text_embed.",
    *numbered_keys("transformer_blocks.", 60),
    "proj_out.",
]

_Z_IMAGE_LAYER_KEYS = [
    "cap_embedder.", "cap_pad_token",
    *numbered_keys("context_refiner.", 2),
    *numbered_keys("layers.", 30),
    *numbered_keys("noise_refiner.", 2),
    "final_layer.", "t_embedder.", "x_embedder.", "x_pad_token",
]

# Krea2 uses ComfyUI's internal (post-load) module names rather than the
# safetensors key names; see class docstring for the safetensors -> internal mapping.
_KREA2_LAYER_KEYS = [
    "first.", "txtmlp.", "tmlp.", "tproj.",
    *numbered_keys("txtfusion.refiner_blocks.", 2),
    *numbered_keys("txtfusion.layerwise_blocks.", 2),
    "txtfusion.projector.",
    *numbered_keys("blocks.", 28),
    "last.",
]

_FLUX2_KLEIN_LAYER_KEYS = [
    "img_in.", "time_in.", "txt_in.",
    *[key for i in range(5) for key in (
        f"double_blocks.{i}.",
        f"double_blocks.{i}.img_attn.",
        f"double_blocks.{i}.img_mlp.",
        f"double_blocks.{i}.txt_attn.",
        f"double_blocks.{i}.txt_mlp.",
    )],
    "double_stream_modulation_img.", "double_stream_modulation_txt.",
    *numbered_keys("single_blocks.", 20),
    "single_stream_modulation.", "final_layer.",
]

_ERNIE_IMAGE_LAYER_KEYS = [
    "x_embedder.", "text_proj.", "time_embedding.",
    *[key for i in range(36) for key in (
        f"layers.{i}.",
        f"layers.{i}.self_attention.",
        f"layers.{i}.self_attention.to_q.",
        f"layers.{i}.self_attention.to_k.",
        f"layers.{i}.self_attention.to_v.",
        f"layers.{i}.self_attention.to_out.",
        f"layers.{i}.self_attention.norm_q.",
        f"layers.{i}.self_attention.norm_k.",
        f"layers.{i}.mlp.",
        f"layers.{i}.mlp.gate_proj.",
        f"layers.{i}.mlp.up_proj.",
        f"layers.{i}.mlp.linear_fc2.",
        f"layers.{i}.adaLN_mlp_ln.",
        f"layers.{i}.adaLN_sa_ln.",
    )],
    "adaLN_modulation.", "final_norm.", "final_linear.",
]

_HIDREAM_O1_LAYER_KEYS = [
    "x_embedder.", "t_embedder1.", "final_layer2.", "lm_head.",
    *numbered_keys("language_model.layers.", 36),
    *numbered_keys("visual.blocks.", 27),
    "visual.merger.", "visual.deepstack_merger_list.", "visual.patch_embed.", "visual.pos_embed.",
]

_CLIP_SDXL_LAYER_KEYS = [
    "clip_l.embeddings",
    *numbered_keys("clip_l.encoder.layers.", 12, suffix=""),
    "clip_l.final_layer_norm",
    "clip_g.embeddings",
    *numbered_keys("clip_g.encoder.layers.", 32, suffix=""),
    "clip_g.final_layer_norm", "clip_g.text_projection",
]

_CLIP_QWEN_LAYER_KEYS = [
    "model.embed_tokens", "visual.patch_embed",
    *numbered_keys("visual.blocks.", 32, suffix=""),
    "visual.merger",
    *numbered_keys("model.layers.", 28, suffix=""),
    "model.norm", "lm_head",
]

_VAE_SDXL_LAYER_KEYS = [
    "quant_conv", "post_quant_conv",
    "encoder.conv_in",
    *numbered_keys("encoder.down.", 4),
    "encoder.mid.attn_1.", "encoder.mid.block_1.", "encoder.mid.block_2.",
    "encoder.norm_out", "encoder.conv_out",
    "decoder.conv_in",
    "decoder.mid.attn_1.", "decoder.mid.block_1.", "decoder.mid.block_2.",
    *numbered_keys("decoder.up.", 4),
    "decoder.norm_out", "decoder.conv_out",
]

_VAE_FLUX_LAYER_KEYS = [
    "encoder.conv_in", "encoder.conv_out", "encoder.norm_out",
    *numbered_keys("encoder.down.", 4),
    *[f"encoder.down.{i}.block.{j}." for i in range(4) for j in range(3)],
    *numbered_keys("encoder.down.", 3, suffix=".downsample."),
    "encoder.mid.", "encoder.mid.block_1.", "encoder.mid.block_2.", "encoder.mid.attn_1.",
    "decoder.conv_in", "decoder.conv_out", "decoder.norm_out",
    *numbered_keys("decoder.up.", 4),
    *[f"decoder.up.{i}.block.{j}." for i in range(4) for j in range(3)],
    *numbered_keys("decoder.up.", 3, suffix=".upsample.", start=1),
    "decoder.mid.", "decoder.mid.block_1.", "decoder.mid.block_2.", "decoder.mid.attn_1.",
]

_VAE_FLUX2_LAYER_KEYS = [
    "quant_conv", "post_quant_conv",
    "encoder.conv_in", "encoder.conv_out", "encoder.norm_out",
    *numbered_keys("encoder.down.", 4),
    "encoder.mid.attn_1.", "encoder.mid.block_1.", "encoder.mid.block_2.",
    "decoder.conv_in", "decoder.conv_out", "decoder.norm_out",
    "decoder.mid.attn_1.", "decoder.mid.block_1.", "decoder.mid.block_2.",
    *numbered_keys("decoder.up.", 4),
]
_VAE_FLUX2_SKIP_DTYPES = {torch.int32, torch.int64, torch.bool}

_VAE_QWEN_LAYER_KEYS = [
    "conv1", "conv2",
    "encoder.conv1",
    *numbered_keys("encoder.downsamples.", 11),
    "encoder.middle.0.", "encoder.middle.1.", "encoder.middle.2.",
    "encoder.head.",
    "decoder.conv1",
    "decoder.middle.0.", "decoder.middle.1.", "decoder.middle.2.",
    *numbered_keys("decoder.upsamples.", 15),
    "decoder.head.",
]

_VAE_WAN_VIDEO_LAYER_KEYS = [
    "conv1.", "conv2.",
    "encoder.conv1.", "encoder.head.", "encoder.middle.",
    "encoder.middle.0.", "encoder.middle.1.", "encoder.middle.2.",
    *numbered_keys("encoder.downsamples.", 11),
    "decoder.conv1.", "decoder.head.", "decoder.middle.",
    "decoder.middle.0.", "decoder.middle.1.", "decoder.middle.2.",
    *numbered_keys("decoder.upsamples.", 15),
]


# ------------------------------------------------------------------------------
# Node: Model Scale SDXL
# ------------------------------------------------------------------------------

class ModelScaleSDXL(io.ComfyNode):
    """Scale SDXL model layers. scale=1.0 keeps original, scale=0.0 zeroes out."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelScaleSDXL",
            display_name="Model Scale SDXL",
            category="advanced/model_merging/model_specific",
            inputs=build_scale_inputs(_SDXL_LAYER_KEYS, "model", io.Model),
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(scale_model_by_prefix(model, kwargs))


# ------------------------------------------------------------------------------
# Node: Model Merge HiDream
# ------------------------------------------------------------------------------

class ModelMergeHiDream(io.ComfyNode):
    """Merge node for HiDream series models (Full, Dev, Fast).
    Assumes double_stream_blocks 0-12 and single_stream_blocks 0-31."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelMergeHiDream",
            display_name="Model Merge HiDream",
            category="advanced/model_merging/model_specific",
            description="Merge node for HiDream series models (Full, Dev, Fast). "
                        "Assumes double_stream_blocks 0-12 and single_stream_blocks 0-31.",
            inputs=[
                io.Model.Input("model1"),
                io.Model.Input("model2"),
                *[io.Float.Input(key, **_RATIO_ARG) for key in _HIDREAM_LAYER_KEYS],
            ],
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model1, model2, **kwargs) -> io.NodeOutput:
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for key in kp:
            ratio = longest_prefix_match(key[len("diffusion_model."):], kwargs)
            m.add_patches({key: kp[key]}, 1.0 - ratio, ratio)
        return io.NodeOutput(m)


# ------------------------------------------------------------------------------
# Node: Model Scale HiDream
# ------------------------------------------------------------------------------

class ModelScaleHiDream(io.ComfyNode):
    """Scale HiDream series model layers. scale=1.0 keeps original, scale=0.0 zeroes out."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelScaleHiDream",
            display_name="Model Scale HiDream",
            category="advanced/model_merging/model_specific",
            description="Scale specific layers of HiDream series models (Full, Dev, Fast). "
                        "Assumes double_stream_blocks 0-12 and single_stream_blocks 0-31.",
            inputs=build_scale_inputs(_HIDREAM_LAYER_KEYS, "model", io.Model),
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(scale_model_by_prefix(model, kwargs))


# ------------------------------------------------------------------------------
# Node: Model Scale Qwen Image
# ------------------------------------------------------------------------------

class ModelScaleQwenImage(io.ComfyNode):
    """Scale Qwen Image model layers. scale=1.0 keeps original, scale=0.0 zeroes out."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelScaleQwenImage",
            display_name="Model Scale Qwen Image",
            category="advanced/model_merging/model_specific",
            inputs=build_scale_inputs(_QWEN_IMAGE_LAYER_KEYS, "model", io.Model),
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(scale_model_by_prefix(model, kwargs))


# ------------------------------------------------------------------------------
# Node: Model Merge Z-Image
# ------------------------------------------------------------------------------

class ModelMergeZImage(io.ComfyNode):
    """Merge node for Z-Image models. ratio=1.0 uses model2, ratio=0.0 uses model1."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelMergeZImage",
            display_name="Model Merge Z-Image",
            category="advanced/model_merging/model_specific",
            inputs=[
                io.Model.Input("model1"),
                io.Model.Input("model2"),
                *[io.Float.Input(key, **_RATIO_ARG) for key in _Z_IMAGE_LAYER_KEYS],
            ],
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model1, model2, **kwargs) -> io.NodeOutput:
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for key in kp:
            ratio = longest_prefix_match(key[len("diffusion_model."):], kwargs)
            m.add_patches({key: kp[key]}, 1.0 - ratio, ratio)
        return io.NodeOutput(m)


# ------------------------------------------------------------------------------
# Node: Model Scale Z-Image
# ------------------------------------------------------------------------------

class ModelScaleZImage(io.ComfyNode):
    """Scale Z-Image model layers. scale=1.0 keeps original, scale=0.0 zeroes out."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelScaleZImage",
            display_name="Model Scale Z-Image",
            category="advanced/model_merging/model_specific",
            inputs=build_scale_inputs(_Z_IMAGE_LAYER_KEYS, "model", io.Model),
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(scale_model_by_prefix(model, kwargs))


# ------------------------------------------------------------------------------
# Node: Model Scale Krea2
# ------------------------------------------------------------------------------

class ModelScaleKrea2(io.ComfyNode):
    """Scale Krea2 (Krea-2-Turbo) model layers.

    Prefixes match ComfyUI's post-load internal module names, not the
    safetensors key names. UNETLoader renames as follows:
      img_in.* -> first.*, txt_in.* -> txtmlp.*, time_embed.* -> tmlp.*,
      time_mod_proj.* -> tproj.*, text_fusion.* -> txtfusion.*,
      transformer_blocks.{0..27}.* -> blocks.{0..27}.*, final_layer.* -> last.*
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelScaleKrea2",
            display_name="Model Scale Krea2",
            category="advanced/model_merging/model_specific",
            inputs=build_scale_inputs(_KREA2_LAYER_KEYS, "model", io.Model),
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(scale_model_by_prefix(model, kwargs))


# ------------------------------------------------------------------------------
# Node: Model Scale Flux2 Klein
# ------------------------------------------------------------------------------

class ModelScaleFlux2Klein(io.ComfyNode):
    """Scale FLUX2 Klein model layers. scale=1.0 keeps original, scale=0.0 zeroes out."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelScaleFlux2Klein",
            display_name="Model Scale Flux2 Klein",
            category="advanced/model_merging/model_specific",
            inputs=build_scale_inputs(_FLUX2_KLEIN_LAYER_KEYS, "model", io.Model),
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(scale_model_by_prefix(model, kwargs))


# ------------------------------------------------------------------------------
# Node: Model Scale ERNIE Image
# ------------------------------------------------------------------------------

class ModelScaleErnieImage(io.ComfyNode):
    """Scale ERNIE Image model layers (ernie-image.safetensors, layers.0-35)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelScaleErnieImage",
            display_name="Model Scale ERNIE Image",
            category="advanced/model_merging/model_specific",
            inputs=build_scale_inputs(_ERNIE_IMAGE_LAYER_KEYS, "model", io.Model),
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(scale_model_by_prefix(model, kwargs))


# ------------------------------------------------------------------------------
# Node: Model Scale HiDream-O1-Image
# ------------------------------------------------------------------------------

class ModelScaleHiDreamO1Image(io.ComfyNode):
    """Scale HiDream-O1-Image (UiT architecture) layers.
    Uses ModelPatcher clone/patch — no deepcopy, no in-place mutation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ModelScaleHiDreamO1Image",
            display_name="Model Scale HiDream-O1-Image",
            category="HiDream O1",
            description="Scale specific layers of HiDream-O1-Image (UiT architecture). "
                        "scale=1.0: unchanged | scale=0.0: zero out | scale>1.0: amplify",
            inputs=build_scale_inputs(_HIDREAM_O1_LAYER_KEYS, "model", io.Model),
            outputs=single_output(io.Model),
        )

    @classmethod
    def execute(cls, model, **kwargs) -> io.NodeOutput:
        ratios = {k: v for k, v in kwargs.items() if v != 1.0}
        if not ratios:
            LOGGER.info("ModelScaleHiDreamO1Image: all scales are 1.0, skipping.")
            return io.NodeOutput(model)

        patcher = getattr(model, "patcher", model)
        m = patcher.clone()

        try:
            kp_all = m.get_key_patches()
        except TypeError:
            kp_all = m.get_key_patches("")
        if not kp_all:
            LOGGER.warning("ModelScaleHiDreamO1Image: get_key_patches() returned empty dict.")
            return io.NodeOutput(model)

        # Auto-detect the active key prefix from the first matching candidate.
        detected_prefix = ""
        for candidate in ("diffusion_model.", "model."):
            if any(k.startswith(candidate) for k in kp_all):
                detected_prefix = candidate
                break

        kp = {k: v for k, v in kp_all.items() if k.startswith(detected_prefix)} if detected_prefix else kp_all

        modified = 0
        for key, patch in kp.items():
            scale = longest_prefix_match(key[len(detected_prefix):], ratios)
            if scale == 1.0:
                continue
            m.add_patches({key: patch}, scale - 1.0, 1.0)
            modified += 1

        LOGGER.info("ModelScaleHiDreamO1Image: patched %d / %d keys (prefix=%r).",
                    modified, len(kp), detected_prefix)

        try:
            result = model.clone_with_patcher(m)
        except (AttributeError, TypeError):
            result = m
        return io.NodeOutput(result)


# ------------------------------------------------------------------------------
# Node: CLIP Scale Dual SDXL Block
# ------------------------------------------------------------------------------

class CLIPScaleDualSDXLBlock(io.ComfyNode):
    """Scale SDXL Dual CLIP (CLIP-L: 12 layers, CLIP-G: 32 layers)."""

    _SKIP_SUFFIXES = (".position_ids", ".logit_scale")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_CLIPScaleDualSDXLBlock",
            display_name="CLIP Scale Dual SDXL Block",
            category="advanced/model_merging/model_specific",
            description="Scale specific layers of SDXL Dual CLIP (CLIP-L: 12 layers, CLIP-G: 32 layers). "
                        "Scale=1.0 keeps original, Scale=0.0 zeroes out the layer.",
            inputs=build_scale_inputs(_CLIP_SDXL_LAYER_KEYS, "clip", io.Clip),
            outputs=single_output(io.Clip),
        )

    @classmethod
    def execute(cls, clip, **kwargs) -> io.NodeOutput:
        m = clip.clone()
        ratios = {k: v for k, v in kwargs.items() if v != 1.0}
        if not ratios:
            return io.NodeOutput(m)

        # SDXL's full state-dict keys are longer than the widget prefixes
        # (e.g. "clip_l.transformer.text_model...."), so normalize before matching.
        patches_by_scale: dict[float, dict] = {}
        for key, weight in m.get_sd().items():
            if key.endswith(cls._SKIP_SUFFIXES):
                continue
            normalized = key.replace(".transformer.text_model.", ".").replace(".text_model.", ".")
            scale = longest_prefix_match(normalized, ratios)
            if scale != 1.0:
                patches_by_scale.setdefault(scale, {})[key] = (weight,)

        for scale, patches in patches_by_scale.items():
            # add_patches: output = weight * strength_model + patch * strength_patch.
            # With patch == weight: weight * 1.0 + weight * (scale - 1.0) = weight * scale.
            m.add_patches(patches, scale - 1.0, 1.0)
        return io.NodeOutput(m)


# ------------------------------------------------------------------------------
# Node: CLIP Scale Qwen Block
# ------------------------------------------------------------------------------

class CLIPScaleQwenBlock(io.ComfyNode):
    """Scale Qwen-2.5-VL-7B CLIP layers."""

    _SKIP_SUFFIXES = (".position_ids", ".logit_scale")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_CLIPScaleQwenBlock",
            display_name="CLIP Scale Qwen Block",
            category="advanced/model_merging/model_specific",
            description="Scale specific layers of Qwen-2.5-VL-7B CLIP. "
                        "Scale=1.0 keeps original, Scale=0.0 zeroes out the layer.",
            inputs=build_scale_inputs(_CLIP_QWEN_LAYER_KEYS, "clip", io.Clip),
            outputs=single_output(io.Clip),
        )

    @classmethod
    def execute(cls, clip, **kwargs) -> io.NodeOutput:
        m = clip.clone()
        ratios = {k: v for k, v in kwargs.items() if v != 1.0}
        if not ratios:
            return io.NodeOutput(m)

        patches_by_scale: dict[float, dict] = {}
        for key, patch in clip.get_key_patches().items():
            if key.endswith(cls._SKIP_SUFFIXES):
                continue
            normalized = key[len("transformer."):] if key.startswith("transformer.") else key

            # Require an exact segment boundary match to avoid e.g. "layers.1" matching "layers.10".
            scale = 1.0
            matched_len = 0
            for prefix, value in ratios.items():
                if normalized.startswith(prefix) and len(prefix) > matched_len:
                    tail = normalized[len(prefix):]
                    if tail == "" or tail.startswith("."):
                        scale = value
                        matched_len = len(prefix)
            if scale != 1.0:
                patches_by_scale.setdefault(scale, {})[key] = patch

        for scale, patches in patches_by_scale.items():
            # Here strength_model=0, strength_patch=scale, since patch already equals weight.
            m.add_patches(patches, scale, 0.0)
        return io.NodeOutput(m)


# ------------------------------------------------------------------------------
# Node: CLIP Save Qwen
# ------------------------------------------------------------------------------

class CLIPSaveQwen(io.ComfyNode):
    """Save Qwen-2.5-VL-7B CLIP models, stripping the internal 'qwen25_7b.transformer.' prefix."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_CLIPSaveQwen",
            display_name="CLIP Save Qwen",
            category="advanced/model_merging/model_specific",
            description="Saves Qwen-2.5-VL-7B CLIP models by stripping the internal "
                        "'qwen25_7b.transformer.' prefix.",
            is_output_node=True,
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("filename_prefix", default="qwen_2.5_vl_merged"),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, clip, filename_prefix: str) -> io.NodeOutput:
        metadata = {}
        if not args.disable_metadata:
            metadata["format"] = "pt"
            metadata["prompt"] = json.dumps(cls.hidden.prompt) if cls.hidden.prompt is not None else ""
            if cls.hidden.extra_pnginfo is not None:
                for k, v in cls.hidden.extra_pnginfo.items():
                    metadata[k] = json.dumps(v)

        strip_prefix = "qwen25_7b.transformer."
        output_sd = {}
        for key, value in clip.get_sd().items():
            if key.startswith(strip_prefix):
                output_sd[key[len(strip_prefix):]] = value
            elif key.startswith("qwen25_7b."):
                output_sd[key.replace("qwen25_7b.", "")] = value
            else:
                output_sd[key] = value

        output_dir = folder_paths.get_output_directory()
        full_folder, filename, counter, _subfolder, _prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )
        output_path = os.path.join(full_folder, f"{filename}_{counter:05}_.safetensors")
        comfy.utils.save_torch_file(output_sd, output_path, metadata=metadata)

        return io.NodeOutput()


# ------------------------------------------------------------------------------
# Node: VAE Merge Simple
# ------------------------------------------------------------------------------

class VAEMergeSimple(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEMergeSimple",
            display_name="VAE Merge Simple",
            category="advanced/model_merging",
            inputs=[
                io.Vae.Input("vae1"),
                io.Vae.Input("vae2"),
                io.Float.Input("ratio", **_RATIO_ARG),
            ],
            outputs=single_output(io.Vae),
        )

    @classmethod
    def execute(cls, vae1, vae2, ratio: float) -> io.NodeOutput:
        return io.NodeOutput(merge_vae_sd(vae1, vae2, ratios={}, default_ratio=ratio))


# ------------------------------------------------------------------------------
# Node: VAE Merge Subtract
# ------------------------------------------------------------------------------

class VAEMergeSubtract(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEMergeSubtract",
            display_name="VAE Merge Subtract",
            category="advanced/model_merging",
            inputs=[
                io.Vae.Input("vae1"),
                io.Vae.Input("vae2"),
                io.Float.Input("multiplier", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=single_output(io.Vae),
        )

    @classmethod
    def execute(cls, vae1, vae2, multiplier: float) -> io.NodeOutput:
        sd1, sd2 = vae1.get_sd(), vae2.get_sd()
        merged_sd = {
            key: (tensor - multiplier * sd2[key]) if key in sd2 else tensor
            for key, tensor in sd1.items()
        }
        return io.NodeOutput(comfy.sd.VAE(sd=merged_sd))


# ------------------------------------------------------------------------------
# Node: VAE Merge Add
# ------------------------------------------------------------------------------

class VAEMergeAdd(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEMergeAdd",
            display_name="VAE Merge Add",
            category="advanced/model_merging",
            inputs=[
                io.Vae.Input("vae1"),
                io.Vae.Input("vae2"),
            ],
            outputs=single_output(io.Vae),
        )

    @classmethod
    def execute(cls, vae1, vae2) -> io.NodeOutput:
        sd1, sd2 = vae1.get_sd(), vae2.get_sd()
        merged_sd = {
            key: (tensor + sd2[key]) if key in sd2 else tensor
            for key, tensor in sd1.items()
        }
        return io.NodeOutput(comfy.sd.VAE(sd=merged_sd))


# ------------------------------------------------------------------------------
# Node: VAE Scale SDXL Block
# ------------------------------------------------------------------------------

class VAEScaleSDXLBlock(io.ComfyNode):
    """Scale SDXL VAE layers. scale=1.0 keeps original, scale=0.0 zeroes out.

    get_sd() returns ComfyUI's internal ldm-style keys:
      encoder.down_blocks.N.resnets.M.* -> encoder.down.N.block.M.*
      encoder.mid_block.resnets.{0,1}.* -> encoder.mid.block_{1,2}.*
      encoder.mid_block.attentions.0.*  -> encoder.mid.attn_1.*
      (decoder mirrors the same remapping)
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEScaleSDXLBlock",
            display_name="VAE Scale SDXL Block",
            category="advanced/model_merging/model_specific",
            description="Scale specific layers of SDXL VAE. "
                        "Scale=1.0 keeps original, Scale=0.0 zeroes out the layer.",
            inputs=build_scale_inputs(_VAE_SDXL_LAYER_KEYS, "vae", io.Vae),
            outputs=single_output(io.Vae),
        )

    @classmethod
    def execute(cls, vae, **kwargs) -> io.NodeOutput:
        ratios = restore_layer_keys(kwargs, _VAE_SDXL_LAYER_KEYS)
        return io.NodeOutput(scale_vae_sd(vae, ratios))


# ------------------------------------------------------------------------------
# Node: VAE Merge SDXL Block
# ------------------------------------------------------------------------------

class VAEMergeSDXLBlock(io.ComfyNode):
    """Block-wise merge for two SDXL VAEs. ratio=0.0 keeps vae1, ratio=1.0 uses vae2."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEMergeSDXLBlock",
            display_name="VAE Merge SDXL Block",
            category="advanced/model_merging/model_specific",
            description="Block-wise merging for SDXL VAE. Ratio=0.0 keeps vae1, Ratio=1.0 uses vae2.",
            inputs=[
                io.Vae.Input("vae1"),
                io.Vae.Input("vae2"),
                *[io.Float.Input(key, default=0.5, min=0.0, max=1.0, step=0.01)
                  for key in _VAE_SDXL_LAYER_KEYS],
            ],
            outputs=single_output(io.Vae),
        )

    @classmethod
    def execute(cls, vae1, vae2, **kwargs) -> io.NodeOutput:
        ratios = restore_layer_keys(kwargs, _VAE_SDXL_LAYER_KEYS)
        return io.NodeOutput(merge_vae_sd(vae1, vae2, ratios))


# ------------------------------------------------------------------------------
# Node: VAE Scale FLUX Block
# ------------------------------------------------------------------------------

class VAEScaleFluxBlock(io.ComfyNode):
    """Scale FLUX1 VAE layers. FLUX1's get_sd() already uses ldm-style keys directly."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEScaleFluxBlock",
            display_name="VAE Scale FLUX Block",
            category="advanced/model_merging/model_specific",
            description="Scale specific layers of FLUX1 VAE. "
                        "Scale=1.0 keeps original, Scale=0.0 zeroes out the layer.",
            inputs=build_scale_inputs(_VAE_FLUX_LAYER_KEYS, "vae", io.Vae),
            outputs=single_output(io.Vae),
        )

    @classmethod
    def execute(cls, vae, **kwargs) -> io.NodeOutput:
        ratios = restore_layer_keys(kwargs, _VAE_FLUX_LAYER_KEYS)
        return io.NodeOutput(scale_vae_sd(vae, ratios))


# ------------------------------------------------------------------------------
# Node: VAE Scale FLUX2 Block
# ------------------------------------------------------------------------------

class VAEScaleFlux2Block(io.ComfyNode):
    """Scale FLUX2 VAE layers. Integer/bool tensors (e.g. buffer counters) are left untouched."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEScaleFlux2Block",
            display_name="VAE Scale FLUX2 Block",
            category="advanced/model_merging/model_specific",
            description="Scale specific layers of FLUX2 VAE. "
                        "Scale=1.0 keeps original, Scale=0.0 zeroes out the layer.",
            inputs=build_scale_inputs(_VAE_FLUX2_LAYER_KEYS, "vae", io.Vae),
            outputs=single_output(io.Vae),
        )

    @classmethod
    def execute(cls, vae, **kwargs) -> io.NodeOutput:
        ratios = restore_layer_keys(kwargs, _VAE_FLUX2_LAYER_KEYS)
        return io.NodeOutput(scale_vae_sd(vae, ratios, skip_dtypes=_VAE_FLUX2_SKIP_DTYPES))


# ------------------------------------------------------------------------------
# Node: VAE Scale Qwen Block
# ------------------------------------------------------------------------------

class VAEScaleQwenBlock(io.ComfyNode):
    """Scale Qwen-Image VAE layers."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEScaleQwenBlock",
            display_name="VAE Scale Qwen Block",
            category="advanced/model_merging/model_specific",
            description="Scale specific layers of the Qwen-Image VAE. "
                        "Scale=1.0 keeps original, Scale=0.0 zeroes out the layer.",
            inputs=build_scale_inputs(_VAE_QWEN_LAYER_KEYS, "vae", io.Vae),
            outputs=single_output(io.Vae),
        )

    @classmethod
    def execute(cls, vae, **kwargs) -> io.NodeOutput:
        ratios = restore_layer_keys(kwargs, _VAE_QWEN_LAYER_KEYS)
        return io.NodeOutput(scale_vae_sd(vae, ratios))


# ------------------------------------------------------------------------------
# Node: VAE Scale Wan Video
# ------------------------------------------------------------------------------

class VAEScaleWanVideoBlock(io.ComfyNode):
    """Scale Wan2.1 VAE layers."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEScaleWanVideo",
            display_name="VAE Scale Wan Video",
            category="advanced/model_merging/model_specific",
            description="Scale specific layers of the Wan2.1 VAE. "
                        "Scale=1.0 keeps original weights, Scale=0.0 zeroes out the layer.",
            inputs=build_scale_inputs(_VAE_WAN_VIDEO_LAYER_KEYS, "vae", io.Vae),
            outputs=single_output(io.Vae),
        )

    @classmethod
    def execute(cls, vae, **kwargs) -> io.NodeOutput:
        ratios = restore_layer_keys(kwargs, _VAE_WAN_VIDEO_LAYER_KEYS)
        return io.NodeOutput(scale_vae_sd(vae, ratios))


# ------------------------------------------------------------------------------
# Extension registration
# ------------------------------------------------------------------------------

NODE_LIST = [
    KeyNameInspector,
    ModelScaleSDXL,
    ModelMergeHiDream,
    ModelScaleHiDream,
    ModelScaleQwenImage,
    ModelMergeZImage,
    ModelScaleZImage,
    ModelScaleKrea2,
    ModelScaleFlux2Klein,
    ModelScaleErnieImage,
    ModelScaleHiDreamO1Image,
    CLIPScaleDualSDXLBlock,
    CLIPScaleQwenBlock,
    CLIPSaveQwen,
    VAEMergeSimple,
    VAEMergeSubtract,
    VAEMergeAdd,
    VAEScaleSDXLBlock,
    VAEMergeSDXLBlock,
    VAEScaleFluxBlock,
    VAEScaleFlux2Block,
    VAEScaleQwenBlock,
    VAEScaleWanVideoBlock,
]


class MergeNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODE_LIST


async def comfy_entrypoint() -> MergeNodesExtension:
    return MergeNodesExtension()