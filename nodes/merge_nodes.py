"""
merge_nodes.py - block-wise scale / merge nodes for ComfyUI (V3 schema).

Semantics (identical for every generated node)
----------------------------------------------
Scale : weight * value
        1.0 = unchanged, 0.0 = zeroed, >1.0 = amplified
Merge : model1 * (1 - value) + model2 * value
        0.0 = model1, 1.0 = model2, in-between = linear blend
        Weights that no widget covers are kept from model1.

Layer widgets are matched by the longest *dot-separated prefix* of a weight key,
so "layers.1" never captures "layers.10", and a more specific widget
("double_blocks.0.img_attn") overrides its parent ("double_blocks.0").

Layout of this file
-------------------
1. Helpers          prefix matching, patch / state-dict operations
2. Layer tables     one tuple of layer prefixes per architecture
3. Registry         Arch specs for MODEL / CLIP / VAE
4. Node factory     builds a "Scale" and a "Merge" node for every registered Arch
5. Other nodes      Key Name Inspector, CLIP Save Qwen, VAE arithmetic
6. Extension entry

To support a new architecture: add a layer table (use Key Name Inspector to see the
keys) and one `Arch(...)` line in the registry. Both nodes are generated for it.
After running a node, check the ComfyUI console: it logs how many keys were changed
and which keys were not covered by any widget.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

import comfy.sd
import comfy.utils
import folder_paths
from comfy.cli_args import args

from comfy_api.latest import ComfyExtension, io

LOGGER = logging.getLogger("EasygoingNodes.merge")

CATEGORY = "advanced/model_merging/model_specific"
MODEL_PREFIX = "diffusion_model."

_SCALE_ARG = dict(default=1.0, min=0.0, max=2.0, step=0.01)
_MERGE_ARG = dict(default=0.0, min=0.0, max=1.0, step=0.01)   # 0.0 = first input (model1 / clip1 / vae1)


# ==============================================================================
# 1. Helpers
# ==============================================================================

def numbered(prefix: str, count: int, *, start: int = 0, sub: str = "") -> list[str]:
    """numbered("layers", 2) -> ["layers.0", "layers.1"];  sub="mlp" appends ".mlp"."""
    tail = f".{sub}" if sub else ""
    return [f"{prefix}.{i}{tail}" for i in range(start, start + count)]


def numbered_with_parts(prefix: str, count: int, parts: tuple[str, ...]) -> list[str]:
    """Each numbered block followed by its named sub-modules (for finer-grained widgets)."""
    keys = []
    for i in range(count):
        base = f"{prefix}.{i}"
        keys.append(base)
        keys.extend(f"{base}.{part}" for part in parts)
    return keys


def longest_prefix_match(name: str, ratios: dict[str, float]) -> Optional[float]:
    """Value of the longest dot-segment prefix of `name` found in `ratios`, else None."""
    parts = name.split(".")
    for n in range(len(parts), 0, -1):
        value = ratios.get(".".join(parts[:n]))
        if value is not None:
            return value
    return None


def collect_ratios(kwargs: dict, layer_keys: tuple[str, ...]) -> dict[str, float]:
    """Map widget values back onto layer keys.
    ComfyUI may turn '.' into '_' in widget ids, so both spellings are accepted."""
    ids = {}
    for key in layer_keys:
        ids[key] = key
        ids[key.replace(".", "_")] = key
    return {ids[k]: float(v) for k, v in kwargs.items() if k in ids}


def _key_patches(patcher, prefix: str = "") -> dict:
    """get_key_patches() for a ModelPatcher (optionally prefix-filtered) or a CLIP."""
    return patcher.get_key_patches(prefix) if prefix else patcher.get_key_patches()


def _layer_name(key: str, prefix: str, normalize: Optional[Callable[[str], Optional[str]]]):
    name = key[len(prefix):]
    return normalize(name) if normalize is not None else name


def _weight_shape(patch) -> Optional[tuple]:
    """Shape of the base weight inside a get_key_patches() entry ([(weight, convert_fn), ...])."""
    try:
        return tuple(patch[0][0].shape)
    except Exception:
        return None


def _report(label: str, total: int, changed: int, unmatched: list[str], mismatched: int = 0) -> None:
    if total == 0 or len(unmatched) == total:
        LOGGER.warning("%s: no key matched any layer widget (%d keys seen). "
                       "Wrong architecture or unexpected key names?", label, total)
        return
    LOGGER.info("%s: changed %d / %d keys.", label, changed, total)
    if unmatched:
        LOGGER.info("%s: %d keys are not covered by any widget, e.g. %s",
                    label, len(unmatched), unmatched[:5])
    if mismatched:
        LOGGER.warning("%s: %d keys skipped because the two models' shapes differ.",
                       label, mismatched)


# ---- ModelPatcher / CLIP --------------------------------------------------------

def scale_patcher(patcher, ratios: dict[str, float], *, prefix: str = "",
                  normalize=None, label: str = "Scale"):
    """Clone `patcher` (ModelPatcher or CLIP) with each weight multiplied by its ratio."""
    m = patcher.clone()
    if all(v == 1.0 for v in ratios.values()):
        return m

    total = changed = 0
    unmatched: list[str] = []
    for key, patch in _key_patches(m, prefix).items():
        name = _layer_name(key, prefix, normalize)
        if name is None:
            continue
        total += 1
        scale = longest_prefix_match(name, ratios)
        if scale is None:
            unmatched.append(name)
            continue
        if scale == 1.0:
            continue
        # add_patches computes: out = weight * strength_model + patch * strength_patch.
        # The patch is the weight itself, so this yields weight * scale.
        m.add_patches({key: patch}, scale - 1.0, 1.0)
        changed += 1
    _report(label, total, changed, unmatched)
    return m


def merge_patchers(patcher1, patcher2, ratios: dict[str, float], *, prefix: str = "",
                   normalize=None, label: str = "Merge"):
    """Clone `patcher1`, blending in `patcher2`: out = p1 * (1 - ratio) + p2 * ratio."""
    m = patcher1.clone()
    if not any(v != 0.0 for v in ratios.values()):
        return m

    kp1 = _key_patches(m, prefix)
    total = changed = mismatched = 0
    unmatched: list[str] = []
    for key, patch in _key_patches(patcher2, prefix).items():
        if key not in kp1:
            continue
        name = _layer_name(key, prefix, normalize)
        if name is None:
            continue
        total += 1
        ratio = longest_prefix_match(name, ratios)
        if ratio is None:
            unmatched.append(name)       # uncovered -> keep model1
            continue
        if ratio == 0.0:
            continue
        shape1, shape2 = _weight_shape(kp1[key]), _weight_shape(patch)
        if shape1 is not None and shape2 is not None and shape1 != shape2:
            mismatched += 1
            continue
        # add_patches(patches, strength_patch, strength_model):
        #   out = weight(model1) * strength_model + patch(model2) * strength_patch
        m.add_patches({key: patch}, ratio, 1.0 - ratio)
        changed += 1
    _report(label, total, changed, unmatched, mismatched)
    return m


# ---- VAE (state-dict based) -----------------------------------------------------

def _lerp(a, b, ratio: float):
    """a * (1 - ratio) + b * ratio, computed in float32 and cast back to a's dtype."""
    if ratio == 0.0:
        return None
    return (a.float() * (1.0 - ratio) + b.float() * ratio).to(a.dtype)


def _combine_vae(vae1, vae2, fn, label: str):
    """New VAE where fn(key, a, b) -> tensor | None (None keeps vae1's tensor).
    Non-float tensors (integer buffers etc.) and keys missing from vae2 are kept as is."""
    sd1, sd2 = vae1.get_sd(), vae2.get_sd()
    out = {}
    changed = mismatched = 0
    for key, a in sd1.items():
        b = sd2.get(key)
        if b is None or not a.is_floating_point():
            out[key] = a
            continue
        if a.shape != b.shape:
            mismatched += 1
            out[key] = a
            continue
        res = fn(key, a, b.to(a.device))
        if res is None:
            out[key] = a
        else:
            out[key] = res
            changed += 1
    LOGGER.info("%s: changed %d / %d keys.", label, changed, len(sd1))
    if mismatched:
        LOGGER.warning("%s: %d keys skipped because the two VAEs' shapes differ.", label, mismatched)
    return comfy.sd.VAE(sd=out)


def scale_vae(vae, ratios: dict[str, float], label: str = "VAE Scale"):
    out = {}
    changed = 0
    unmatched: list[str] = []
    for key, tensor in vae.get_sd().items():
        scale = longest_prefix_match(key, ratios)
        if scale is None:
            unmatched.append(key)
        if scale in (None, 1.0) or not tensor.is_floating_point():
            out[key] = tensor
        else:
            out[key] = tensor * scale
            changed += 1
    _report(label, len(out), changed, unmatched)
    return comfy.sd.VAE(sd=out)


def merge_vae(vae1, vae2, ratios: dict[str, float], label: str = "VAE Merge"):
    unmatched: list[str] = []

    def fn(key, a, b):
        ratio = longest_prefix_match(key, ratios)
        if ratio is None:
            unmatched.append(key)
            return None
        return _lerp(a, b, ratio)

    result = _combine_vae(vae1, vae2, fn, label)
    if unmatched:
        LOGGER.info("%s: %d keys are not covered by any widget (kept from vae1), e.g. %s",
                    label, len(unmatched), unmatched[:5])
    return result


# ---- CLIP key normalisation -----------------------------------------------------

_CLIP_SKIP_SUFFIXES = (".position_ids", ".logit_scale")


def normalize_sdxl_clip(name: str) -> Optional[str]:
    """'clip_l.transformer.text_model.encoder.layers.0.x' -> 'clip_l.encoder.layers.0.x'."""
    if name.endswith(_CLIP_SKIP_SUFFIXES):
        return None
    return re.sub(r"\.transformer\.(?:text_model\.)?|\.text_model\.", ".", name, count=1)


def normalize_qwen_clip(name: str) -> Optional[str]:
    """'qwen25_7b.transformer.model.layers.0.x' -> 'model.layers.0.x'."""
    if name.endswith(_CLIP_SKIP_SUFFIXES):
        return None
    return re.sub(r"^(?:[^.]+\.)?transformer\.", "", name)


# ==============================================================================
# 2. Layer tables
#
# Each entry is a dot-separated key prefix (no trailing dot needed). The longest
# matching prefix of a weight key decides which widget controls it.
# ==============================================================================

# ---- Diffusion models -----------------------------------------------------------

_SDXL = (
    "time_embed", "label_emb",
    *numbered("input_blocks", 9),
    *numbered("middle_block", 3),
    *numbered("output_blocks", 9),
    "out",
)

_HIDREAM_I1 = (
    "x_embedder", "t_embedder", "p_embedder", "caption_projection",
    *numbered("double_stream_blocks", 16),
    *numbered("single_stream_blocks", 32),
    "final_layer",
)

# Verified against the HiDream O1 Image FP16 key dump (740 keys, full coverage).
_HIDREAM_O1 = (
    "x_embedder", "t_embedder1", "final_layer2",
    "language_model.embed_tokens",
    *numbered("language_model.layers", 36),
    "language_model.norm",
    *numbered("visual.blocks", 27),
    "visual.merger", "visual.patch_embed", "visual.pos_embed",
)

_QWEN_IMAGE = (
    "pos_embeds", "img_in", "txt_norm", "txt_in", "time_text_embed",
    *numbered("transformer_blocks", 60),
    "norm_out", "proj_out",
)

# Distinct key layout from Qwen-Image: no pos_embeds / txt_norm, adds modulation / norm_out,
# 32 transformer_blocks.
_QWEN_IMAGE_2_1 = (
    "img_in", "modulation.1", "norm_out", "time_text_embed", "txt_in",
    *numbered("transformer_blocks", 32),
    "proj_out",
)

_Z_IMAGE = (
    "cap_embedder", "cap_pad_token",
    *numbered("context_refiner", 2),
    *numbered("layers", 30),
    *numbered("noise_refiner", 2),
    "final_layer", "t_embedder", "x_embedder", "x_pad_token",
)

# Krea2 uses ComfyUI's post-load module names, not the safetensors key names:
#   img_in -> first, txt_in -> txtmlp, time_embed -> tmlp, time_mod_proj -> tproj,
#   text_fusion -> txtfusion, transformer_blocks.N -> blocks.N, final_layer -> last
_KREA2 = (
    "first", "txtmlp", "tmlp", "tproj",
    *numbered("txtfusion.refiner_blocks", 2),
    *numbered("txtfusion.layerwise_blocks", 2),
    "txtfusion.projector",
    *numbered("blocks", 28),
    "last",
)

_FLUX2_KLEIN = (
    "img_in", "time_in", "txt_in",
    *numbered_with_parts("double_blocks", 5, ("img_attn", "img_mlp", "txt_attn", "txt_mlp")),
    "double_stream_modulation_img", "double_stream_modulation_txt",
    *numbered("single_blocks", 20),
    "single_stream_modulation", "final_layer",
)

_ERNIE_LAYER_PARTS = (
    "self_attention",
    "self_attention.to_q", "self_attention.to_k", "self_attention.to_v", "self_attention.to_out",
    "self_attention.norm_q", "self_attention.norm_k",
    "mlp", "mlp.gate_proj", "mlp.up_proj", "mlp.linear_fc2",
    "adaLN_mlp_ln", "adaLN_sa_ln",
)
_ERNIE_IMAGE = (
    "x_embedder", "text_proj", "time_embedding",
    *numbered_with_parts("layers", 36, _ERNIE_LAYER_PARTS),
    "adaLN_modulation", "final_norm", "final_linear",
)

# rope.bands (F32 constant buffer) is intentionally not covered.
_MOSS_IMAGE = (
    "video_embedder", "time_embedding", "text_embedder.proj",
    *numbered("text_embedder.blocks", 2),
    "shared_block_modulation",
    *numbered("layers", 42),
    "video_head_modulation", "final_norm_video", "final_linear_video",
)

# ---- CLIP / text encoders (names after normalize_*_clip) -------------------------

_CLIP_SDXL = (
    "clip_l.embeddings",
    *numbered("clip_l.encoder.layers", 12),
    "clip_l.final_layer_norm",
    "clip_g.embeddings",
    *numbered("clip_g.encoder.layers", 32),
    "clip_g.final_layer_norm", "clip_g.text_projection",
)

_CLIP_QWEN25_VL = (
    "model.embed_tokens", "visual.patch_embed",
    *numbered("visual.blocks", 32),
    "visual.merger",
    *numbered("model.layers", 28),
    "model.norm", "lm_head",
)

# ---- VAE (ComfyUI get_sd() keys) -------------------------------------------------

_VAE_SDXL = (
    "quant_conv", "post_quant_conv",
    "encoder.conv_in", *numbered("encoder.down", 4),
    "encoder.mid.attn_1", "encoder.mid.block_1", "encoder.mid.block_2",
    "encoder.norm_out", "encoder.conv_out",
    "decoder.conv_in",
    "decoder.mid.attn_1", "decoder.mid.block_1", "decoder.mid.block_2",
    *numbered("decoder.up", 4),
    "decoder.norm_out", "decoder.conv_out",
)

_VAE_FLUX = (
    "encoder.conv_in", "encoder.conv_out", "encoder.norm_out",
    *numbered("encoder.down", 4),
    *[f"encoder.down.{i}.block.{j}" for i in range(4) for j in range(2)],
    *numbered("encoder.down", 3, sub="downsample"),
    "encoder.mid.block_1", "encoder.mid.block_2", "encoder.mid.attn_1",
    "decoder.conv_in", "decoder.conv_out", "decoder.norm_out",
    *numbered("decoder.up", 4),
    *[f"decoder.up.{i}.block.{j}" for i in range(4) for j in range(3)],
    *numbered("decoder.up", 3, start=1, sub="upsample"),
    "decoder.mid.block_1", "decoder.mid.block_2", "decoder.mid.attn_1",
)

# Integer buffers (e.g. batch-norm counters) are skipped automatically.
_VAE_FLUX2 = (
    "quant_conv", "post_quant_conv",
    "encoder.conv_in", "encoder.conv_out", "encoder.norm_out",
    *numbered("encoder.down", 4),
    "encoder.mid.attn_1", "encoder.mid.block_1", "encoder.mid.block_2",
    "decoder.conv_in", "decoder.conv_out", "decoder.norm_out",
    "decoder.mid.attn_1", "decoder.mid.block_1", "decoder.mid.block_2",
    *numbered("decoder.up", 4),
)

# Wan2.1 and Qwen-Image share the same VAE architecture.
_VAE_WAN = (
    "conv1", "conv2",
    "encoder.conv1", *numbered("encoder.downsamples", 11),
    *numbered("encoder.middle", 3), "encoder.head",
    "decoder.conv1", *numbered("decoder.middle", 3),
    *numbered("decoder.upsamples", 15), "decoder.head",
)


# ==============================================================================
# 3. Registry
# ==============================================================================

@dataclass(frozen=True)
class Arch:
    slug: str                       # node_id suffix, e.g. "HiDreamO1Image"
    name: str                       # display name, e.g. "HiDream O1 Image"
    layer_keys: tuple[str, ...]
    note: str = ""                  # appended to the node description
    normalize: Optional[Callable[[str], Optional[str]]] = None   # CLIP key normaliser


MODEL_ARCHS = (
    Arch("SDXL", "SDXL", _SDXL),
    Arch("HiDream", "HiDream I1", _HIDREAM_I1, "Full / Dev / Fast."),
    Arch("HiDreamO1Image", "HiDream O1 Image", _HIDREAM_O1, "UiT architecture."),
    Arch("QwenImage", "Qwen Image", _QWEN_IMAGE),
    Arch("QwenImage2_1", "Qwen Image 2.1", _QWEN_IMAGE_2_1),
    Arch("ZImage", "Z-Image", _Z_IMAGE),
    Arch("Krea2", "Krea2", _KREA2, "Prefixes are ComfyUI's post-load module names."),
    Arch("Flux2Klein", "Flux2 Klein", _FLUX2_KLEIN),
    Arch("ErnieImage", "ERNIE Image", _ERNIE_IMAGE),
    Arch("MossImage", "MOSS Image", _MOSS_IMAGE),
)

CLIP_ARCHS = (
    Arch("DualSDXL", "Dual SDXL", _CLIP_SDXL, "CLIP-L (12 layers) + CLIP-G (32 layers).",
         normalize=normalize_sdxl_clip),
    Arch("Qwen25VL", "Qwen2.5-VL 7B", _CLIP_QWEN25_VL, normalize=normalize_qwen_clip),
)

VAE_ARCHS = (
    Arch("SDXL", "SDXL", _VAE_SDXL),
    Arch("Flux", "FLUX", _VAE_FLUX),
    Arch("Flux2", "FLUX2", _VAE_FLUX2),
    Arch("Wan21", "Wan2.1 / Qwen-Image", _VAE_WAN),
)


@dataclass(frozen=True)
class Component:
    """What differs between MODEL / CLIP / VAE nodes."""
    label: str          # node_id / display-name part: "Model", "CLIP", "VAE"
    io_type: type       # io.Model / io.Clip / io.Vae
    arg: str            # input name: "model", "clip", "vae"
    scale: Callable     # (obj, ratios, arch, label) -> obj
    merge: Callable     # (obj1, obj2, ratios, arch, label) -> obj


MODEL = Component(
    "Model", io.Model, "model",
    scale=lambda o, r, a, lbl: scale_patcher(o, r, prefix=MODEL_PREFIX, label=lbl),
    merge=lambda o1, o2, r, a, lbl: merge_patchers(o1, o2, r, prefix=MODEL_PREFIX, label=lbl),
)
CLIP = Component(
    "CLIP", io.Clip, "clip",
    scale=lambda o, r, a, lbl: scale_patcher(o, r, normalize=a.normalize, label=lbl),
    merge=lambda o1, o2, r, a, lbl: merge_patchers(o1, o2, r, normalize=a.normalize, label=lbl),
)
VAE = Component(
    "VAE", io.Vae, "vae",
    scale=lambda o, r, a, lbl: scale_vae(o, r, lbl),
    merge=lambda o1, o2, r, a, lbl: merge_vae(o1, o2, r, lbl),
)


# ==============================================================================
# 4. Node factory
# ==============================================================================

def _float_inputs(keys: tuple[str, ...], arg: dict) -> list:
    return [io.Float.Input(key, **arg) for key in keys]


def _make_scale_node(comp: Component, arch: Arch) -> type[io.ComfyNode]:
    label = f"{comp.label} Scale {arch.name}"

    class _ScaleNode(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id=f"EasygoingNodes_{comp.label}Scale{arch.slug}",
                display_name=label,
                category=CATEGORY,
                description=f"Scale specific layers of {arch.name}. "
                            "1.0 keeps the original, 0.0 zeroes the layer, >1.0 amplifies. "
                            f"{arch.note}".strip(),
                inputs=[comp.io_type.Input(comp.arg), *_float_inputs(arch.layer_keys, _SCALE_ARG)],
                outputs=[comp.io_type.Output()],
            )

        @classmethod
        def execute(cls, **kwargs) -> io.NodeOutput:
            obj = kwargs.pop(comp.arg)
            ratios = collect_ratios(kwargs, arch.layer_keys)
            return io.NodeOutput(comp.scale(obj, ratios, arch, label))

    _ScaleNode.__name__ = _ScaleNode.__qualname__ = f"{comp.label}Scale{arch.slug}"
    return _ScaleNode


def _make_merge_node(comp: Component, arch: Arch) -> type[io.ComfyNode]:
    label = f"{comp.label} Merge {arch.name}"

    class _MergeNode(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id=f"EasygoingNodes_{comp.label}Merge{arch.slug}",
                display_name=label,
                category=CATEGORY,
                description=f"Block-wise merge of two {arch.name}. "
                            "0.0 keeps the first, 1.0 uses the second, in-between values blend linearly. "
                            "Weights not covered by any widget are kept from the first. "
                            f"{arch.note}".strip(),
                inputs=[
                    comp.io_type.Input(f"{comp.arg}1"),
                    comp.io_type.Input(f"{comp.arg}2"),
                    *_float_inputs(arch.layer_keys, _MERGE_ARG),
                ],
                outputs=[comp.io_type.Output()],
            )

        @classmethod
        def execute(cls, **kwargs) -> io.NodeOutput:
            obj1 = kwargs.pop(f"{comp.arg}1")
            obj2 = kwargs.pop(f"{comp.arg}2")
            ratios = collect_ratios(kwargs, arch.layer_keys)
            return io.NodeOutput(comp.merge(obj1, obj2, ratios, arch, label))

    _MergeNode.__name__ = _MergeNode.__qualname__ = f"{comp.label}Merge{arch.slug}"
    return _MergeNode


def _build_block_nodes() -> list[type[io.ComfyNode]]:
    nodes = []
    for comp, archs in ((MODEL, MODEL_ARCHS), (CLIP, CLIP_ARCHS), (VAE, VAE_ARCHS)):
        for arch in archs:
            nodes.append(_make_scale_node(comp, arch))
            nodes.append(_make_merge_node(comp, arch))
    return nodes


# ==============================================================================
# 5. Other nodes
# ==============================================================================

class KeyNameInspector(io.ComfyNode):
    """Debug node: lists internal weight key names for MODEL / CLIP / VAE inputs and
    passes every input through unchanged. Use it to write a layer table for a new Arch."""

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
    def _model_keys(model) -> list[str]:
        return sorted(model.get_key_patches(MODEL_PREFIX))

    @staticmethod
    def _clip_keys(clip) -> list[str]:
        return sorted(clip.get_sd())

    @staticmethod
    def _vae_keys(vae) -> list[str]:
        # VAE objects may expose get_sd() or only first_stage_model.state_dict().
        if hasattr(vae, "get_sd"):
            return sorted(vae.get_sd())
        return sorted(vae.first_stage_model.state_dict())

    @staticmethod
    def _section(title: str, get_keys, obj, max_console_lines: int = 80) -> str:
        try:
            keys = get_keys(obj)
        except Exception as e:
            LOGGER.warning("KeyNameInspector: failed to read %s: %s", title, e)
            keys = []
        print("=" * 80)
        print(f"[KeyNameInspector] {title}: total keys = {len(keys)}")
        for k in keys[:max_console_lines]:
            print(" ", k)
        if len(keys) > max_console_lines:
            print(f"  ... and {len(keys) - max_console_lines} more")
        print("=" * 80)
        return "\n".join([f"### {title} (total keys: {len(keys)}) ###", *keys])

    @classmethod
    def execute(cls, model=None, clip=None, vae=None) -> io.NodeOutput:
        sections = []
        if model is not None:
            sections.append(cls._section("MODEL keys (diffusion_model.)", cls._model_keys, model))
        if clip is not None:
            sections.append(cls._section("CLIP keys", cls._clip_keys, clip))
        if vae is not None:
            sections.append(cls._section("VAE keys", cls._vae_keys, vae))

        if sections:
            report = "\n\n".join(sections)
        else:
            report = "KeyNameInspector: no input connected (model / clip / vae all unconnected)"
            print(report)
        return io.NodeOutput(model, clip, vae, report)


class CLIPSaveQwen(io.ComfyNode):
    """Save a Qwen-2.5-VL-7B CLIP, stripping the internal 'qwen25_7b.transformer.' prefix."""

    _STRIP_PREFIXES = ("qwen25_7b.transformer.", "qwen25_7b.")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_CLIPSaveQwen",
            display_name="CLIP Save Qwen",
            category=CATEGORY,
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
    def _strip(cls, key: str) -> str:
        for prefix in cls._STRIP_PREFIXES:
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    @classmethod
    def execute(cls, clip, filename_prefix: str) -> io.NodeOutput:
        metadata = {}
        if not args.disable_metadata:
            metadata["format"] = "pt"
            metadata["prompt"] = json.dumps(cls.hidden.prompt) if cls.hidden.prompt is not None else ""
            if cls.hidden.extra_pnginfo is not None:
                for k, v in cls.hidden.extra_pnginfo.items():
                    metadata[k] = json.dumps(v)

        output_sd = {cls._strip(key): value for key, value in clip.get_sd().items()}

        full_folder, filename, counter, _subfolder, _prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        output_path = os.path.join(full_folder, f"{filename}_{counter:05}_.safetensors")
        comfy.utils.save_torch_file(output_sd, output_path, metadata=metadata)
        return io.NodeOutput()


# ---- Whole-VAE arithmetic (architecture independent) -----------------------------

class VAEMergeSimple(io.ComfyNode):
    """Uniform blend of two VAEs: vae1 * (1 - ratio) + vae2 * ratio."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_VAEMergeSimple",
            display_name="VAE Merge Simple",
            category="advanced/model_merging",
            inputs=[
                io.Vae.Input("vae1"),
                io.Vae.Input("vae2"),
                io.Float.Input("ratio", **_MERGE_ARG),
            ],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae1, vae2, ratio: float) -> io.NodeOutput:
        return io.NodeOutput(_combine_vae(
            vae1, vae2, lambda key, a, b: _lerp(a, b, ratio), "VAE Merge Simple"))


def _make_vae_arith_node(action: str, sign: float) -> type[io.ComfyNode]:
    """VAE Merge Add: vae1 + m * vae2.   VAE Merge Subtract: vae1 - m * vae2."""
    label = f"VAE Merge {action}"

    class _ArithNode(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id=f"EasygoingNodes_VAEMerge{action}",
                display_name=label,
                category="advanced/model_merging",
                description=f"vae1 {'+' if sign > 0 else '-'} multiplier * vae2, for every float tensor.",
                inputs=[
                    io.Vae.Input("vae1"),
                    io.Vae.Input("vae2"),
                    io.Float.Input("multiplier", default=0.0, min=-10.0, max=10.0, step=0.01),
                ],
                outputs=[io.Vae.Output()],
            )

        @classmethod
        def execute(cls, vae1, vae2, multiplier: float) -> io.NodeOutput:
            factor = sign * multiplier
            return io.NodeOutput(_combine_vae(
                vae1, vae2, lambda key, a, b: (a.float() + factor * b.float()).to(a.dtype), label))

    _ArithNode.__name__ = _ArithNode.__qualname__ = f"VAEMerge{action}"
    return _ArithNode


VAEMergeAdd = _make_vae_arith_node("Add", +1.0)
VAEMergeSubtract = _make_vae_arith_node("Subtract", -1.0)


# ==============================================================================
# 6. Extension entry
# ==============================================================================

NODE_LIST: list[type[io.ComfyNode]] = [
    KeyNameInspector,
    *_build_block_nodes(),
    CLIPSaveQwen,
    VAEMergeSimple,
    VAEMergeAdd,
    VAEMergeSubtract,
]


class MergeNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODE_LIST


async def comfy_entrypoint() -> MergeNodesExtension:
    return MergeNodesExtension()