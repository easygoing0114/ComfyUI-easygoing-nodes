from __future__ import annotations

import logging

import comfy.model_management as mm
from comfy_api.latest import io

logger = logging.getLogger(__name__)


def _find_loaded_entry(patcher) -> "mm.LoadedModel | None":
    """Return the LoadedModel entry in current_loaded_models that owns the given patcher."""
    for lm in mm.current_loaded_models:
        if lm.model is patcher:
            return lm
    return None


def _offload_vram(clip) -> None:
    """
    Move the CLIP patcher from VRAM to CPU (offload_device).

    ComfyUI's free_memory() selects eviction candidates via a scoring heuristic,
    so it cannot guarantee that CLIP will be unloaded. Here we operate directly
    on the patcher to ensure a deterministic offload.
    """
    patcher = clip.patcher  # ModelPatcher instance

    lm = _find_loaded_entry(patcher)
    if lm is None:
        logger.debug("CLIPOffload: CLIP is not in current_loaded_models, skipping VRAM offload")
        return

    loaded_mb = patcher.loaded_size() / (1024 ** 2)
    if loaded_mb <= 0:
        logger.debug("CLIPOffload: CLIP already offloaded from VRAM")
        return

    logger.info(f"CLIPOffload: Offloading CLIP from VRAM ({loaded_mb:.1f} MB loaded)")

    # LoadedModel.model_unload() internally calls patcher.detach(),
    # moving all GPU weights back to offload_device (CPU).
    lm.model_unload(unpatch_weights=True)

    # Remove from current_loaded_models so it is excluded from future scoring.
    try:
        mm.current_loaded_models.remove(lm)
    except ValueError:
        pass

    mm.soft_empty_cache()
    logger.info("CLIPOffload: CLIP offloaded from VRAM")


def _offload_ram(clip) -> None:
    """
    Release pinned (page-locked) memory held by the CLIP patcher.

    partially_unload_ram() is only implemented in ModelPatcherDynamic;
    the base ModelPatcher version is a no-op. We therefore call
    unpin_all_weights() directly to achieve equivalent behaviour on both.
    """
    patcher = clip.patcher

    # VRAM offload must come first; pinned memory lives on CPU weights.
    _offload_vram(clip)

    if hasattr(patcher, "unpin_all_weights"):
        pinned_count = len(patcher.pinned)
        if pinned_count > 0:
            logger.info(f"CLIPOffload: Unpinning {pinned_count} weight(s) from RAM")
            patcher.unpin_all_weights()
        else:
            logger.debug("CLIPOffload: No pinned weights to release")

    # For ModelPatcherDynamic also call partially_unload_ram() to free all staged memory.
    if hasattr(patcher, "partially_unload_ram") and patcher.is_dynamic():
        logger.info("CLIPOffload: Calling partially_unload_ram() for dynamic patcher")
        patcher.partially_unload_ram(1e32)  # free everything

    mm.soft_empty_cache()
    logger.info("CLIPOffload: CLIP unpinned from RAM")


# ------------------------------------------------------------------------------
# Node: CLIP Text Encode (with Offload)
# ------------------------------------------------------------------------------

class CLIPTextEncodeWithOffload(io.ComfyNode):
    """
    Extends the standard CLIPTextEncode node with VRAM and RAM offload toggles.

    offload_from_vram
        When True, moves the CLIP model from VRAM to CPU after encoding.
        The conditioning tensor has already been returned, so this is safe.
        On repeated generations with the same prompt, ComfyUI's output cache
        prevents this node from re-executing, so CLIP is not reloaded.

    offload_from_ram
        When True, performs the VRAM offload and additionally releases
        pinned (page-locked) memory. Useful for further reducing RAM usage.
        Note: the next load will be slightly slower without pinned memory.
    """

    NODE_ID_LEGACY = "CLIPTextEncodeWithOffload"
    # Widget-only inputs, in the order used by the V1 INPUT_TYPES (excludes the
    # "clip" socket input). Required for positional widget-value migration.
    NODE_ID_INPUT_ORDER = (
        "text",
        "offload_from_vram",
        "offload_from_ram",
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_CLIPTextEncodeWithOffload",
            display_name="CLIP Text Encode (with Offload)",
            category="conditioning",
            description=(
                "CLIPTextEncode with optional VRAM and RAM offload. "
                "Releases the CLIP model from VRAM and/or RAM after encoding to save memory. "
                "ComfyUI's output cache ensures the node does not re-execute on repeated "
                "generations with the same prompt, keeping performance impact minimal."
            ),
            search_aliases=[
                "text", "prompt", "text prompt", "positive prompt", "negative prompt",
                "encode text", "text encoder", "encode prompt", "offload", "vram", "ram",
            ],
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="The text prompt to encode.",
                ),
                io.Clip.Input(
                    "clip",
                    tooltip="The CLIP model used for encoding the text.",
                ),
                io.Boolean.Input(
                    "offload_from_vram",
                    default=True,
                    tooltip=(
                        "When True, unloads the CLIP model from VRAM to CPU after encoding. "
                        "Repeated generations with the same prompt use the cache, "
                        "so no re-encoding occurs."
                    ),
                ),
                io.Boolean.Input(
                    "offload_from_ram",
                    default=True,
                    tooltip=(
                        "When True, also releases pinned (page-locked) memory after encoding. "
                        "offload_from_vram is implied. "
                        "Reduces RAM usage further but slightly slows the next load."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    tooltip="A conditioning containing the embedded text used to guide the diffusion model.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        text: str,
        offload_from_vram: bool,
        offload_from_ram: bool,
    ) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node, your checkpoint does not "
                "contain a valid clip or text encoder model."
            )

        # --- Encode (identical to the standard CLIPTextEncode node) ---
        tokens = clip.tokenize(text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # --- Offload logic (offload_from_ram implies offload_from_vram) ---
        if offload_from_ram:
            try:
                _offload_ram(clip)
            except Exception as e:
                logger.warning(f"CLIPOffload: RAM offload failed (non-fatal): {e}")
        elif offload_from_vram:
            try:
                _offload_vram(clip)
            except Exception as e:
                logger.warning(f"CLIPOffload: VRAM offload failed (non-fatal): {e}")

        return io.NodeOutput(conditioning)


NODE_LIST = [
    CLIPTextEncodeWithOffload,
]