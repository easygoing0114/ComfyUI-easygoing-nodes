import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageCms

from comfy_api.latest import io

# ICC profiles used for the sRGB <-> LAB round trip.
SRGB_PROFILE = ImageCms.createProfile("sRGB")
LAB_PROFILE = ImageCms.createProfile("LAB")

def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Convert a single ComfyUI IMAGE tensor (H, W, C), float32 in [0, 1],
    into a PIL Image (uint8, [0, 255])."""
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image (uint8) into a ComfyUI IMAGE tensor
    (1, H, W, C), float32 in [0, 1]."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def adjust_shadows_non_linear(
    luminance: Image.Image,
    shadow_intensity: float,
    max_shadow_adjustment: float = 1.5,
) -> np.ndarray:
    """Brighten/darken shadows using a power-law (gamma-like) curve."""
    lum_array = np.array(luminance, dtype=np.float32) / 255.0
    shadows = lum_array ** (1 / (1 + shadow_intensity * max_shadow_adjustment))
    return np.clip(shadows * 255, 0, 255).astype(np.uint8)


def adjust_highlights_non_linear(
    luminance: Image.Image,
    highlight_intensity: float,
    max_highlight_adjustment: float = 1.5,
) -> np.ndarray:
    """Compress/expand highlights using an inverse power-law curve."""
    lum_array = np.array(luminance, dtype=np.float32) / 255.0
    highlights = 1 - (1 - lum_array) ** (
        1 + highlight_intensity * max_highlight_adjustment
    )
    return np.clip(highlights * 255, 0, 255).astype(np.uint8)


def merge_adjustments_with_blend_modes(
    luminance: np.ndarray,
    shadows: np.ndarray,
    highlights: np.ndarray,
    hdr_intensity: float,
    shadow_intensity: float,
    highlight_intensity: float,
) -> Image.Image:
    """Blend the shadow- and highlight-adjusted luminance back with the
    original luminance, weighted by per-pixel shadow/highlight masks and
    the overall HDR intensity."""
    base = np.array(luminance, dtype=np.float32)

    scaled_shadow_intensity = shadow_intensity**2 * hdr_intensity
    scaled_highlight_intensity = highlight_intensity**2 * hdr_intensity

    shadow_mask = np.clip((1 - (base / 255)) ** 2, 0, 1)
    highlight_mask = np.clip((base / 255) ** 2, 0, 1)

    adjusted_shadows = np.clip(
        base * (1 - shadow_mask * scaled_shadow_intensity), 0, 255
    )
    adjusted_highlights = np.clip(
        base + (255 - base) * highlight_mask * scaled_highlight_intensity, 0, 255
    )

    adjusted_luminance = np.clip(adjusted_shadows + adjusted_highlights - base, 0, 255)
    final_luminance = np.clip(
        base * (1 - hdr_intensity) + adjusted_luminance * hdr_intensity, 0, 255
    ).astype(np.uint8)

    return Image.fromarray(final_luminance)


def apply_gamma_correction(lum_array: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to a luminance array. `gamma == 0` is treated
    as a no-op (pass-through)."""
    if gamma == 0:
        return np.clip(lum_array, 0, 255).astype(np.uint8)
    gamma_corrected = 1 / (1.1 - gamma)
    adjusted = 255 * ((lum_array / 255) ** gamma_corrected)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Chroma (A/B channel) adjustments
# ---------------------------------------------------------------------------

def apply_midtone_weight(values: np.ndarray, adjustment_strength: float) -> np.ndarray:
    """Apply a weighting curve that peaks in the midtones and tapers off
    toward both extremes, so shifts are strongest where they're least
    likely to clip highlights/shadows."""
    normalized = values / 255.0
    midtone_weight = 4.0 * normalized * (1.0 - normalized)
    adjustment = adjustment_strength * midtone_weight
    adjusted_values = values * (1.0 + adjustment)
    return np.clip(adjusted_values, 0, 255).astype(np.uint8)


def blend_ab_channels(
    original_a: Image.Image,
    original_b: Image.Image,
    adjusted_a: Image.Image,
    adjusted_b: Image.Image,
    ab_strength: float,
) -> tuple[Image.Image, Image.Image]:
    """Linearly blend the original and adjusted A/B channels by `ab_strength`
    (0 = original untouched, 1 = fully adjusted)."""
    orig_a_array = np.array(original_a, dtype=np.float32)
    orig_b_array = np.array(original_b, dtype=np.float32)
    adj_a_array = np.array(adjusted_a, dtype=np.float32)
    adj_b_array = np.array(adjusted_b, dtype=np.float32)

    blended_a = orig_a_array * (1 - ab_strength) + adj_a_array * ab_strength
    blended_b = orig_b_array * (1 - ab_strength) + adj_b_array * ab_strength

    blended_a = np.clip(blended_a, 0, 255).astype(np.uint8)
    blended_b = np.clip(blended_b, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_a), Image.fromarray(blended_b)

def process_one_image(
    img: Image.Image,
    hdr_intensity: float,
    shadow_intensity: float,
    highlight_intensity: float,
    ab_strength: float,
    a_adjustment: float,
    b_adjustment: float,
    gamma_intensity: float,
    contrast: float,
    enhance_color: float,
) -> Image.Image:
    """Apply the full HDR/LAB adjustment pipeline to a single PIL image and
    return the resulting PIL image."""
    # sRGB -> LAB
    img_lab = ImageCms.profileToProfile(
        img, SRGB_PROFILE, LAB_PROFILE, outputMode="LAB"
    )
    luminance, a, b = img_lab.split()

    lum_array = np.array(luminance, dtype=np.float32)
    a_array = np.array(a, dtype=np.float32)
    b_array = np.array(b, dtype=np.float32)

    # Midtone-weighted adjustment of the A/B chroma channels.
    adjusted_a_array = a_array.copy()
    adjusted_b_array = b_array.copy()

    if a_adjustment != 0.0:
        adjusted_a_array = apply_midtone_weight(adjusted_a_array, a_adjustment)
    if b_adjustment != 0.0:
        adjusted_b_array = apply_midtone_weight(adjusted_b_array, b_adjustment)

    a_adjusted_temp = Image.fromarray(adjusted_a_array.astype(np.uint8))
    b_adjusted_temp = Image.fromarray(adjusted_b_array.astype(np.uint8))

    a_adjusted, b_adjusted = blend_ab_channels(
        a, b, a_adjusted_temp, b_adjusted_temp, ab_strength
    )

    # Shadow/highlight (HDR) adjustment of the L channel.
    shadows_adjusted = adjust_shadows_non_linear(luminance, shadow_intensity)
    highlights_adjusted = adjust_highlights_non_linear(luminance, highlight_intensity)
    merged_adjustments = merge_adjustments_with_blend_modes(
        lum_array,
        shadows_adjusted,
        highlights_adjusted,
        hdr_intensity,
        shadow_intensity,
        highlight_intensity,
    )

    # Gamma correction.
    gamma_corrected = apply_gamma_correction(
        np.array(merged_adjustments), gamma_intensity
    )
    gamma_corrected = Image.fromarray(gamma_corrected).resize(a.size)

    # Reassemble LAB -> sRGB.
    adjusted_lab = Image.merge("LAB", (gamma_corrected, a_adjusted, b_adjusted))
    img_adjusted = ImageCms.profileToProfile(
        adjusted_lab, LAB_PROFILE, SRGB_PROFILE, outputMode="RGB"
    )

    # Final contrast / saturation boost.
    contrast_adjusted = ImageEnhance.Contrast(img_adjusted).enhance(1 + contrast)
    color_adjusted = ImageEnhance.Color(contrast_adjusted).enhance(
        1 + enhance_color * 0.2
    )

    return color_adjusted

class HDREffectsLabAdjust(io.ComfyNode):

    NODE_ID_LEGACY = "HDR Effects with LAB Adjust"
    # Widget-only inputs, in the order used by the V1 INPUT_TYPES (excludes the
    # "image" socket input). Required for positional widget-value migration.
    NODE_ID_INPUT_ORDER = (
        "hdr_intensity",
        "shadow_intensity",
        "highlight_intensity",
        "gamma_intensity",
        "ab_strength",
        "a_adjustment",
        "b_adjustment",
        "contrast",
        "enhance_color",
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_HDREffectsLabAdjust",
            display_name="HDR Effects with LAB Adjusts",
            category="SuperBeastsAI/Image",
            description=(
                "Apply HDR tone-mapping with control over shadows, highlights, "
                "gamma, contrast, color boost, and LAB A/B channel adjustments "
                "with blend strength."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Image batch to process."),
                io.Float.Input(
                    "hdr_intensity",
                    default=0.75,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    tooltip="Overall strength of the HDR shadow/highlight effect.",
                ),
                io.Float.Input(
                    "shadow_intensity",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Strength of the shadow tone-mapping curve.",
                ),
                io.Float.Input(
                    "highlight_intensity",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Strength of the highlight tone-mapping curve.",
                ),
                io.Float.Input(
                    "gamma_intensity",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Gamma correction applied after tone-mapping.",
                ),
                io.Float.Input(
                    "ab_strength",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Blend strength between original and adjusted A/B "
                    "chroma channels.",
                ),
                io.Float.Input(
                    "a_adjustment",
                    default=0.03,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Midtone-weighted adjustment of the LAB A channel "
                    "(green-red).",
                ),
                io.Float.Input(
                    "b_adjustment",
                    default=-0.05,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Midtone-weighted adjustment of the LAB B channel "
                    "(blue-yellow).",
                ),
                io.Float.Input(
                    "contrast",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Final contrast boost applied after LAB adjustments.",
                ),
                io.Float.Input(
                    "enhance_color",
                    default=0.03,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Final color/saturation boost applied after LAB "
                    "adjustments.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="result_img"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        hdr_intensity: float,
        shadow_intensity: float,
        highlight_intensity: float,
        gamma_intensity: float,
        ab_strength: float,
        a_adjustment: float,
        b_adjustment: float,
        contrast: float,
        enhance_color: float,
    ) -> io.NodeOutput:
        results = [
            pil2tensor(
                process_one_image(
                    tensor2pil(img),
                    hdr_intensity,
                    shadow_intensity,
                    highlight_intensity,
                    ab_strength,
                    a_adjustment,
                    b_adjustment,
                    gamma_intensity,
                    contrast,
                    enhance_color,
                )
            )
            for img in image
        ]
        result_batch = torch.cat(results, dim=0)
        return io.NodeOutput(result_batch)

NODE_LIST = [
    HDREffectsLabAdjust,
]