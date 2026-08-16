import numpy as np
import torch
from PIL import Image, ImageEnhance

from comfy_api.latest import io


_LAB_EPSILON = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0
_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    out = rgb / np.float32(12.92)
    high = rgb > np.float32(0.04045)
    if np.any(high):
        out[high] = ((rgb[high] + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4)
    return out


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    out = rgb * np.float32(12.92)
    high = rgb > np.float32(0.0031308)
    if np.any(high):
        out[high] = (
            np.float32(1.055) * np.power(np.maximum(rgb[high], np.float32(0.0)), np.float32(1.0 / 2.4))
            - np.float32(0.055)
        )
    return out


def _f_xyz_to_lab(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    out = (_LAB_KAPPA * t + np.float32(16.0)) / np.float32(116.0)
    high = t > np.float32(_LAB_EPSILON)
    if np.any(high):
        out[high] = np.cbrt(t[high])
    return out


def _f_lab_to_xyz(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    t3 = t * t * t
    out = (np.float32(116.0) * t - np.float32(16.0)) / np.float32(_LAB_KAPPA)
    high = t3 > np.float32(_LAB_EPSILON)
    if np.any(high):
        out[high] = t3[high]
    return out


def rgb_to_lab(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb_u8.astype(np.float32) / 255.0
    linear = _srgb_to_linear(rgb)

    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]

    x = (_SRGB_TO_XYZ[0, 0] * r + _SRGB_TO_XYZ[0, 1] * g + _SRGB_TO_XYZ[0, 2] * b) / _D65_WHITE[0]
    y = (_SRGB_TO_XYZ[1, 0] * r + _SRGB_TO_XYZ[1, 1] * g + _SRGB_TO_XYZ[1, 2] * b) / _D65_WHITE[1]
    z = (_SRGB_TO_XYZ[2, 0] * r + _SRGB_TO_XYZ[2, 1] * g + _SRGB_TO_XYZ[2, 2] * b) / _D65_WHITE[2]

    fx, fy, fz = _f_xyz_to_lab(x), _f_xyz_to_lab(y), _f_xyz_to_lab(z)

    L = (np.float32(116.0) * fy) - np.float32(16.0)
    a = np.float32(500.0) * (fx - fy)
    b_ch = np.float32(200.0) * (fy - fz)

    L_u8 = np.clip(L * (255.0 / 100.0), 0, 255).astype(np.uint8)
    return L_u8, a.astype(np.float32), b_ch.astype(np.float32)


def lab_to_rgb(L_u8: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    L = L_u8.astype(np.float32) * (100.0 / 255.0)

    fy = (L + np.float32(16.0)) / np.float32(116.0)
    fx = a / np.float32(500.0) + fy
    fz = fy - b / np.float32(200.0)

    x = _D65_WHITE[0] * _f_lab_to_xyz(fx)
    y = _D65_WHITE[1] * _f_lab_to_xyz(fy)
    z = _D65_WHITE[2] * _f_lab_to_xyz(fz)

    r_lin = _XYZ_TO_SRGB[0, 0] * x + _XYZ_TO_SRGB[0, 1] * y + _XYZ_TO_SRGB[0, 2] * z
    g_lin = _XYZ_TO_SRGB[1, 0] * x + _XYZ_TO_SRGB[1, 1] * y + _XYZ_TO_SRGB[1, 2] * z
    b_lin = _XYZ_TO_SRGB[2, 0] * x + _XYZ_TO_SRGB[2, 1] * y + _XYZ_TO_SRGB[2, 2] * z

    rgb_lin = np.stack([r_lin, g_lin, b_lin], axis=-1)
    rgb = _linear_to_srgb(rgb_lin)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def adjust_shadows_non_linear(
    luminance: np.ndarray,
    shadow_intensity: float,
    max_shadow_adjustment: float = 1.5,
) -> np.ndarray:
    lum_array = luminance.astype(np.float32) / 255.0
    shadows = lum_array ** (1 / (1 + shadow_intensity * max_shadow_adjustment))
    return np.clip(shadows * 255, 0, 255).astype(np.uint8)


def adjust_highlights_non_linear(
    luminance: np.ndarray,
    highlight_intensity: float,
    max_highlight_adjustment: float = 1.5,
) -> np.ndarray:
    lum_array = luminance.astype(np.float32) / 255.0
    highlights = 1 - (1 - lum_array) ** (1 + highlight_intensity * max_highlight_adjustment)
    return np.clip(highlights * 255, 0, 255).astype(np.uint8)


def merge_adjustments_with_blend_modes(
    luminance: np.ndarray,
    shadows: np.ndarray,
    highlights: np.ndarray,
    hdr_intensity: float,
    shadow_intensity: float,
    highlight_intensity: float,
) -> np.ndarray:
    base = luminance.astype(np.float32)

    scaled_shadow_intensity = shadow_intensity**2 * hdr_intensity
    scaled_highlight_intensity = highlight_intensity**2 * hdr_intensity

    shadow_mask = np.clip((1 - (base / 255)) ** 2, 0, 1)
    highlight_mask = np.clip((base / 255) ** 2, 0, 1)

    adjusted_shadows = np.clip(base * (1 - shadow_mask * scaled_shadow_intensity), 0, 255)
    adjusted_highlights = np.clip(
        base + (255 - base) * highlight_mask * scaled_highlight_intensity, 0, 255
    )

    adjusted_luminance = np.clip(adjusted_shadows + adjusted_highlights - base, 0, 255)
    final_luminance = np.clip(
        base * (1 - hdr_intensity) + adjusted_luminance * hdr_intensity, 0, 255
    ).astype(np.uint8)

    return final_luminance


def apply_gamma_correction(lum_array: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 0:
        return np.clip(lum_array, 0, 255).astype(np.uint8)
    gamma_corrected = 1 / (1.1 - gamma)
    adjusted = 255 * ((lum_array.astype(np.float32) / 255) ** gamma_corrected)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_midtone_weight(values: np.ndarray, adjustment_strength: float) -> np.ndarray:
    normalized = np.clip((values + 128.0) / 255.0, 0.0, 1.0)
    midtone_weight = 4.0 * normalized * (1.0 - normalized)
    adjustment = adjustment_strength * midtone_weight
    return (values * (1.0 + adjustment)).astype(np.float32)


def blend_ab_channels(
    original_a: np.ndarray,
    original_b: np.ndarray,
    adjusted_a: np.ndarray,
    adjusted_b: np.ndarray,
    ab_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    blended_a = original_a * (1.0 - ab_strength) + adjusted_a * ab_strength
    blended_b = original_b * (1.0 - ab_strength) + adjusted_b * ab_strength
    return blended_a.astype(np.float32), blended_b.astype(np.float32)


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
    if img.mode != "RGB":
        img = img.convert("RGB")

    rgb_u8 = np.asarray(img, dtype=np.uint8)
    L_u8, a, b = rgb_to_lab(rgb_u8)

    adjusted_a = a.copy()
    adjusted_b = b.copy()
    if a_adjustment != 0.0:
        adjusted_a = apply_midtone_weight(adjusted_a, a_adjustment)
    if b_adjustment != 0.0:
        adjusted_b = apply_midtone_weight(adjusted_b, b_adjustment)

    a_final, b_final = blend_ab_channels(a, b, adjusted_a, adjusted_b, ab_strength)

    shadows_adjusted = adjust_shadows_non_linear(L_u8, shadow_intensity)
    highlights_adjusted = adjust_highlights_non_linear(L_u8, highlight_intensity)
    merged = merge_adjustments_with_blend_modes(
        L_u8, shadows_adjusted, highlights_adjusted,
        hdr_intensity, shadow_intensity, highlight_intensity,
    )
    L_final = apply_gamma_correction(merged, gamma_intensity)

    rgb_out = lab_to_rgb(L_final, a_final, b_final)
    img_adjusted = Image.fromarray(rgb_out, mode="RGB")

    contrast_adjusted = ImageEnhance.Contrast(img_adjusted).enhance(1 + contrast)
    color_adjusted = ImageEnhance.Color(contrast_adjusted).enhance(1 + enhance_color * 0.2)
    return color_adjusted


class HDREffectsLabAdjust(io.ComfyNode):

    NODE_ID_LEGACY = "HDR Effects with LAB Adjust"
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
                    tooltip="Blend strength between original and adjusted A/B chroma channels.",
                ),
                io.Float.Input(
                    "a_adjustment",
                    default=0.03,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Midtone-weighted adjustment of the LAB A channel (green-red).",
                ),
                io.Float.Input(
                    "b_adjustment",
                    default=-0.05,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Midtone-weighted adjustment of the LAB B channel (blue-yellow).",
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
                    tooltip="Final color/saturation boost applied after LAB adjustments.",
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