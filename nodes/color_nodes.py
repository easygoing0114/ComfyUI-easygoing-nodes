import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageCms

# LABカラープロファイル
sRGB_profile = ImageCms.createProfile("sRGB")
Lab_profile = ImageCms.createProfile("LAB")


# ---- ヘルパー関数 ----

def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def adjust_shadows_non_linear(luminance, shadow_intensity, max_shadow_adjustment=1.5):
    lum_array = np.array(luminance, dtype=np.float32) / 255.0
    shadows = lum_array ** (1 / (1 + shadow_intensity * max_shadow_adjustment))
    return np.clip(shadows * 255, 0, 255).astype(np.uint8)


def adjust_highlights_non_linear(
    luminance, highlight_intensity, max_highlight_adjustment=1.5
):
    lum_array = np.array(luminance, dtype=np.float32) / 255.0
    highlights = 1 - (1 - lum_array) ** (
        1 + highlight_intensity * max_highlight_adjustment
    )
    return np.clip(highlights * 255, 0, 255).astype(np.uint8)


def merge_adjustments_with_blend_modes(
    luminance, shadows, highlights, hdr_intensity, shadow_intensity, highlight_intensity
):
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


def apply_gamma_correction(lum_array, gamma):
    if gamma == 0:
        return np.clip(lum_array, 0, 255).astype(np.uint8)
    gamma_corrected = 1 / (1.1 - gamma)
    adjusted = 255 * ((lum_array / 255) ** gamma_corrected)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_midtone_weight(values, adjustment_strength):
    """中間調部分で最大の変化を持ち、両極で変化が少ない重み付けを適用"""
    normalized = values / 255.0
    midtone_weight = 4.0 * normalized * (1.0 - normalized)
    adjustment = adjustment_strength * midtone_weight
    adjusted_values = values * (1.0 + adjustment)
    return np.clip(adjusted_values, 0, 255).astype(np.uint8)


def blend_ab_channels(original_a, original_b, adjusted_a, adjusted_b, ab_strength):
    """元のA/Bチャンネルと調整後のA/Bチャンネルを指定した強度でブレンドする"""
    orig_a_array = np.array(original_a, dtype=np.float32)
    orig_b_array = np.array(original_b, dtype=np.float32)
    adj_a_array = np.array(adjusted_a, dtype=np.float32)
    adj_b_array = np.array(adjusted_b, dtype=np.float32)

    blended_a = orig_a_array * (1 - ab_strength) + adj_a_array * ab_strength
    blended_b = orig_b_array * (1 - ab_strength) + adj_b_array * ab_strength

    blended_a = np.clip(blended_a, 0, 255).astype(np.uint8)
    blended_b = np.clip(blended_b, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_a), Image.fromarray(blended_b)


def apply_to_batch(func):
    def wrapper(self, image, *args, **kwargs):
        images = [func(self, img, *args, **kwargs) for img in image]
        return (torch.cat(images, dim=0),)
    return wrapper


# ---- ノードクラス ----

class HDREffectsLabAdjust:
    DESCRIPTION = "Apply HDR tone-mapping with control over shadows, highlights, gamma, contrast, color boost, and LAB A/B channel adjustments with blend strength."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hdr_intensity": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "shadow_intensity": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "highlight_intensity": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "gamma_intensity": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "ab_strength": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "a_adjustment": (
                    "FLOAT",
                    {"default": 0.03, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "b_adjustment": (
                    "FLOAT",
                    {"default": -0.05, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "enhance_color": (
                    "FLOAT",
                    {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_img",)
    FUNCTION = "apply_hdr2"
    CATEGORY = "SuperBeastsAI/Image"

    @apply_to_batch
    def apply_hdr2(
        self,
        image,
        hdr_intensity=0.75,
        shadow_intensity=0.75,
        highlight_intensity=0.25,
        ab_strength=0.1,
        a_adjustment=0.03,
        b_adjustment=-0.05,
        gamma_intensity=0,
        contrast=0.1,
        enhance_color=0.25,
    ):
        img = tensor2pil(image)

        # LABへ変換
        img_lab = ImageCms.profileToProfile(
            img, sRGB_profile, Lab_profile, outputMode="LAB"
        )
        luminance, a, b = img_lab.split()

        lum_array = np.array(luminance, dtype=np.float32)
        a_array = np.array(a, dtype=np.float32)
        b_array = np.array(b, dtype=np.float32)

        # A/BチャンネルへのMidtone重み付き調整
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

        # HDR調整
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

        # ガンマ補正
        gamma_corrected = apply_gamma_correction(
            np.array(merged_adjustments), gamma_intensity
        )
        gamma_corrected = Image.fromarray(gamma_corrected).resize(a.size)

        # LABチャンネルをマージしてRGBに戻す
        adjusted_lab = Image.merge("LAB", (gamma_corrected, a_adjusted, b_adjusted))
        img_adjusted = ImageCms.profileToProfile(
            adjusted_lab, Lab_profile, sRGB_profile, outputMode="RGB"
        )

        # コントラスト・カラー強調
        contrast_adjusted = ImageEnhance.Contrast(img_adjusted).enhance(1 + contrast)
        color_adjusted = ImageEnhance.Color(contrast_adjusted).enhance(
            1 + enhance_color * 0.2
        )

        return pil2tensor(color_adjusted)


# ---- ノード登録用マッピング ----

NODE_CLASS_MAPPINGS = {
    "HDR Effects with LAB Adjust": HDREffectsLabAdjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HDREffectsLabAdjust": "HDR Effects with LAB Adjusts",
}
