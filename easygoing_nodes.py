import os
import json
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageCms
from PIL.PngImagePlugin import PngInfo
import folder_paths
import comfy.sd
from comfy.cli_args import args
import comfy_extras.nodes_model_merging

# LAB color space profiles
sRGB_profile = ImageCms.createProfile("sRGB")
Lab_profile = ImageCms.createProfile("LAB")


# Helper functions
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
    epsilon = 1e-7
    gamma_corrected = 1 / (1.1 - gamma)
    adjusted = 255 * ((lum_array / 255) ** gamma_corrected)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_midtone_weight(values, adjustment_strength):
    """
    中間調部分で最大の変化を持ち、両極で変化が少ない重み付けを適用
    values: 0-255の値の配列
    adjustment_strength: 調整の強さ
    """
    # 0-1の範囲に正規化
    normalized = values / 255.0

    # ベル曲線のような重み（中間調で最大、両極で最小）
    # 0.5で最大値1.0、0と1で最小値0を持つ関数
    midtone_weight = 4.0 * normalized * (1.0 - normalized)  # 0.5で1.0、0と1で0

    # 調整値を計算（重み付きで適用）
    adjustment = adjustment_strength * midtone_weight

    # 調整を適用
    adjusted_values = values * (1.0 + adjustment)

    return np.clip(adjusted_values, 0, 255).astype(np.uint8)


def blend_ab_channels(original_a, original_b, adjusted_a, adjusted_b, ab_strength):
    """
    元のA/Bチャンネルと調整後のA/Bチャンネルを指定した強度でブレンドする
    """
    # NumPy配列に変換
    orig_a_array = np.array(original_a, dtype=np.float32)
    orig_b_array = np.array(original_b, dtype=np.float32)
    adj_a_array = np.array(adjusted_a, dtype=np.float32)
    adj_b_array = np.array(adjusted_b, dtype=np.float32)

    # ブレンド (ab_strength=0で元画像、ab_strength=1で調整後)
    blended_a = orig_a_array * (1 - ab_strength) + adj_a_array * ab_strength
    blended_b = orig_b_array * (1 - ab_strength) + adj_b_array * ab_strength

    # クリッピングしてPILイメージに変換
    blended_a = np.clip(blended_a, 0, 255).astype(np.uint8)
    blended_b = np.clip(blended_b, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_a), Image.fromarray(blended_b)


def apply_to_batch(func):
    def wrapper(self, image, *args, **kwargs):
        images = [func(self, img, *args, **kwargs) for img in image]
        return (torch.cat(images, dim=0),)

    return wrapper


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

        # Convert to LAB
        img_lab = ImageCms.profileToProfile(
            img, sRGB_profile, Lab_profile, outputMode="LAB"
        )
        luminance, a, b = img_lab.split()

        # Convert to NumPy arrays
        lum_array = np.array(luminance, dtype=np.float32)
        a_array = np.array(a, dtype=np.float32)
        b_array = np.array(b, dtype=np.float32)

        # Apply midtone-weighted adjustments to A and B channels
        adjusted_a_array = a_array.copy()
        adjusted_b_array = b_array.copy()

        if a_adjustment != 0.0:
            adjusted_a_array = apply_midtone_weight(adjusted_a_array, a_adjustment)
        if b_adjustment != 0.0:
            adjusted_b_array = apply_midtone_weight(adjusted_b_array, b_adjustment)

        # Create adjusted A/B channel images
        a_adjusted_temp = Image.fromarray(adjusted_a_array.astype(np.uint8))
        b_adjusted_temp = Image.fromarray(adjusted_b_array.astype(np.uint8))

        # Blend original and adjusted A/B channels using ab_strength
        a_adjusted, b_adjusted = blend_ab_channels(
            a, b, a_adjusted_temp, b_adjusted_temp, ab_strength
        )

        # Apply HDR adjustments
        shadows_adjusted = adjust_shadows_non_linear(luminance, shadow_intensity)
        highlights_adjusted = adjust_highlights_non_linear(
            luminance, highlight_intensity
        )
        merged_adjustments = merge_adjustments_with_blend_modes(
            lum_array,
            shadows_adjusted,
            highlights_adjusted,
            hdr_intensity,
            shadow_intensity,
            highlight_intensity,
        )

        # Apply gamma correction
        gamma_corrected = apply_gamma_correction(
            np.array(merged_adjustments), gamma_intensity
        )
        gamma_corrected = Image.fromarray(gamma_corrected).resize(a.size)

        # Merge LAB channels
        adjusted_lab = Image.merge("LAB", (gamma_corrected, a_adjusted, b_adjusted))

        # Convert back to RGB
        img_adjusted = ImageCms.profileToProfile(
            adjusted_lab, Lab_profile, sRGB_profile, outputMode="RGB"
        )

        # Enhance contrast and color
        contrast_adjusted = ImageEnhance.Contrast(img_adjusted).enhance(1 + contrast)
        color_adjusted = ImageEnhance.Color(contrast_adjusted).enhance(
            1 + enhance_color * 0.2
        )

        return pil2tensor(color_adjusted)


class SaveImageWithPrompt:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "positive_prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "caption": ("STRING", {"default": ""}),
                "numbers": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Include Numbers",
                        "label_off": "No Numbers",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves images to your ComfyUI output directory with positive and negative prompts and caption in metadata."

    def save_images(
        self,
        images,
        filename_prefix="ComfyUI",
        positive_prompt="",
        negative_prompt="",
        caption="",
        numbers=True,
        prompt=None,
        extra_pnginfo=None,
    ):
        # Truncate filename_prefix to 200 characters if it exceeds that length
        if len(filename_prefix) > 180:
            filename_prefix = filename_prefix[:180]

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()

        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                # 隠し入力のpromptを保存
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                # positive_promptを保存
                if positive_prompt:
                    metadata.add_text("positive_prompt", json.dumps(positive_prompt))
                # negative_promptを保存
                if negative_prompt:
                    metadata.add_text("negative_prompt", json.dumps(negative_prompt))
                # captionを保存
                if caption:
                    metadata.add_text("caption", json.dumps(caption))
                # extra_pnginfoを保存
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

            # numbersトグルによってファイル名の形式を変更
            if numbers:
                file = f"{filename_with_batch_num}_{counter:05}_.png"
            else:
                file = f"{filename_with_batch_num}.png"

            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}

class ModelScaleSDXL(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    """
    SDXL モデルの特定の層をスケーリングするノード。
    scale=1.0 で元のまま、scale=0.0 でゼロ、1.0以上で強調します。
    ModelMergeBlocksを継承し、単一モデルの各層に対してスケール係数を適用します。
    """

    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"model": ("MODEL",)}

        # スケーリング用の引数設定（デフォルト1.0、範囲は0.0〜2.0）
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})

        arg_dict["time_embed."] = argument
        arg_dict["label_emb."] = argument

        for i in range(9):
            arg_dict["input_blocks.{}".format(i)] = argument

        for i in range(3):
            arg_dict["middle_block.{}".format(i)] = argument

        for i in range(9):
            arg_dict["output_blocks.{}".format(i)] = argument

        arg_dict["out."] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "scale"

    def scale(self, model, **kwargs):
        """
        モデルの各層をスケーリングする

        Args:
            model: 入力モデル
            **kwargs: 各層のスケール値

        Returns:
            tuple: スケーリング済みモデル
        """
        m = model.clone()

        # diffusion_model 以下のパラメータを対象とする
        kp = m.get_key_patches("diffusion_model.")

        for k in kp:
            scale_value = 1.0
            k_unet = k[len("diffusion_model."):]

            # 最も長く一致するプレフィックスを探す
            matched_arg_len = 0
            for arg_name, arg_value in kwargs.items():
                if k_unet.startswith(arg_name) and len(arg_name) > matched_arg_len:
                    scale_value = arg_value
                    matched_arg_len = len(arg_name)

            # W_new = W * scale_value
            # add_patches: output = W * strength_model + P * strength_patch
            # P = W（元の重み）なので: output = W * 1.0 + W * (scale_value - 1.0) = W * scale_value
            if scale_value != 1.0:
                m.add_patches({k: kp[k]}, scale_value - 1.0, 1.0)

        return (m,)

class ModelMergeHiDream(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"
    DESCRIPTION = "Merge node for HiDream series models (Full, Dev, Fast). Assumes double_stream_blocks 0-12 and single_stream_blocks 0-31 based on provided keys."

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"model1": ("MODEL",), "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["x_embedder."] = argument
        arg_dict["t_embedder."] = argument
        arg_dict["caption_projection."] = argument

        for i in range(13):
            arg_dict["double_stream_blocks.{}.".format(i)] = argument

        for i in range(32):
            arg_dict["single_stream_blocks.{}.".format(i)] = argument

        return {"required": arg_dict}
    
class ModelScaleHiDream:
    """
    HiDream系モデル（Full, Dev, Fast）の特定の層をスケーリングするノード。
    scale=1.0 で元のまま、scale=0.0 でゼロ、1.0以上で強調します。
    double_stream_blocks 0-12 と single_stream_blocks 0-31 に対応。
    """
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"model": ("MODEL",)}
        # スケーリング用の引数設定（デフォルト1.0、範囲は0.0〜2.0）
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        
        # 基本コンポーネント
        arg_dict["x_embedder."] = argument
        arg_dict["t_embedder."] = argument
        arg_dict["caption_projection."] = argument
        
        # Double Stream Blocks (0-12)
        for i in range(13):
            arg_dict["double_stream_blocks.{}.".format(i)] = argument
        
        # Single Stream Blocks (0-31)
        for i in range(32):
            arg_dict["single_stream_blocks.{}.".format(i)] = argument
        
        return {"required": arg_dict}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"
    DESCRIPTION = "Scale specific layers of HiDream series models (Full, Dev, Fast). Assumes double_stream_blocks 0-12 and single_stream_blocks 0-31."
    
    def scale(self, model, **kwargs):
        # モデルの複製
        m = model.clone()
        
        # スケーリング比率の辞書（'model'キー以外を抽出）
        ratios = {k: v for k, v in kwargs.items() if k != "model"}
        
        # モデルのパッチ可能なキー（重み）を取得
        kp = m.get_key_patches("diffusion_model.")
        
        # 全ての重みキーに対してスケーリングを適用
        for k in kp:
            scale_value = 1.0
            # diffusion_model. を除いた純粋なレイヤー名
            k_unet = k[len("diffusion_model."):]
            
            # 最も長く一致するプレフィックスを探すロジック
            matched_arg_len = 0
            for arg_name, arg_value in ratios.items():
                if k_unet.startswith(arg_name):
                    if len(arg_name) > matched_arg_len:
                        scale_value = arg_value
                        matched_arg_len = len(arg_name)
            
            # スケーリングの適用
            # scale_value != 1.0 の場合のみパッチを適用
            if scale_value != 1.0:
                # kp[k] はすでに適切なパッチ形式
                # add_patches(patches_dict, strength_patch, strength_model)
                # 出力 = weight * strength_model + patch * strength_patch
                # スケーリングを実現: weight * scale_value
                # = weight * 1.0 + weight * (scale_value - 1.0)
                m.add_patches({k: kp[k]}, scale_value - 1.0, 1.0)
        
        return (m,)

class ModelScaleQwenImage:
    """
    Qwen Image Modelの特定の層をスケーリングするノード。
    scale=1.0 で元のまま、scale=0.0 でゼロ、1.0以上で強調します。
    """

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"model": ("MODEL",)}

        # スケーリング用の引数設定（デフォルト1.0、範囲は0.0〜2.0程度に設定）
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})

        # 基本コンポーネント
        arg_dict["pos_embeds."] = argument
        arg_dict["img_in."] = argument
        arg_dict["txt_norm."] = argument
        arg_dict["txt_in."] = argument
        arg_dict["time_text_embed."] = argument

        # Transformer Block (0-59)
        for i in range(60):
            arg_dict["transformer_blocks.{}.".format(i)] = argument

        arg_dict["proj_out."] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"

    def scale(self, model, **kwargs):
        # モデルの複製
        m = model.clone()

        # スケーリング比率の辞書（'model'キー以外を抽出）
        ratios = {k: v for k, v in kwargs.items() if k != "model"}

        # モデルのパッチ可能なキー（重み）を取得
        # diffusion_model 以下のパラメータを対象とする
        kp = m.get_key_patches("diffusion_model.")

        # 全ての重みキーに対してスケーリングを適用
        for k in kp:
            scale_value = 1.0
            # diffusion_model. を除いた純粋なレイヤー名
            k_unet = k[len("diffusion_model.") :]

            # 最も長く一致するプレフィックスを探すロジック（ModelMergeBlocks参照）
            matched_arg_len = 0
            for arg_name, arg_value in ratios.items():
                if k_unet.startswith(arg_name):
                    if len(arg_name) > matched_arg_len:
                        scale_value = arg_value
                        matched_arg_len = len(arg_name)

            # スケーリングの適用
            # ComfyUIのadd_patchesは (patch, strength_patch, strength_model) を計算する
            # Output = W * strength_model + P * strength_patch
            # スケーリングを行うため: W_new = W * scale_value としたい
            # ここでは W * 1.0 + W * (scale_value - 1.0) として実装する
            if scale_value != 1.0:
                # kp[k] は (tensor,) のタプル
                weight_tensor = kp[k][0]
                # 元の重みに対して (scale - 1.0) 分をパッチとして追加することで乗算を実現
                m.add_patches({k: (weight_tensor,)}, scale_value - 1.0, 1.0)

        return (m,)

class ModelMergeZImage(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    """
    Z-Image Model専用のマージノード。
    各層ごとに異なるマージ比率を設定できます。
    ratio=1.0 でmodel2を100%使用、ratio=0.0 でmodel1を100%使用します。
    """
    
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "model1": ("MODEL",),
            "model2": ("MODEL",)
        }

        # マージ比率の引数設定（デフォルト1.0、範囲は0.0〜1.0）
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        # Caption Embedder
        arg_dict["cap_embedder."] = argument
        arg_dict["cap_pad_token"] = argument
        
        # Context Refiner (2層: 0-1)
        for i in range(2):
            arg_dict["context_refiner.{}.".format(i)] = argument
        
        # Main Layers (30層: 0-29)
        for i in range(30):
            arg_dict["layers.{}.".format(i)] = argument
        
        # Noise Refiner (2層: 0-1)
        for i in range(2):
            arg_dict["noise_refiner.{}.".format(i)] = argument
        
        # Final Layer
        arg_dict["final_layer."] = argument
        
        # Time Embedder
        arg_dict["t_embedder."] = argument
        
        # Image Embedder
        arg_dict["x_embedder."] = argument
        arg_dict["x_pad_token"] = argument

        return {"required": arg_dict}

class ModelScaleZImage:
    """
    Z-Image Modelの特定の層をスケーリングするノード。
    scale=1.0 で元のまま、scale=0.0 でゼロ、1.0以上で強調します。
    """
    
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"model": ("MODEL",)}
        
        # スケーリング用の引数設定（デフォルト1.0、範囲は0.0〜2.0）
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        
        # Caption Embedder
        arg_dict["cap_embedder."] = argument
        arg_dict["cap_pad_token"] = argument
        
        # Context Refiner (2層)
        for i in range(2):
            arg_dict["context_refiner.{}.".format(i)] = argument
        
        # Main Layers (30層: 0-29)
        for i in range(30):
            arg_dict["layers.{}.".format(i)] = argument
        
        # Noise Refiner (2層)
        for i in range(2):
            arg_dict["noise_refiner.{}.".format(i)] = argument
        
        # Final Layer
        arg_dict["final_layer."] = argument
        
        # Time Embedder
        arg_dict["t_embedder."] = argument
        
        # Image Embedder
        arg_dict["x_embedder."] = argument
        arg_dict["x_pad_token"] = argument
        
        return {"required": arg_dict}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"
    
    def scale(self, model, **kwargs):
        """
        モデルの各層をスケーリングする
        
        Args:
            model: 入力モデル
            **kwargs: 各層のスケール値
            
        Returns:
            tuple: スケーリング済みモデル
        """
        # モデルの複製
        m = model.clone()
        
        # スケーリング比率の辞書（'model'キー以外を抽出）
        ratios = {k: v for k, v in kwargs.items() if k != "model"}
        
        # モデルのパッチ可能なキー（重み）を取得
        # diffusion_model 以下のパラメータを対象とする
        kp = m.get_key_patches("diffusion_model.")
        
        # 全ての重みキーに対してスケーリングを適用
        for k in kp:
            scale_value = 1.0
            
            # diffusion_model. を除いた純粋なレイヤー名
            k_unet = k[len("diffusion_model."):]
            
            # 最も長く一致するプレフィックスを探すロジック
            # （ModelMergeBlocks参照）
            matched_arg_len = 0
            for arg_name, arg_value in ratios.items():
                if k_unet.startswith(arg_name):
                    if len(arg_name) > matched_arg_len:
                        scale_value = arg_value
                        matched_arg_len = len(arg_name)
            
            # スケーリングの適用
            # ComfyUIのadd_patchesは (patch, strength_patch, strength_model) を計算する
            # Output = W * strength_model + P * strength_patch
            # スケーリングを行うため: W_new = W * scale_value としたい
            # ここでは W * scale_value として実装
            if scale_value != 1.0:
                # kp[k] は元の重みパッチ情報
                # scale_value を適用したパッチを作成
                m.add_patches({k: kp[k]}, scale_value - 1.0, 1.0)
        
        return (m,)
    
class ModelScaleFlux2Klein:
    """
    FLUX2 Klein Modelの特定の層をスケーリングするノード。
    scale=1.0 で元のまま、scale=0.0 でゼロ、1.0以上で強調します。
    """

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"model": ("MODEL",)}

        # スケーリング用の引数設定（デフォルト1.0、範囲は0.0〜2.0に設定）
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})

        # 基本コンポーネント
        arg_dict["img_in."] = argument
        arg_dict["time_in."] = argument
        arg_dict["txt_in."] = argument

        # Double Blocks (0-4)
        for i in range(5):
            arg_dict["double_blocks.{}.".format(i)] = argument
            # さらに細かく制御したい場合
            arg_dict["double_blocks.{}.img_attn.".format(i)] = argument
            arg_dict["double_blocks.{}.img_mlp.".format(i)] = argument
            arg_dict["double_blocks.{}.txt_attn.".format(i)] = argument
            arg_dict["double_blocks.{}.txt_mlp.".format(i)] = argument

        # Double Stream Modulation
        arg_dict["double_stream_modulation_img."] = argument
        arg_dict["double_stream_modulation_txt."] = argument

        # Single Blocks (0-19)
        for i in range(20):
            arg_dict["single_blocks.{}.".format(i)] = argument

        # Single Stream Modulation
        arg_dict["single_stream_modulation."] = argument

        # Final Layer
        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"

    def scale(self, model, **kwargs):
        # モデルの複製
        m = model.clone()

        # スケーリング比率の辞書（'model'キー以外を抽出）
        ratios = {k: v for k, v in kwargs.items() if k != "model"}

        # モデルのパッチ可能なキー（重み）を取得
        kp = m.get_key_patches("diffusion_model.")

        # 全ての重みキーに対してスケーリングを適用
        for k in kp:
            scale_value = 1.0
            # diffusion_model. を除いた純粋なレイヤー名
            k_unet = k[len("diffusion_model."):]

            # 最も長く一致するプレフィックスを探すロジック
            matched_arg_len = 0
            for arg_name, arg_value in ratios.items():
                if k_unet.startswith(arg_name):
                    if len(arg_name) > matched_arg_len:
                        scale_value = arg_value
                        matched_arg_len = len(arg_name)

            # スケーリングの適用
            # scale_value != 1.0 のときのみパッチを追加
            if scale_value != 1.0:
                # kp[k] は (tensor,) のタプル
                weight_tensor = kp[k][0]
                # 元の重みに対して (scale - 1.0) 分をパッチとして追加することで乗算を実現
                # Output = W * strength_model + P * strength_patch
                # W_new = W * 1.0 + W * (scale - 1.0) = W * scale
                m.add_patches({k: (weight_tensor,)}, scale_value - 1.0, 1.0)

        return (m,)


class CLIPScaleDualSDXLBlock:
    """
    SDXL DualCLIP（CLIP-L + CLIP-G）の特定の層をスケーリングするノード
    scale=1.0 でそのまま、scale=0.0 で完全に抑制
    """

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "clip": ("CLIP",),
        }

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})

        # --- CLIP-L (12 layers) ---
        arg_dict["clip_l.embeddings"] = argument
        for i in range(12):
            arg_dict[f"clip_l.encoder.layers.{i}"] = argument
        arg_dict["clip_l.final_layer_norm"] = argument

        # --- CLIP-G (32 layers) ---
        arg_dict["clip_g.embeddings"] = argument
        for i in range(32):
            arg_dict[f"clip_g.encoder.layers.{i}"] = argument
        arg_dict["clip_g.final_layer_norm"] = argument
        arg_dict["clip_g.text_projection"] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"

    DESCRIPTION = "Scale specific layers of SDXL Dual CLIP (CLIP-L: 12 layers, CLIP-G: 32 layers). Scale=1.0 keeps original, Scale=0.0 zeroes out the layer."

    def scale(self, clip, **kwargs):
        # クローンを作成（モデルラッパーのみ、重みは共有）
        m = clip.clone()

        # モデルの全重み（State Dict）を取得
        # これにより "clip_l.transformer.text_model..." のような完全なキーが取得できます
        sd = m.get_sd()

        # 設定されたスケール値を辞書として整理
        scales_args = {k: v for k, v in kwargs.items() if k != "clip" and v != 1.0}

        # 何も変更がない場合はそのまま返す
        if not scales_args:
            return (m,)

        # スケールごとに適用するパッチをまとめるための辞書
        # { scale_value: { key: (weight_tensor,) } }
        patches_by_scale = {}

        for key, weight in sd.items():
            # 無視するキー
            if key.endswith(".position_ids") or key.endswith(".logit_scale"):
                continue

            # キーのマッチング処理
            # SDXLの内部キーは "clip_l.transformer.text_model.encoder..." のように長いため
            # ユーザー引数 ("clip_l.encoder...") とマッチするように正規化します
            normalized_key = key.replace(".transformer.text_model.", ".")
            normalized_key = normalized_key.replace(".text_model.", ".")  # 念のため

            target_scale = 1.0

            # 最も具体的にマッチする引数を探す
            # 例: "clip_l.encoder.layers.0" は "clip_l.encoder" よりも優先されるべき
            matched_arg_len = 0

            for arg_name, scale_val in scales_args.items():
                if normalized_key.startswith(arg_name):
                    # より長いキー名でのマッチを優先（より具体的であるため）
                    if len(arg_name) > matched_arg_len:
                        target_scale = scale_val
                        matched_arg_len = len(arg_name)

            # スケール変更が必要な場合
            if target_scale != 1.0:
                if target_scale not in patches_by_scale:
                    patches_by_scale[target_scale] = {}

                # パッチとして「元の重み」を登録
                patches_by_scale[target_scale][key] = (weight,)

        # パッチの適用
        # add_patches({key: (patch,)}, strength_patch, strength_model)
        # 計算式: Final = Model * strength_model + Patch * strength_patch
        # ここで Patch = Model なので
        # Final = Model * 1.0 + Model * (scale - 1.0)
        #       = Model * (1.0 + scale - 1.0)
        #       = Model * scale
        for s, patches in patches_by_scale.items():
            m.add_patches(patches, s - 1.0, 1.0)

        return (m,)


class CLIPScaleQwenBlock:
    """
    Qwen-2.5-VL-7B CLIPの特定の層をスケーリングするノード
    scale=1.0 でそのまま、scale=0.0 で完全に抑制
    """

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "clip": ("CLIP",),
        }

        # scale: 1.0 = そのまま, 0.0 = 完全に抑制（ゼロ化）
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})

        arg_dict["model.embed_tokens"] = argument
        arg_dict["visual.patch_embed"] = argument

        # Qwen2-VL-7B (Visual: 32 blocks)
        for i in range(32):
            arg_dict[f"visual.blocks.{i}"] = argument

        arg_dict["visual.merger"] = argument

        # Qwen2-VL-7B (LLM: 28 layers)
        for i in range(28):
            arg_dict[f"model.layers.{i}"] = argument

        arg_dict["model.norm"] = argument
        arg_dict["lm_head"] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"
    DESCRIPTION = "Scale specific layers of Qwen-2.5-VL-7B CLIP. Scale=1.0 keeps original, Scale=0.0 zeroes out the layer."

    def scale(self, clip, **kwargs):
        # モデルのクローンを作成
        m = clip.clone()

        # 設定されたスケール値を辞書として整理
        scales_args = {k: v for k, v in kwargs.items() if k != "clip" and v != 1.0}

        # 何も変更がない場合はそのまま返す
        if not scales_args:
            return (m,)

        # キーパッチを取得
        kp = clip.get_key_patches()

        # スケールごとにパッチをまとめる
        patches_by_scale = {}

        for key in kp:
            # 不要なキーをスキップ
            if key.endswith(".position_ids") or key.endswith(".logit_scale"):
                continue

            # キーの正規化
            # "transformer." プレフィックスを除去（ある場合）
            normalized_key = key
            if normalized_key.startswith("transformer."):
                normalized_key = normalized_key[len("transformer.") :]

            target_scale = 1.0
            matched_arg_len = 0

            # 引数名とキーの前方一致で判定（最長マッチ）
            for arg_name, scale_val in scales_args.items():
                if normalized_key.startswith(arg_name):
                    # より長くマッチする場合のみ更新
                    # "model.layers.1" と "model.layers.10" の誤爆を防ぐ
                    next_char_idx = len(arg_name)
                    if (
                        next_char_idx == len(normalized_key)
                        or normalized_key[next_char_idx] == "."
                    ):
                        if len(arg_name) > matched_arg_len:
                            target_scale = scale_val
                            matched_arg_len = len(arg_name)

            # スケール変更が必要な場合
            if target_scale != 1.0:
                if target_scale not in patches_by_scale:
                    patches_by_scale[target_scale] = {}
                patches_by_scale[target_scale][key] = kp[key]

        # パッチの適用
        # add_patches(patches, strength_patch, strength_model)
        # Final = Model * strength_model + Patch * strength_patch
        # 目標: Model * scale なので、Patch = Model として
        # scale = strength_model + strength_patch
        # strength_model = 0, strength_patch = scale とする
        for scale_val, patches in patches_by_scale.items():
            m.add_patches(patches, scale_val, 0.0)

        return (m,)


class CLIPSaveQwen:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "filename_prefix": ("STRING", {"default": "qwen_2.5_vl_merged"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "advanced/model_merging/model_specific"

    DESCRIPTION = "Saves Qwen-2.5-VL-7B CLIP models by stripping the internal 'qwen25_7b.transformer.' prefix."

    def save(self, clip, filename_prefix, prompt=None, extra_pnginfo=None):
        # メタデータの準備
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = {}
        if not args.disable_metadata:
            metadata["format"] = "pt"
            metadata["prompt"] = prompt_info
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        # モデルのState Dictを取得
        # ComfyUIはモデルをラップしているため、内部キーにアクセスします
        clip_sd = clip.get_sd()

        # 保存用の新しい辞書を作成
        output_sd = {}

        # 削除対象のプレフィックス
        # ログ に基づき "qwen25_7b.transformer." を削除対象とします
        prefix_to_strip = "qwen25_7b.transformer."

        for k, v in clip_sd.items():
            if k.startswith(prefix_to_strip):
                # プレフィックスを削除したキー名にする
                # 例: qwen25_7b.transformer.model.layers.0... -> model.layers.0...
                new_key = k[len(prefix_to_strip) :]
                output_sd[new_key] = v
            elif k.startswith("qwen25_7b."):
                # logit_scaleなどの例外処理
                new_key = k.replace("qwen25_7b.", "")
                output_sd[new_key] = v
            else:
                # プレフィックスがない場合はそのまま（通常はないはずですが念のため）
                output_sd[k] = v

        # ファイルパスの生成
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        # 保存実行
        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=metadata)

        return {}


class VAEMergeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae1": ("VAE",),
                "vae2": ("VAE",),
                "ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "merge"
    CATEGORY = "advanced/model_merging"

    def merge(self, vae1, vae2, ratio):
        vae1_sd = vae1.get_sd()
        vae2_sd = vae2.get_sd()

        merged_sd = {}
        for key in vae1_sd.keys():
            if key in vae2_sd:
                merged_sd[key] = (1.0 - ratio) * vae1_sd[key] + ratio * vae2_sd[key]
            else:
                merged_sd[key] = vae1_sd[key]

        merged_vae = comfy.sd.VAE(sd=merged_sd)
        return (merged_vae,)


class VAEMergeSubtract:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae1": ("VAE",),
                "vae2": ("VAE",),
                "multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "merge"
    CATEGORY = "advanced/model_merging"

    def merge(self, vae1, vae2, multiplier):
        vae1_sd = vae1.get_sd()
        vae2_sd = vae2.get_sd()

        merged_sd = {}
        for key in vae1_sd.keys():
            if key in vae2_sd:
                merged_sd[key] = vae1_sd[key] - multiplier * vae2_sd[key]
            else:
                merged_sd[key] = vae1_sd[key]

        merged_vae = comfy.sd.VAE(sd=merged_sd)
        return (merged_vae,)


class VAEMergeAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae1": ("VAE",),
                "vae2": ("VAE",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "merge"
    CATEGORY = "advanced/model_merging"

    def merge(self, vae1, vae2):
        vae1_sd = vae1.get_sd()
        vae2_sd = vae2.get_sd()

        merged_sd = {}
        for key in vae1_sd.keys():
            if key in vae2_sd:
                merged_sd[key] = vae1_sd[key] + vae2_sd[key]
            else:
                merged_sd[key] = vae1_sd[key]

        merged_vae = comfy.sd.VAE(sd=merged_sd)
        return (merged_vae,)


class VAEScaleSDXLBlock:
    """
    SDXL VAEの特定の層をスケーリングするノード
    scale=1.0 でそのまま、scale=0.0 で完全に抑制
    """

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "vae": ("VAE",),
        }

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})

        # --- Quantization layers ---
        arg_dict["quant_conv"] = argument
        arg_dict["post_quant_conv"] = argument

        # --- Encoder ---
        arg_dict["encoder.conv_in"] = argument
        # Encoder down_blocks (0-3)
        for i in range(4):
            arg_dict[f"encoder.down_blocks.{i}."] = argument
        # Encoder mid_block
        arg_dict["encoder.mid_block.attentions.0."] = argument
        arg_dict["encoder.mid_block.resnets.0."] = argument
        arg_dict["encoder.mid_block.resnets.1."] = argument
        arg_dict["encoder.conv_norm_out"] = argument
        arg_dict["encoder.conv_out"] = argument

        # --- Decoder ---
        arg_dict["decoder.conv_in"] = argument
        # Decoder mid_block
        arg_dict["decoder.mid_block.attentions.0."] = argument
        arg_dict["decoder.mid_block.resnets.0."] = argument
        arg_dict["decoder.mid_block.resnets.1."] = argument
        # Decoder up_blocks (0-3)
        for i in range(4):
            arg_dict[f"decoder.up_blocks.{i}."] = argument
        arg_dict["decoder.conv_norm_out"] = argument
        arg_dict["decoder.conv_out"] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"

    DESCRIPTION = "Scale specific layers of SDXL VAE. Scale=1.0 keeps original, Scale=0.0 zeroes out the layer."

    def scale(self, vae, **kwargs):
        import torch

        ratios = {k: v for k, v in kwargs.items() if k != "vae"}

        sd = vae.get_sd()
        new_sd = {}

        for k, v in sd.items():
            scale = 1.0
            matched_arg_len = 0

            for arg_name, arg_value in ratios.items():
                if k.startswith(arg_name):
                    if len(arg_name) > matched_arg_len:
                        scale = arg_value
                        matched_arg_len = len(arg_name)

            if scale != 1.0:
                new_sd[k] = v * scale
            else:
                new_sd[k] = v

        new_vae = comfy.sd.VAE(sd=new_sd)

        return (new_vae,)


class VAEMergeSDXLBlock:
    """
    2つのSDXL VAEをブロック単位でマージするノード
    ratio=1.0 で vae2 を使用、ratio=0.0 で vae1 を使用
    """

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "vae1": ("VAE",),
            "vae2": ("VAE",),
        }

        argument = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})

        # --- Quantization layers ---
        arg_dict["quant_conv"] = argument
        arg_dict["post_quant_conv"] = argument

        # --- Encoder ---
        arg_dict["encoder.conv_in"] = argument
        for i in range(4):
            arg_dict[f"encoder.down_blocks.{i}."] = argument
        arg_dict["encoder.mid_block.attentions.0."] = argument
        arg_dict["encoder.mid_block.resnets.0."] = argument
        arg_dict["encoder.mid_block.resnets.1."] = argument
        arg_dict["encoder.conv_norm_out"] = argument
        arg_dict["encoder.conv_out"] = argument

        # --- Decoder ---
        arg_dict["decoder.conv_in"] = argument
        arg_dict["decoder.mid_block.attentions.0."] = argument
        arg_dict["decoder.mid_block.resnets.0."] = argument
        arg_dict["decoder.mid_block.resnets.1."] = argument
        for i in range(4):
            arg_dict[f"decoder.up_blocks.{i}."] = argument
        arg_dict["decoder.conv_norm_out"] = argument
        arg_dict["decoder.conv_out"] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "merge"
    CATEGORY = "advanced/model_merging/model_specific"

    DESCRIPTION = (
        "Block-wise merging for SDXL VAE. Ratio=0.0 keeps vae1, Ratio=1.0 uses vae2."
    )

    def merge(self, vae1, vae2, **kwargs):
        import torch

        ratios = {k: v for k, v in kwargs.items() if k not in ["vae1", "vae2"]}

        sd1 = vae1.get_sd()
        sd2 = vae2.get_sd()

        new_sd = {}

        for k in sd1.keys():
            if k not in sd2:
                new_sd[k] = sd1[k]
                continue

            ratio = 0.5
            matched_arg_len = 0

            for arg_name, arg_value in ratios.items():
                if k.startswith(arg_name):
                    if len(arg_name) > matched_arg_len:
                        ratio = arg_value
                        matched_arg_len = len(arg_name)

            new_sd[k] = sd1[k] * (1.0 - ratio) + sd2[k] * ratio

        for k in sd2.keys():
            if k not in new_sd:
                new_sd[k] = sd2[k]

        new_vae = comfy.sd.VAE(sd=new_sd)

        return (new_vae,)


class VAEScaleFluxBlock:
    """
    FLUX1 VAEの特定の層をスケーリングするノード
    scale=1.0 でそのまま、scale=0.0 で完全に抑制
    """

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "vae": ("VAE",),
        }

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})

        # --- Encoder ---
        arg_dict["encoder.conv_in"] = argument
        arg_dict["encoder.conv_out"] = argument
        arg_dict["encoder.norm_out"] = argument

        # Encoder down blocks (0-3)
        for i in range(4):
            arg_dict[f"encoder.down.{i}."] = argument
            # Each down block has multiple sub-blocks (0-2)
            for j in range(3):
                arg_dict[f"encoder.down.{i}.block.{j}."] = argument
            # Downsample layers (0-2 have downsample)
            if i < 3:
                arg_dict[f"encoder.down.{i}.downsample."] = argument

        # Encoder middle blocks
        arg_dict["encoder.mid."] = argument
        arg_dict["encoder.mid.block_1."] = argument
        arg_dict["encoder.mid.block_2."] = argument
        arg_dict["encoder.mid.attn_1."] = argument

        # --- Decoder ---
        arg_dict["decoder.conv_in"] = argument
        arg_dict["decoder.conv_out"] = argument
        arg_dict["decoder.norm_out"] = argument

        # Decoder up blocks (0-3)
        for i in range(4):
            arg_dict[f"decoder.up.{i}."] = argument
            # Each up block has multiple sub-blocks (0-2)
            for j in range(3):
                arg_dict[f"decoder.up.{i}.block.{j}."] = argument
            # Upsample layers (1-3 have upsample)
            if i > 0:
                arg_dict[f"decoder.up.{i}.upsample."] = argument

        # Decoder middle blocks
        arg_dict["decoder.mid."] = argument
        arg_dict["decoder.mid.block_1."] = argument
        arg_dict["decoder.mid.block_2."] = argument
        arg_dict["decoder.mid.attn_1."] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"

    DESCRIPTION = "Scale specific layers of FLUX1 VAE. Scale=1.0 keeps original, Scale=0.0 zeroes out the layer."

    def scale(self, vae, **kwargs):
        ratios = {k: v for k, v in kwargs.items() if k != "vae"}

        # VAEのstate_dictを取得
        sd = vae.get_sd()
        new_sd = {}

        for k, v in sd.items():
            scale = 1.0
            matched_arg_len = 0

            # 最も長くマッチするプレフィックスを見つける
            for arg_name, arg_value in ratios.items():
                if k.startswith(arg_name):
                    if len(arg_name) > matched_arg_len:
                        scale = arg_value
                        matched_arg_len = len(arg_name)

            # スケーリングを適用
            if scale != 1.0:
                new_sd[k] = v * scale
            else:
                new_sd[k] = v

        # 新しいVAEを作成
        new_vae = comfy.sd.VAE(sd=new_sd)

        return (new_vae,)


class VAEScaleFlux2Block:
    """
    FLUX2 VAEの特定の層をスケーリングするノード
    scale=1.0 でそのまま、scale=0.0 で完全に抑制
    """

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "vae": ("VAE",),
        }

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})

        # --- Batch Normalization ---
        arg_dict["bn."] = argument

        # --- Encoder ---
        arg_dict["encoder.conv_in"] = argument
        arg_dict["encoder.conv_out"] = argument
        arg_dict["encoder.conv_norm_out"] = argument

        # Encoder down blocks (0-3)
        for i in range(4):
            arg_dict[f"encoder.down_blocks.{i}."] = argument
            # Each down block has resnets (0-1)
            for j in range(2):
                arg_dict[f"encoder.down_blocks.{i}.resnets.{j}."] = argument
                arg_dict[f"encoder.down_blocks.{i}.resnets.{j}.conv1"] = argument
                arg_dict[f"encoder.down_blocks.{i}.resnets.{j}.conv2"] = argument
                arg_dict[f"encoder.down_blocks.{i}.resnets.{j}.norm1"] = argument
                arg_dict[f"encoder.down_blocks.{i}.resnets.{j}.norm2"] = argument
                # conv_shortcut (only in some blocks)
                arg_dict[f"encoder.down_blocks.{i}.resnets.{j}.conv_shortcut"] = argument
            
            # Downsamplers (0-2 have downsample)
            if i < 3:
                arg_dict[f"encoder.down_blocks.{i}.downsamplers."] = argument
                arg_dict[f"encoder.down_blocks.{i}.downsamplers.0.conv"] = argument

        # Encoder middle block
        arg_dict["encoder.mid_block."] = argument
        arg_dict["encoder.mid_block.resnets.0."] = argument
        arg_dict["encoder.mid_block.resnets.0.conv1"] = argument
        arg_dict["encoder.mid_block.resnets.0.conv2"] = argument
        arg_dict["encoder.mid_block.resnets.0.norm1"] = argument
        arg_dict["encoder.mid_block.resnets.0.norm2"] = argument
        arg_dict["encoder.mid_block.resnets.1."] = argument
        arg_dict["encoder.mid_block.resnets.1.conv1"] = argument
        arg_dict["encoder.mid_block.resnets.1.conv2"] = argument
        arg_dict["encoder.mid_block.resnets.1.norm1"] = argument
        arg_dict["encoder.mid_block.resnets.1.norm2"] = argument
        arg_dict["encoder.mid_block.attentions.0."] = argument
        arg_dict["encoder.mid_block.attentions.0.group_norm"] = argument
        arg_dict["encoder.mid_block.attentions.0.to_q"] = argument
        arg_dict["encoder.mid_block.attentions.0.to_k"] = argument
        arg_dict["encoder.mid_block.attentions.0.to_v"] = argument
        arg_dict["encoder.mid_block.attentions.0.to_out"] = argument

        # --- Quantization ---
        arg_dict["quant_conv"] = argument
        arg_dict["post_quant_conv"] = argument

        # --- Decoder ---
        arg_dict["decoder.conv_in"] = argument
        arg_dict["decoder.conv_out"] = argument
        arg_dict["decoder.conv_norm_out"] = argument

        # Decoder middle block
        arg_dict["decoder.mid_block."] = argument
        arg_dict["decoder.mid_block.resnets.0."] = argument
        arg_dict["decoder.mid_block.resnets.0.conv1"] = argument
        arg_dict["decoder.mid_block.resnets.0.conv2"] = argument
        arg_dict["decoder.mid_block.resnets.0.norm1"] = argument
        arg_dict["decoder.mid_block.resnets.0.norm2"] = argument
        arg_dict["decoder.mid_block.resnets.1."] = argument
        arg_dict["decoder.mid_block.resnets.1.conv1"] = argument
        arg_dict["decoder.mid_block.resnets.1.conv2"] = argument
        arg_dict["decoder.mid_block.resnets.1.norm1"] = argument
        arg_dict["decoder.mid_block.resnets.1.norm2"] = argument
        arg_dict["decoder.mid_block.attentions.0."] = argument
        arg_dict["decoder.mid_block.attentions.0.group_norm"] = argument
        arg_dict["decoder.mid_block.attentions.0.to_q"] = argument
        arg_dict["decoder.mid_block.attentions.0.to_k"] = argument
        arg_dict["decoder.mid_block.attentions.0.to_v"] = argument
        arg_dict["decoder.mid_block.attentions.0.to_out"] = argument

        # Decoder up blocks (0-3)
        for i in range(4):
            arg_dict[f"decoder.up_blocks.{i}."] = argument
            # Each up block has resnets (0-2)
            for j in range(3):
                arg_dict[f"decoder.up_blocks.{i}.resnets.{j}."] = argument
                arg_dict[f"decoder.up_blocks.{i}.resnets.{j}.conv1"] = argument
                arg_dict[f"decoder.up_blocks.{i}.resnets.{j}.conv2"] = argument
                arg_dict[f"decoder.up_blocks.{i}.resnets.{j}.norm1"] = argument
                arg_dict[f"decoder.up_blocks.{i}.resnets.{j}.norm2"] = argument
                # conv_shortcut (only in some blocks)
                arg_dict[f"decoder.up_blocks.{i}.resnets.{j}.conv_shortcut"] = argument
            
            # Upsamplers (0-2 have upsample)
            if i < 3:
                arg_dict[f"decoder.up_blocks.{i}.upsamplers."] = argument
                arg_dict[f"decoder.up_blocks.{i}.upsamplers.0.conv"] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"

    DESCRIPTION = "Scale specific layers of FLUX2 VAE. Scale=1.0 keeps original, Scale=0.0 zeroes out the layer."

    def scale(self, vae, **kwargs):
        import comfy.sd
        
        ratios = {k: v for k, v in kwargs.items() if k != "vae"}

        # VAEのstate_dictを取得
        sd = vae.get_sd()
        new_sd = {}

        for k, v in sd.items():
            scale = 1.0
            matched_arg_len = 0

            # 最も長くマッチするプレフィックスを見つける
            for arg_name, arg_value in ratios.items():
                if k.startswith(arg_name):
                    if len(arg_name) > matched_arg_len:
                        scale = arg_value
                        matched_arg_len = len(arg_name)

            # スケーリングを適用
            if scale != 1.0:
                new_sd[k] = v * scale
            else:
                new_sd[k] = v

        # 新しいVAEを作成
        new_vae = comfy.sd.VAE(sd=new_sd)

        return (new_vae,)


class VAEScaleQwenBlock:
    """
    Qwen Image VAEの特定の層をスケーリングするノード
    scale=1.0 でそのまま、scale=0.0 で完全に抑制
    """

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "vae": ("VAE",),
        }

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})

        # --- Global conv layers ---
        arg_dict["conv1"] = argument
        arg_dict["conv2"] = argument

        # --- Encoder ---
        arg_dict["encoder.conv1"] = argument
        # Encoder downsamples (0-10)
        for i in range(11):
            arg_dict[f"encoder.downsamples.{i}."] = argument
        # Encoder middle (0-2)
        for i in range(3):
            arg_dict[f"encoder.middle.{i}."] = argument
        arg_dict["encoder.head"] = argument

        # --- Decoder ---
        arg_dict["decoder.conv1"] = argument
        # Decoder middle (0-2)
        for i in range(3):
            arg_dict[f"decoder.middle.{i}."] = argument
        # Decoder upsamples (0-14)
        for i in range(15):
            arg_dict[f"decoder.upsamples.{i}."] = argument
        arg_dict["decoder.head"] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"

    DESCRIPTION = "Scale specific layers of Qwen Image VAE. Scale=1.0 keeps original, Scale=0.0 zeroes out the layer."

    def scale(self, vae, **kwargs):
        import torch
        import copy

        # VAEをクローン
        # ComfyUIのVAEオブジェクトは直接cloneメソッドを持たない場合があるので
        # 内部のstate_dictを操作する

        ratios = {k: v for k, v in kwargs.items() if k != "vae"}

        # VAEの内部モデルにアクセス
        # ComfyUIのVAEラッパーを通じてstate_dictを取得・設定
        device = vae.device
        sd = vae.get_sd()

        new_sd = {}

        for k, v in sd.items():
            scale = 1.0
            matched_arg_len = 0

            for arg_name, arg_value in ratios.items():
                if k.startswith(arg_name):
                    if len(arg_name) > matched_arg_len:
                        scale = arg_value
                        matched_arg_len = len(arg_name)

            if scale != 1.0:
                new_sd[k] = v * scale
            else:
                new_sd[k] = v

        # 新しいVAEを作成
        new_vae = comfy.sd.VAE(sd=new_sd)

        return (new_vae,)


NODE_CLASS_MAPPINGS = {
    "HDR Effects with LAB Adjust": HDREffectsLabAdjust,
    "SaveImageWithPrompt": SaveImageWithPrompt,
    "ModelScaleSDXL": ModelScaleSDXL,
    "ModelMergeHiDream": ModelMergeHiDream,
    "ModelScaleHiDream": ModelScaleHiDream,
    "ModelScaleQwenImage": ModelScaleQwenImage,
    "ModelMergeZImage": ModelMergeZImage,
    "ModelScaleZImage": ModelScaleZImage,
    "ModelScaleFlux2Klein": ModelScaleFlux2Klein,
    "CLIPScaleDualSDXLBlock": CLIPScaleDualSDXLBlock,
    "CLIPScaleQwenBlock": CLIPScaleQwenBlock,
    "CLIPSaveQwen": CLIPSaveQwen,
    "VAEMergeSimple": VAEMergeSimple,
    "VAEMergeSubtract": VAEMergeSubtract,
    "VAEMergeAdd": VAEMergeAdd,
    "VAEScaleSDXLBlock": VAEScaleSDXLBlock,
    "VAEMergeSDXLBlock": VAEMergeSDXLBlock,
    "VAEScaleFluxBlock": VAEScaleFluxBlock,
    "VAEScaleFlux2Block": VAEScaleFlux2Block,
    "VAEScaleQwenBlock": VAEScaleQwenBlock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HDREffectsLabAdjust": "HDR Effects with LAB Adjusts",
    "SaveImageWithPrompt": "Save Image With Prompt",
    "ModelScaleSDXL": "Model Scale SDXL",
    "ModelMergeHiDream": "Model Merge HiDream",
    "ModelScaleHiDream": "Model Scale HiDream",
    "ModelScaleQwenImage": "Model Scale Qwen Image",
    "ModelMergeZImage": "Model Merge Z-Image",
    "ModelScaleZImage": "Model Scale Z-Image",
    "ModelScaleFlux2Klein": "Model Scale Flux2 Klein",
    "CLIPScaleDualSDXLBlock": "CLIP Scale Dual SDXL Block",
    "CLIPScaleQwenBlock": "CLIP Scale Qwen Block",
    "CLIPSaveQwen": "CLIP Save Qwen (Fix Prefix)",
    "VAEMergeSimple": "VAE Merge Simple",
    "VAEMergeSubtract": "VAE Merge Subtract",
    "VAEMergeAdd": "VAE Merge Add",
    "VAEScaleSDXLBlock": "VAE Scale SDXL Block",
    "VAEMergeSDXLBlock": "VAE Merge SDXL Block",
    "VAEScaleFluxBlock": "VAE Scale FLUX Block",
    "VAEScaleFlux2Block": "VAE Scale FLUX2 Block",
    "VAEScaleQwenBlock": "VAE Scale Qwen Block",
}
