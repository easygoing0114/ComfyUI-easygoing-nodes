import os
import json
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageCms
from PIL.PngImagePlugin import PngInfo
import folder_paths
import comfy.sd
from comfy.cli_args import args
import comfy.clip_vision
import comfy.model_management

# LAB color space profiles
sRGB_profile = ImageCms.createProfile("sRGB")
Lab_profile = ImageCms.createProfile("LAB")

# Helper functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def adjust_shadows_non_linear(luminance, shadow_intensity, max_shadow_adjustment=1.5):
    lum_array = np.array(luminance, dtype=np.float32) / 255.0
    shadows = lum_array ** (1 / (1 + shadow_intensity * max_shadow_adjustment))
    return np.clip(shadows * 255, 0, 255).astype(np.uint8)

def adjust_highlights_non_linear(luminance, highlight_intensity, max_highlight_adjustment=1.5):
    lum_array = np.array(luminance, dtype=np.float32) / 255.0
    highlights = 1 - (1 - lum_array) ** (1 + highlight_intensity * max_highlight_adjustment)
    return np.clip(highlights * 255, 0, 255).astype(np.uint8)

def merge_adjustments_with_blend_modes(luminance, shadows, highlights, hdr_intensity, shadow_intensity, highlight_intensity):
    base = np.array(luminance, dtype=np.float32)
    scaled_shadow_intensity = shadow_intensity ** 2 * hdr_intensity
    scaled_highlight_intensity = highlight_intensity ** 2 * hdr_intensity
    shadow_mask = np.clip((1 - (base / 255)) ** 2, 0, 1)
    highlight_mask = np.clip((base / 255) ** 2, 0, 1)
    adjusted_shadows = np.clip(base * (1 - shadow_mask * scaled_shadow_intensity), 0, 255)
    adjusted_highlights = np.clip(base + (255 - base) * highlight_mask * scaled_highlight_intensity, 0, 255)
    adjusted_luminance = np.clip(adjusted_shadows + adjusted_highlights - base, 0, 255)
    final_luminance = np.clip(base * (1 - hdr_intensity) + adjusted_luminance * hdr_intensity, 0, 255).astype(np.uint8)
    return Image.fromarray(final_luminance)

def apply_gamma_correction(lum_array, gamma):
    if gamma == 0:
        return np.clip(lum_array, 0, 255).astype(np.uint8)
    epsilon = 1e-7
    gamma_corrected = 1 / (1.1 - gamma)
    adjusted = 255 * ((lum_array / 255) ** gamma_corrected)
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def adjust_ab_exponential(ab_array, adjustment_intensity, center=128, max_adjustment=2.0):
    """
    AB色相チャンネルに指数関数的調整を適用
    
    Args:
        ab_array: A または B チャンネルの配列 (0-255)
        adjustment_intensity: 調整強度 (-1.0 to 1.0)
        center: 中性点（通常128）
        max_adjustment: 最大調整係数
    """
    if adjustment_intensity == 0:
        return ab_array
    
    # 0-1範囲に正規化し、中性点を0基準にシフト
    normalized = (ab_array.astype(np.float32) - center) / 128.0
    
    if adjustment_intensity > 0:
        # 正の調整: 色を強調（外側に拡張）
        gamma = 1 / (1 + adjustment_intensity * max_adjustment)
        adjusted = np.sign(normalized) * (np.abs(normalized) ** gamma)
    else:
        # 負の調整: 色を抑制（中性点に収束）
        gamma = 1 + (-adjustment_intensity) * max_adjustment
        adjusted = np.sign(normalized) * (np.abs(normalized) ** gamma)
    
    # 元の範囲に戻す
    result = adjusted * 128.0 + center
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_to_batch(func):
    def wrapper(self, image, *args, **kwargs):
        images = [func(self, img, *args, **kwargs) for img in image]
        return (torch.cat(images, dim=0),)
    return wrapper

# Custom node: HDREffectsLabAdjusts with Exponential AB Processing
class HDREffectsLabAdjust:
    DESCRIPTION = "Apply HDR tone-mapping with control over shadows, highlights, gamma, contrast, color boost, and exponential LAB A/B channel adjustments. Use ab_strength to control color adjustment intensity (0.5=gentle, 2.0=standard, 5.0=dramatic)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'hdr_intensity': ('FLOAT', {'default': 0.8, 'min': 0.0, 'max': 5.0, 'step': 0.01}),
                'shadow_intensity': ('FLOAT', {'default': 0.8, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'highlight_intensity': ('FLOAT', {'default': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'gamma_intensity': ('FLOAT', {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'ab_strength': ('FLOAT', {'default': 1.0, 'min': 0.5, 'max': 5.0, 'step': 0.1}),
                'a_adjustment': ('FLOAT', {'default': 0.15, 'min': -1.0, 'max': 1.0, 'step': 0.01}),
                'b_adjustment': ('FLOAT', {'default': -0.3, 'min': -1.0, 'max': 1.0, 'step': 0.01}),
                'contrast': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'enhance_color': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
            }
        }

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('result_img',)
    FUNCTION = 'apply_hdr2'
    CATEGORY = 'SuperBeastsAI/Image'

    @apply_to_batch
    def apply_hdr2(self, image, hdr_intensity=0.8, shadow_intensity=0.8, highlight_intensity=0.1, gamma_intensity=0.2,
                   ab_strength=1.0, a_adjustment=0.15, b_adjustment=-0.3,
                   contrast=0, enhance_color=0):
        img = tensor2pil(image)
        
        # Convert to LAB
        img_lab = ImageCms.profileToProfile(img, sRGB_profile, Lab_profile, outputMode='LAB')
        luminance, a, b = img_lab.split()
        
        # Convert to NumPy arrays
        lum_array = np.array(luminance, dtype=np.float32)
        a_array = np.array(a, dtype=np.uint8)
        b_array = np.array(b, dtype=np.uint8)

        # Apply exponential adjustments to A and B channels
        if a_adjustment != 0.0:
            a_array = adjust_ab_exponential(a_array, a_adjustment, max_adjustment=ab_strength)
        if b_adjustment != 0.0:
            b_array = adjust_ab_exponential(b_array, b_adjustment, max_adjustment=ab_strength)

        a_adjusted = Image.fromarray(a_array)
        b_adjusted = Image.fromarray(b_array)

        # Apply HDR adjustments to luminance
        shadows_adjusted = adjust_shadows_non_linear(luminance, shadow_intensity)
        highlights_adjusted = adjust_highlights_non_linear(luminance, highlight_intensity)
        merged_adjustments = merge_adjustments_with_blend_modes(lum_array, shadows_adjusted, highlights_adjusted, 
                                                               hdr_intensity, shadow_intensity, highlight_intensity)

        # Apply gamma correction
        gamma_corrected = apply_gamma_correction(np.array(merged_adjustments), gamma_intensity)
        gamma_corrected = Image.fromarray(gamma_corrected).resize(a.size)

        # Merge LAB channels
        adjusted_lab = Image.merge('LAB', (gamma_corrected, a_adjusted, b_adjusted))

        # Convert back to RGB
        img_adjusted = ImageCms.profileToProfile(adjusted_lab, Lab_profile, sRGB_profile, outputMode='RGB')
        
        # Enhance contrast and color
        contrast_adjusted = ImageEnhance.Contrast(img_adjusted).enhance(1 + contrast)
        color_adjusted = ImageEnhance.Color(contrast_adjusted).enhance(1 + enhance_color * 0.2)
        
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
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "positive_prompt": ("STRING", {"default": "", "tooltip": "The positive prompt to embed in the image metadata."}),
                "negative_prompt": ("STRING", {"default": "", "tooltip": "The negative prompt to embed in the image metadata."}),
                "caption": ("STRING", {"default": "", "tooltip": "The caption to embed in the image metadata."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory with positive and negative prompts and caption in metadata."

    def save_images(self, images, filename_prefix="ComfyUI", positive_prompt="", negative_prompt="", caption="", prompt=None, extra_pnginfo=None):
        # Truncate filename_prefix to 200 characters if it exceeds that length
        if len(filename_prefix) > 180:
            filename_prefix = filename_prefix[:180]
            
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
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
            file = f"{filename_with_batch_num}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        
        return {"ui": {"images": results}}

class QuadrupleCLIPLoaderSetDevice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name3": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name4": (folder_paths.get_filename_list("text_encoders"), ),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "[Recipes]\n\nhidream: long clip-l, long clip-g, t5xxl, llama_8b_3.1_instruct"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, device="default"):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        clip_path4 = folder_paths.get_full_path_or_raise("text_encoders", clip_name4)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2, clip_path3, clip_path4],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options
        )
        return (clip,)
    
class TripleCLIPLoaderSetDevice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name3": (folder_paths.get_filename_list("text_encoders"), ),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "[Recipes]\n\nsd3: clip-l, clip-g, t5"
    
    def load_clip(self, clip_name1, clip_name2, clip_name3, device="default"):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2, clip_path3],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options
        )
        return (clip,)
    
class CLIPVisionLoaderSetDevice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip_vision"), ),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            }
        }
    
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "Loads a CLIP vision model and sets the device (default or CPU)."

    def load_clip(self, clip_name, device="default"):
        # Get the full path of the CLIP vision model
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        
        # Set device configuration
        if device == "cpu":
            load_device = torch.device("cpu")
            offload_device = torch.device("cpu")
        else:
            load_device = comfy.model_management.text_encoder_device()
            offload_device = comfy.model_management.text_encoder_offload_device()
        
        # Load the CLIP vision model
        clip_vision = comfy.clip_vision.load(clip_path)
        if clip_vision is None:
            raise RuntimeError("Error: CLIP vision file is invalid and does not contain a valid vision model.")
        
        # Update ModelPatcher device settings
        clip_vision.patcher.load_device = load_device
        clip_vision.patcher.offload_device = offload_device
        
        # Ensure the model is loaded to the specified device
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        
        # Log the device used for loading
        print(f"CLIP vision model loaded to {load_device}")
        return (clip_vision,)


NODE_CLASS_MAPPINGS = {
    "HDR Effects with LAB Adjust": HDREffectsLabAdjust,
    "SaveImageWithPrompt": SaveImageWithPrompt,
    "QuadrupleCLIPLoaderSetDevice": QuadrupleCLIPLoaderSetDevice,
    "TripleCLIPLoaderSetDevice": TripleCLIPLoaderSetDevice,
    "CLIPVisionLoaderSetDevice": CLIPVisionLoaderSetDevice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'HDREffectsLabAdjust': 'HDR Effects with LAB Adjusts',
    "SaveImageWithPrompt": "Save Image With Prompt",
    "QuadrupleCLIPLoaderSetDevice": "Quadruple CLIP Loader (Set Device)",
    "TripleCLIPLoaderSetDevice": "Triple CLIP Loader (Set Device)",
    "CLIPVisionLoaderSetDevice": "Load CLIP Vision (Set Device)",    
}
