import os
import json
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args


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
        # ファイル名プレフィックスを200文字以内に制限
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
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if positive_prompt:
                    metadata.add_text("positive_prompt", json.dumps(positive_prompt))
                if negative_prompt:
                    metadata.add_text("negative_prompt", json.dumps(negative_prompt))
                if caption:
                    metadata.add_text("caption", json.dumps(caption))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

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


# ---- ノード登録用マッピング ----

NODE_CLASS_MAPPINGS = {
    "SaveImageWithPrompt": SaveImageWithPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageWithPrompt": "Save Image With Prompt",
}
