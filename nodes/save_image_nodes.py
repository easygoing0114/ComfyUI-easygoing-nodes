import json
import re

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io

# ------------------------------------------------------------------------------
# Node: Save Image With Prompt
# ------------------------------------------------------------------------------

_MAX_PREFIX_LEN = 180
_COMPRESS_LEVEL = 4


class SaveImageWithPrompt(io.ComfyNode):
    """Saves images with positive/additional/negative prompt, caption, and seed
    embedded as PNG metadata, in addition to the standard workflow metadata."""

    NODE_ID_LEGACY = "SaveImageWithPrompt"
    # Widget-only inputs, in the order used by the V1 INPUT_TYPES (excludes the
    # "images" socket input). Required for positional widget-value migration.
    NODE_ID_INPUT_ORDER = (
        "filename_prefix",
        "positive_prompt",
        "additional_prompt",
        "negative_prompt",
        "caption",
        "seed",
        "numbers",
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_SaveImageWithPrompt",
            display_name="Save Image With Prompt",
            category="image",
            description="Saves images to your ComfyUI output directory with positive (x2) and "
                        "negative prompts, caption, and seed in metadata.",
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.String.Input("positive_prompt", default=""),
                io.String.Input("additional_prompt", default=""),
                io.String.Input("negative_prompt", default=""),
                io.String.Input("caption", default=""),
                io.String.Input("seed", default=""),
                io.Boolean.Input(
                    "numbers",
                    default=True,
                    label_on="Include Numbers",
                    label_off="No Numbers",
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(
        cls,
        images,
        filename_prefix: str = "ComfyUI",
        positive_prompt: str = "",
        additional_prompt: str = "",
        negative_prompt: str = "",
        caption: str = "",
        seed: str = "",
        numbers: bool = True,
    ) -> io.NodeOutput:
        # Sanitize prefix: cap length, collapse whitespace.
        filename_prefix = re.sub(r'\s+', '_', filename_prefix[:_MAX_PREFIX_LEN])

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
            )
        )

        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo

        text_fields = {
            "positive_prompt": positive_prompt,
            "additional_prompt": additional_prompt,
            "negative_prompt": negative_prompt,
            "caption": caption,
            "seed": seed,
        }

        results = []
        for batch_number, image in enumerate(images):
            img_array = (255.0 * image.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                for key, value in text_fields.items():
                    if value:
                        metadata.add_text(key, json.dumps(value))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = (
                f"{filename_with_batch_num}_{counter:05}_.png"
                if numbers
                else f"{filename_with_batch_num}.png"
            )

            img.save(
                f"{full_output_folder}/{file}",
                pnginfo=metadata,
                compress_level=_COMPRESS_LEVEL,
            )
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1

        return io.NodeOutput(ui={"images": results})


NODE_LIST = [
    SaveImageWithPrompt,
]