<div align="center">
<img width="705" height="500" alt="ComfyUI Easygoing Nodes thumbnail" src="Images/thumbnail image.png">
</div>

# ComfyUI-easygoing-nodes

Enhanced Text Encoder processing for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), featuring custom nodes for HDR effects, image saving with prompt metadata, and model merging.

## ✨ Features

### 🔧 Enhanced Text Encoder Modules

Automatically replaces ComfyUI's built-in Text Encoder modules with enhanced versions that include:

- **CLIP-G Improvements**: [Enhanced attention mask support and improved tokenization](https://note.com/gentle_murre488/n/n12f2ecce1e00)

These replacements can be toggled on or off individually via the ComfyUI settings menu.

### 🌈 HDR Effects with LAB Adjust

<img width="320" height="374" alt="HDR Effects LAB Adjust node" src="Images/HDREffectsLabAdjust node.png">

**Example (Left: Original | Right: HDR Processing)**

<table>
  <tr>
    <td><img width="353" height="250" alt="Before HDR" src="Images/before HDR.png"></td>
    <td><img width="353" height="250" alt="After HDR" src="Images/after HDR.png"></td>
  </tr>
</table>

Advanced tone-mapping with control over shadows, highlights, gamma, contrast, color boost, and LAB channel adjustments.
💡 This node is based on the HDR processing from [ComfyUI-SuperBeasts](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts) with additional color adjustment features.

### 🧬 Model Merging

**ModelMergeHiDream**

<img width="180" height="650" alt="ModelMergeHiDream node" src="Images/ModelMergeHiDream node.png">

Performs hierarchical merging of HiDream models, enabling advanced model blending while preserving structural integrity.

### 💾 Save Image With Prompt

<img width="240" height="356" alt="Save Image With Prompt node" src="Images/SaveImageWithPrompt node.png">

Save images with positive/negative prompts and captions embedded directly into the PNG metadata.

## 🔥 Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/easygoing0114/ComfyUI-easygoing-nodes.git
```

2. For enhanced CLIP/Text Encoder functionality, ensure the modified modules are in place:
   - The enhanced modules are located in `modified_modules/` directory
   - `sdxl_clip.py` - Enhanced SDXL CLIP implementation

3. Restart ComfyUI. The new nodes should now appear in the node search.

## 🔍 Verification

When ComfyUI starts with this custom node, you should see messages like:
```
EasygoingNodes settings loaded: {'enable_sdxl_clip': True}
✓ Applied module replacements: sdxl_clip
```

If you don't see these messages, check that the `modified_modules/` directory contains the necessary files.

## 🙏 Credits

- Enhanced SDXL CLIP module (`sdxl_clip.py`) by [Shiba-2-shiba](https://github.com/Shiba-2-shiba)
- HDR Effects based on [ComfyUI-SuperBeasts](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts)

---

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).

---

## Update History

### 2026.1.9

- Removed CLIP loader nodes following optimizations in the latest ComfyUI core update.

### 2025.12.18

- Removed `hidream.py` replacement functionality.

### 2025.12.1

- Added various Model Merge nodes.

### 2025.10.28

- Implemented `ModelMergeHiDream`.
- Disabled default replacement of `HiDream.py`.

### 2025.9.21

- Added toggle functionality for Experimental Text Encoder Modules in ComfyUI settings.
