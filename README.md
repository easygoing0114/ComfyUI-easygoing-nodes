<div align="center">
<img width="705" height="500" alt="ComfyUI Easygoing Nodes thumbnail" src="Images/thumbnail image.png">
</div>

# ComfyUI-easygoing-nodes

Enhanced text encoders and custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with device-selectable CLIP loaders, HDR effects, and image saving with prompt metadata.

## ✨ Features

### 🔧 Enhanced Text Encoder Modules

Automatically replaces ComfyUI's built-in Text Encoder modules with enhanced versions that include:

- **CLIP-G Improvements**: [Enhanced attention mask support and better tokenization](https://note.com/gentle_murre488/n/n12f2ecce1e00)
- (HiDream Text Encoder) – an advanced encoder optimized for CPU efficiency and improved memory management  
  - Disabled by default because ComfyUI’s built-in `HiDream.py` has been updated.

These replacements can be toggled on or off individually from ComfyUI's settings.

### 🧩 CLIP Loaders

- **Quadruple CLIP Loader (Set Device)**

<img width="400" height="189" alt="Quadruple CLIP Loader (Set Device) node" src="Images/QuadrupleCLIPLoaderSetDevice node.png">

- **Triple CLIP Loader (Set Device)**

<img width="400" height="170" alt="Triple CLIP Loader (Set Device) node" src="Images/TripleCLIPLoaderSetDevice node.png">

- **Load CLIP Vision (Set Device)**

<img width="400" height="155" alt="CLIP Vision Loader (Set Device) node" src="Images/CLIPVisionLoaderSetDevice node.png">

Includes an option to load the text encoder into RAM and process it on the CPU (consistent with the default Load CLIP and DualCLIPLoader nodes).

### 🌈 HDR Effects with LAB Adjust

<img width="320" height="374" alt="HDR Effects LAB Adjust node" src="Images/HDREffectsLabAdjust node.png">

**Example**  
**Left: Original | Right: HDR Processing**

<table>
  <tr>
    <td><img width="353" height="250" alt="Before HDR example" src="Images/before HDR.png"></td>
    <td><img width="353" height="250" alt="After HDR example" src="Images/after HDR.png"></td>
  </tr>
</table>

Tone-mapping with control over shadows, highlights, gamma, contrast, color boost, and LAB channel adjustments.  
💡 This node is based on the HDR processing from [ComfyUI-SuperBeasts](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts) with additional color adjustments.

### 🧬 ModelMergeHiDream

<img width="180" height="650" alt="ModelMergeHiDream node" src="Images/ModelMergeHiDream node.png">

Performs hierarchical merging of HiDream models, enabling advanced model blending while preserving structural integrity.

### 💾 Save Image With Prompt

<img width="240" height="356" alt="Save Image With Prompt node" src="Images/SaveImageWithPrompt node.png">

Save images with positive/negative prompts and captions embedded in PNG metadata.

## 🔥 Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/easygoing0114/ComfyUI-easygoing-nodes.git
```

2. For enhanced CLIP/Text Encoder functionality, ensure the modified modules are in place:
   - The enhanced modules are located in `modified_modules/` directory
   - `sdxl_clip.py` - Enhanced SDXL CLIP implementation
   - `hidream.py` - Enhanced HiDream text encoder implementation

3. Restart ComfyUI. The new nodes should now appear in the node search.

## 🔍 Verification

When ComfyUI starts with this custom node, you should see messages like:
```
EasygoingNodes settings loaded: {'enable_sdxl_clip': True, 'enable_hidream': False}
✓ Applied module replacements: sdxl_clip
⊘ Skipped module replacements: hidream
```

If you don't see these messages, check that the `modified_modules/` directory contains the necessary files.

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Credits

- Enhanced SDXL CLIP module (`sdxl_clip.py`) by [Shiba-2-shiba](https://github.com/Shiba-2-shiba)
- HDR Effects based on [ComfyUI-SuperBeasts](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts)

## Update History

### 2025.12.1

- Implemented CheckpointLoaderSetClipDevice
- Implemented Many Merge nodes

### 2025.10.28

- Implemented ModelMergeHiDream
- Default replacement of HiDream.py turned off

### 2025.9.21

Added toggle functionality for Experimental Text Encoder Modules in ComfyUI settings.
