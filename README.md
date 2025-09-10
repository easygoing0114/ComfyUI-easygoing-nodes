<div align="center">
<img width="705" height="500" alt="thumbnail image" src="Images/thumbnail image.png">
</div>

# ComfyUI-easygoing-nodes

Enhanced Text Encoder modules, add Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), device-select CLIP loaders, providing HDR effects, image saving with prompt metadata.

## ✨ Features

### 🔧 Enhanced Text Encoder Modules

Automatically replaces ComfyUI's built-in Text Encoder modules with enhanced versions that include:

- **CLIP-G Improvements**: [Enhanced attention mask support and better tokenization](https://note.com/gentle_murre488/n/n12f2ecce1e00)
- **HiDream Text Encoder**: Advanced encoder support with CPU optimization for better memory management

### CLIP Loaders
- Quadruple CLIP Loader (Set Device)
- Triple CLIP Loader (Set Device)  
- Load CLIP Vision (Set Device)

<img width="400" height="189" alt="QuadrupleCLIPLoaderSetDevice node" src="Images/QuadrupleCLIPLoaderSetDevice node.png">
<img width="400" height="170" alt="TripleCLIPLoaderSetDevice node" src="Images/TripleCLIPLoaderSetDevice node.png">
<img width="400" height="155" alt="CLIPVisionLoaderSetDevice node" src="Images/CLIPVisionLoaderSetDevice node.png">

  Adds an option to load the text encoder into RAM and process it on the CPU (same implementation as the default Load CLIP and DualCLIPLoader).

### HDR Effects with LAB Adjust

<img width="320" height="374" alt="HDREffectsLabAdjust node" src="Images/HDREffectsLabAdjust node.png">

**Example**
- Left: HDR only | Right: a/b adjust 

<table>
  <tr>
    <td><img width="353" height="250" alt="thumbnail image" src="Images/no ab_adjust.png"></td>
    <td><img width="353" height="250" alt="thumbnail image" src="Images/thumbnail image.png"></td>
  </tr>
</table>

  Tone-mapping with control over shadows, highlights, gamma, contrast, color boost, and LAB channel adjustments.  
  💡 This node is based on the HDR processing from [ComfyUI-SuperBeasts](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts) with additional color adjustments.

### **Save Image With Prompt**

<img width="240" height="356" alt="SaveImageWithPrompt node" src="Images/SaveImageWithPrompt node.png">

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
Loading ComfyUI-easygoing-nodes with module replacements...
✓ Successfully replaced comfy.sdxl_clip with custom implementation
✓ Successfully replaced comfy.text_encoders.hidream with custom implementation
Module replacement process completed!
```

If you don't see these messages, check that the `modified_modules/` directory contains the necessary files.

## ⚖️ License
This project is licensed under the [MIT License](LICENSE).

## 🙏 Credits
- Enhanced SDXL CLIP module (`sdxl_clip.py`) by [Shiba-2-shiba](https://github.com/Shiba-2-shiba)
- HDR Effects based on [ComfyUI-SuperBeasts](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts)
