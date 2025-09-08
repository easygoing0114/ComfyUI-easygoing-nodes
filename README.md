# ComfyUI-easygoing-nodes

Enhanced Text Encoder modules, acd Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), device-select CLIP loaders, providing HDR effects, image saving with prompt metadata.

## ✨ Features

- **🔧 Enhanced Text Encoder Modules**  
  Automatically replaces ComfyUI's built-in Text Encoder modules with enhanced versions that include:
  - **CLIP-G Improvements**: [Enhanced attention mask support and better tokenization](https://note.com/gentle_murre488/n/n12f2ecce1e00)
  - **HiDream Text Encoder**: Advanced encoder support with CPU optimization for better memory management

- **CLIP Loaders**  
  - Quadruple CLIP Loader (Set Device)  
  - Triple CLIP Loader (Set Device)  
  - CLIP Vision Loader (Set Device)

- **HDR Effects with LAB Adjust**  
  Tone-mapping with control over shadows, highlights, gamma, contrast, color boost, and LAB channel adjustments.  
  💡 This node is based on the HDR processing from [ComfyUI-SuperBeasts](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts) with additional color adjustments.

- **Save Image With Prompt**  
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

## 🔄 Module Enhancement Details

This custom node package includes enhanced versions of core ComfyUI modules:

### **Enhanced SDXL CLIP (`sdxl_clip.py`)**
- ✅ **Improved Attention Masks**: Better support for attention masking mechanisms
- ✅ **Enhanced Tokenization**: Optimized tokenizer with improved padding strategies

### **Enhanced HiDream Text Encoder (`hidream.py`)**
- ✅ **CPU Optimization**: Forced CPU processing for better memory management

### **Safety & Compatibility**
- ⚡ **Non-Destructive**: Original ComfyUI files are never modified
- 🔄 **Reversible**: Simply remove the custom node to revert to original functionality
- 🛡️ **Safe**: Changes exist only in memory during ComfyUI runtime
- 📦 **Isolated**: No impact on other custom nodes or ComfyUI core functionality

## 📂 Nodes Overview

| Node Name                        | Display Name                     | Category                  |
|----------------------------------|----------------------------------|---------------------------|
| `QuadrupleCLIPLoaderSetDevice`   | Quadruple CLIP Loader (Set Device) | advanced/loaders        |
| `TripleCLIPLoaderSetDevice`      | Triple CLIP Loader (Set Device)    | advanced/loaders        |
| `CLIPVisionLoaderSetDevice`      | Load CLIP Vision (Set Device)      | advanced/loaders        |
| `HDREffectsLabAdjust`            | HDR Effects with LAB Adjusts     | SuperBeastsAI/Image       |
| `SaveImageWithPrompt`            | Save Image With Prompt           | image                     |

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
- HDR Effects based on [ComfyUI-SuperBeasts](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts)
- Enhanced SDXL CLIP module (`sdxl_clip.py`) by [Shiba-2-shiba](https://github.com/Shiba-2-shiba)
