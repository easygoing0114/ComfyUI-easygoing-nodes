# ComfyUI-easygoing-nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), device-select CLIP loaders,　providing HDR effects, and image saving with prompt metadata.

## ✨ Features
- **HDR Effects with LAB Adjust**  
  Tone-mapping with control over shadows, highlights, gamma, contrast, color boost, and LAB channel adjustments.  
  👉 This node is based on the HDR processing from [ComfyUI-SuperBeasts](https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts) with additional color adjustments.

- **Save Image With Prompt**  
  Save images with positive/negative prompts and captions embedded in PNG metadata.

- **CLIP Loaders**  
  - Quadruple CLIP Loader (Set Device)  
  - Triple CLIP Loader (Set Device)  
  - CLIP Vision Loader (Set Device)

## 📥 Installation
1. Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR-USERNAME/ComfyUI-easygoing-nodes.git
```
2. Restart ComfyUI. The new nodes should now appear in the node search.

## 📂 Nodes Overview

| Node Name                        | Display Name                     | Category                  |
|----------------------------------|----------------------------------|---------------------------|
| `HDREffectsLabAdjust`            | HDR Effects with LAB Adjusts     | SuperBeastsAI/Image       |
| `SaveImageWithPrompt`            | Save Image With Prompt           | image                     |
| `QuadrupleCLIPLoaderSetDevice`   | Quadruple CLIP Loader (Set Device) | advanced/loaders        |
| `TripleCLIPLoaderSetDevice`      | Triple CLIP Loader (Set Device)    | advanced/loaders        |
| `CLIPVisionLoaderSetDevice`      | Load CLIP Vision (Set Device)      | advanced/loaders        |

## ⚖️ License
This project is licensed under the [MIT License](LICENSE).
