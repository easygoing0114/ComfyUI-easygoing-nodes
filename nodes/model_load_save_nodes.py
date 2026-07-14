import re
import os
import torch
import comfy.sd
import folder_paths

from safetensors.torch import load_file, save_file


def clean_ckpt_name(ckpt_name):
    """ckpt_name(ファイル名)から拡張子・精度/形式系のキーワードを除去し、
    末尾に残った '-' や '_' も取り除いた文字列を返す。"""

    name = ckpt_name

    # 拡張子の除去（.safetensors, .pt, .pth）
    name = re.sub(r'\.(safetensors|pt|pth)', '', name, flags=re.IGNORECASE)

    # 精度・形式系キーワードの除去
    keywords = [
        "fp32", "fp16", "bf16", "fp8",
        "svd", "scaled", "mxfp8", "nvfp4",
        "int8", "convrot", "hq",
        "e4m3fn", "e4m3", "e5m2fn", "e5m2",
    ]
    pattern = r'(?:' + '|'.join(keywords) + r')'
    name = re.sub(pattern, '', name, flags=re.IGNORECASE)

    # 文末に残った '-' や '_' の除去（複数連続もまとめて除去）
    name = re.sub(r'[-_]+$', '', name)

    return name


def get_combined_model_list():
    """checkpoints と diffusion_models 両方のフォルダからファイル名一覧を取得し、
    重複を除いてソートしたリストを返す。"""
    ckpt_list = folder_paths.get_filename_list("checkpoints")
    unet_list = folder_paths.get_filename_list("diffusion_models")
    combined = sorted(set(ckpt_list) | set(unet_list))
    return combined


def get_model_full_path(model_name):
    """model_name が checkpoints / diffusion_models のどちらのフォルダに
    存在するかを調べ、(folder_type, full_path) を返す。"""
    ckpt_list = folder_paths.get_filename_list("checkpoints")
    if model_name in ckpt_list:
        return "checkpoints", folder_paths.get_full_path_or_raise("checkpoints", model_name)

    unet_list = folder_paths.get_filename_list("diffusion_models")
    if model_name in unet_list:
        return "diffusion_models", folder_paths.get_full_path_or_raise("diffusion_models", model_name)

    raise FileNotFoundError(f"Model '{model_name}' not found in 'checkpoints' or 'diffusion_models' folders.")


class LoadCheckpointWithName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_combined_model_list(), {"tooltip": "The name of the checkpoint (model) to load. Searches both 'checkpoints' and 'diffusion_models' folders."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "model_name")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space.",
        "Cleaned model name (extension and precision/format keywords removed).",
    )
    FUNCTION = "load_checkpoint"

    CATEGORY = "model/loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint (from 'checkpoints' or 'diffusion_models' folders) and also outputs a cleaned name string."
    SEARCH_ALIASES = ["load model", "checkpoint", "model loader", "load checkpoint", "ckpt", "model"]

    def load_checkpoint(self, model_name):
        folder_type, ckpt_path = get_model_full_path(model_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        cleaned_name = clean_ckpt_name(model_name)
        return out[:3] + (cleaned_name,)


class LoadDiffusionModelWithName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_combined_model_list(), {"tooltip": "The name of the diffusion model (UNET) to load. Searches both 'checkpoints' and 'diffusion_models' folders."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"advanced": True})
            }
        }
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "model_name")
    FUNCTION = "load_unet"

    CATEGORY = "model/loaders"
    DESCRIPTION = "Loads a diffusion model (UNET) from 'checkpoints' or 'diffusion_models' folders and also outputs a cleaned name string."

    def load_unet(self, model_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        folder_type, unet_path = get_model_full_path(model_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        cleaned_name = clean_ckpt_name(model_name)
        return (model, cleaned_name)


"""
Load Original Model / Model Save with Original
===============================================
ComfyUI でモデルを加工・保存する際に生じる 2 つの問題を解決するノード群。

Problem 1 — Key prefix mismatch
  ComfyUI は safetensors をロードするとき、モデルアーキテクチャに応じて
  state_dict のキーに独自のプレフィックスを付与することがある。
  その結果、元ファイルのキー名と ComfyUI 内部のキー名が一致しなくなる。

  例:
    safetensors のキー : "backbone.layer.0.weight"
    ComfyUI state_dict : "diffusion_model.backbone.layer.0.weight"

  対処: 両側のキー集合に対してドット区切りのプレフィックス候補を列挙し、
        総当たりでマッチ数が最大になる組み合わせを自動選択する。

Problem 2 — Missing tensors
  ComfyUI の ModelSave は、ロード時に使用しなかったテンソルを
  state_dict に含めないため、一部テンソルが消失することがある。

  対処: 元ファイルの全テンソルを "original_model" として保持し、
        保存時に ComfyUI の state_dict にないテンソルを補完する。

Nodes:
  LoadOriginalModel     — safetensors を生読みして original_model と model を出力
  ModelSaveWithOriginal — 調整済み model と original_model を合成して保存
"""

# ------------------------------------------------------------------------------
# Type alias
# original_model は {"tensors": dict[str, Tensor], "metadata": dict} の辞書
# ------------------------------------------------------------------------------

ORIGINAL_MODEL_TYPE = "ORIGINAL_MODEL"


# ------------------------------------------------------------------------------
# Utilities — prefix-agnostic key matching
# ------------------------------------------------------------------------------

def _collect_prefix_candidates(keys: list[str]) -> list[str]:
    """キー集合からドット区切りのプレフィックス候補を列挙する。空文字列を常に含む。"""
    candidates = {""}
    for key in keys:
        parts = key.split(".")
        for i in range(1, len(parts)):
            candidates.add(".".join(parts[:i]) + ".")
    return list(candidates)


def _strip_prefix_from_keys(keys: list[str], prefix: str) -> dict[str, str]:
    """
    {正規化キー: 元キー} の辞書を返す。
    prefix に一致しないキーは正規化キーとしてそのまま使用する。
    """
    result = {}
    for k in keys:
        norm = k[len(prefix):] if (prefix and k.startswith(prefix)) else k
        result[norm] = k
    return result


def _find_best_prefix_pair(
    orig_keys: list[str],
    comfy_keys: list[str],
) -> tuple[str, str, dict[str, str], dict[str, str]]:
    """
    両側のプレフィックス候補を総当たりし、正規化後のマッチ数が
    最大になる (orig_prefix, comfy_prefix) の組み合わせを選択する。

    Returns:
        orig_prefix  : ORIG 側で除去するプレフィックス
        comfy_prefix : ComfyUI 側で除去するプレフィックス
        orig_norm    : {正規化キー: ORIG 元キー}
        comfy_norm   : {正規化キー: ComfyUI 元キー}
    """
    orig_candidates = _collect_prefix_candidates(orig_keys)
    comfy_candidates = _collect_prefix_candidates(comfy_keys)

    # キャッシュで重複計算を避ける
    orig_cache: dict[str, dict[str, str]] = {}
    comfy_cache: dict[str, dict[str, str]] = {}

    def get_orig(pfx: str) -> dict[str, str]:
        if pfx not in orig_cache:
            orig_cache[pfx] = _strip_prefix_from_keys(orig_keys, pfx)
        return orig_cache[pfx]

    def get_comfy(pfx: str) -> dict[str, str]:
        if pfx not in comfy_cache:
            comfy_cache[pfx] = _strip_prefix_from_keys(comfy_keys, pfx)
        return comfy_cache[pfx]

    best_count = -1
    best_orig_pfx = ""
    best_comfy_pfx = ""

    for op in orig_candidates:
        on_set = set(get_orig(op).keys())
        for cp in comfy_candidates:
            count = len(on_set & set(get_comfy(cp).keys()))
            if count > best_count:
                best_count = count
                best_orig_pfx = op
                best_comfy_pfx = cp

    print(
        f"[ModelSaveWithOriginal] prefix resolution: "
        f"orig={repr(best_orig_pfx)}, "
        f"comfy={repr(best_comfy_pfx)}, "
        f"matched_keys={best_count}"
    )

    return best_orig_pfx, best_comfy_pfx, get_orig(best_orig_pfx), get_comfy(best_comfy_pfx)


# ------------------------------------------------------------------------------
# Node: Load Original Model
# ------------------------------------------------------------------------------

class LoadOriginalModel:
    """
    safetensors ファイルを生のまま読み込み、2 種類の出力を提供する。

    outputs:
      original_model : ORIGINAL_MODEL — 全テンソルと元キー名を保持した辞書
      model          : MODEL          — 通常の ComfyUI MODEL（UNETLoader 相当）
    """

    @classmethod
    def INPUT_TYPES(cls):
        unet_names = folder_paths.get_filename_list("diffusion_models")
        ckpt_names = folder_paths.get_filename_list("checkpoints")
        all_names = sorted(set(unet_names + ckpt_names))
        return {
            "required": {
                "unet_name": (all_names,),
                "weight_dtype": (
                    ["default", "fp32", "fp16", "bf16"],
                    {"default": "default"},
                ),
            }
        }

    RETURN_TYPES = (ORIGINAL_MODEL_TYPE, "MODEL")
    RETURN_NAMES = ("original_model", "model")
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, unet_name: str, weight_dtype: str):
        # ファイルパス解決（diffusion_models → checkpoints の順にフォールバック）
        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if path is None:
            path = folder_paths.get_full_path("checkpoints", unet_name)
        if path is None:
            raise FileNotFoundError(
                f"Model file not found: {unet_name}\n"
                f"Place it in the diffusion_models or checkpoints folder."
            )

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        load_dtype = dtype_map.get(weight_dtype, None)

        # safetensors を生読みして元キー名・全テンソルを保持する
        raw_sd = load_file(path, device="cpu")
        if load_dtype is not None:
            raw_sd = {k: v.to(load_dtype) for k, v in raw_sd.items()}

        # __metadata__ を取得
        metadata = {}
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                metadata = dict(f.metadata()) if f.metadata() else {}
        except Exception:
            pass

        original_model = {
            "tensors": raw_sd,
            "metadata": metadata,
            "source_path": path,
        }

        # 通常の ComfyUI MODEL をロード（UNETLoader と同等）
        import comfy.sd

        model_options = {}
        if weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16

        comfy_model = comfy.sd.load_diffusion_model(path, model_options=model_options)

        return (original_model, comfy_model)


# ------------------------------------------------------------------------------
# Node: Model Save with Original
# ------------------------------------------------------------------------------

class ModelSaveWithOriginal:
    """
    ComfyUI の調整済み model と original_model を合成して safetensors に保存する。

    処理フロー:
      1. model から state_dict を取得する
      2. 両側のキーに対してプレフィックス総当たりマッチングを実行する
      3. マッチしたテンソルは ComfyUI の調整済み値で上書きする
      4. マッチしなかったテンソル（消失テンソル）は original から復元する
      5. 保存キー名は original_model の元キー名を使用する

    inputs:
      original_model : ORIGINAL_MODEL — LoadOriginalModel の出力
      model          : MODEL          — 調整済み ComfyUI MODEL
      filename_prefix: 保存ファイル名のプレフィックス
      save_metadata  : True のとき original のメタデータを引き継ぐ
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_model": (ORIGINAL_MODEL_TYPE,),
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "model_restored"}),
            },
            "optional": {
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "model_merging"

    def save(
        self,
        original_model: dict,
        model,
        filename_prefix: str,
        save_metadata: bool = True,
    ):
        orig_tensors: dict[str, torch.Tensor] = original_model["tensors"]
        orig_metadata: dict = original_model.get("metadata", {})

        # ComfyUI model から state_dict を取得
        try:
            comfy_sd: dict[str, torch.Tensor] = model.model.state_dict()
        except AttributeError:
            try:
                comfy_sd = model.state_dict()
            except AttributeError:
                raise RuntimeError(
                    "Could not retrieve state_dict from model. "
                    "Check that LoadOriginalModel is connected to ModelSaveWithOriginal."
                )

        # プレフィックス総当たりマッチング
        orig_keys = list(orig_tensors.keys())
        comfy_keys = list(comfy_sd.keys())

        _, _, orig_norm_map, comfy_norm_map = _find_best_prefix_pair(orig_keys, comfy_keys)

        # 正規化キー → ComfyUI テンソル の逆引き辞書
        comfy_norm_tensors: dict[str, torch.Tensor] = {
            norm_k: comfy_sd[orig_k]
            for norm_k, orig_k in comfy_norm_map.items()
        }

        # 合成: マッチしたテンソルは ComfyUI の値を採用し、消失テンソルは original から復元
        output_sd: dict[str, torch.Tensor] = {}
        matched = 0
        restored = 0

        for norm_key, orig_key in orig_norm_map.items():
            if norm_key in comfy_norm_tensors:
                output_sd[orig_key] = comfy_norm_tensors[norm_key].to(
                    orig_tensors[orig_key].dtype
                )
                matched += 1
            else:
                output_sd[orig_key] = orig_tensors[orig_key]
                restored += 1

        comfy_only = set(comfy_norm_map.keys()) - set(orig_norm_map.keys())

        print(
            f"[ModelSaveWithOriginal] "
            f"matched={matched}, restored={restored}, "
            f"comfy_only(skipped)={len(comfy_only)}"
        )
        if comfy_only:
            print(
                "[ModelSaveWithOriginal] comfy_only keys (skipped): "
                + ", ".join(sorted(comfy_only)[:10])
                + ("..." if len(comfy_only) > 10 else "")
            )

        # 保存先パスの決定（連番で衝突回避）
        output_dir = folder_paths.get_output_directory()
        base_dir = os.path.join(output_dir, os.path.dirname(filename_prefix))
        os.makedirs(base_dir, exist_ok=True)
        base_name = os.path.basename(filename_prefix)

        counter = 1
        while True:
            suffix = f"_{counter:04d}" if counter > 1 else ""
            filename = f"{base_name}{suffix}.safetensors"
            full_path = os.path.join(base_dir, filename)
            if not os.path.exists(full_path):
                break
            counter += 1

        # メタデータの構築
        save_meta: dict[str, str] | None = None
        if save_metadata:
            save_meta = {k: str(v) for k, v in orig_metadata.items()} if orig_metadata else {}
            save_meta["restored_by"] = "ModelSaveWithOriginal"
            save_meta["restored_tensors"] = str(restored)
            save_meta["matched_tensors"] = str(matched)

        save_file(output_sd, full_path, metadata=save_meta)

        print(f"[ModelSaveWithOriginal] saved: {full_path}")
        print(
            f"  total tensors: {len(output_sd)} "
            f"(orig: {len(orig_tensors)}, matched: {matched}, restored: {restored})"
        )

        return {}


# ------------------------------------------------------------------------------
# Registrations
# ------------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LoadCheckpointWithName": LoadCheckpointWithName,
    "LoadDiffusionModelWithName": LoadDiffusionModelWithName,
    "LoadOriginalModel": LoadOriginalModel,
    "ModelSaveWithOriginal": ModelSaveWithOriginal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCheckpointWithName": "Load Checkpoint (with Name)",
    "LoadDiffusionModelWithName": "Load Diffusion Model (with Name)",
    "LoadOriginalModel": "Load Original Model",
    "ModelSaveWithOriginal": "Model Save with Original",
}