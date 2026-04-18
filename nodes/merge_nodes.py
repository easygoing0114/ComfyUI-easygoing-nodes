import comfy.sd
import comfy.utils
import comfy_extras.nodes_model_merging
import folder_paths
import json
import os
import torch
from comfy.cli_args import args


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

    get_sd() はComfyUI内部のldm形式キーを返す:
      encoder.down_blocks.N.resnets.M.* -> encoder.down.N.block.M.*
      encoder.mid_block.resnets.0.*     -> encoder.mid.block_1.*
      encoder.mid_block.resnets.1.*     -> encoder.mid.block_2.*
      encoder.mid_block.attentions.0.*  -> encoder.mid.attn_1.*
      encoder.conv_norm_out.*           -> encoder.norm_out.*
      decoder.up_blocks.N.*             -> decoder.up.N.*
      decoder.mid_block.resnets.0.*     -> decoder.mid.block_1.*
      decoder.mid_block.resnets.1.*     -> decoder.mid.block_2.*
      decoder.mid_block.attentions.0.*  -> decoder.mid.attn_1.*
      decoder.conv_norm_out.*           -> decoder.norm_out.*
    """

    LAYER_KEYS = [
        # Quantization
        "quant_conv",
        "post_quant_conv",
        # Encoder
        "encoder.conv_in",
        *[f"encoder.down.{i}." for i in range(4)],
        "encoder.mid.attn_1.",
        "encoder.mid.block_1.",
        "encoder.mid.block_2.",
        "encoder.norm_out",
        "encoder.conv_out",
        # Decoder
        "decoder.conv_in",
        "decoder.mid.attn_1.",
        "decoder.mid.block_1.",
        "decoder.mid.block_2.",
        *[f"decoder.up.{i}." for i in range(4)],
        "decoder.norm_out",
        "decoder.conv_out",
    ]

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"vae": ("VAE",)}
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        for key in s.LAYER_KEYS:
            arg_dict[key] = argument
        return {"required": arg_dict}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"
    DESCRIPTION = (
        "Scale specific layers of SDXL VAE. "
        "Scale=1.0 keeps original, Scale=0.0 zeroes out the layer."
    )

    def scale(self, vae, **kwargs):
        # kwargs のキーを LAYER_KEYS に復元（ComfyUI が "." を "_" に変換する場合に対応）
        kwarg_to_layer = {
            layer_key.replace(".", "_"): layer_key
            for layer_key in self.LAYER_KEYS
        }
        ratios = {}
        for k, v in kwargs.items():
            if k in kwarg_to_layer:
                ratios[kwarg_to_layer[k]] = v
            elif k in self.LAYER_KEYS:
                ratios[k] = v

        # テンソルのスケーリング
        sd = vae.get_sd()
        new_sd = {}
        for tensor_key, tensor_val in sd.items():
            scale = 1.0
            matched_len = 0
            for arg_name, arg_value in ratios.items():
                if tensor_key.startswith(arg_name) and len(arg_name) > matched_len:
                    scale = arg_value
                    matched_len = len(arg_name)
            new_sd[tensor_key] = tensor_val * scale if scale != 1.0 else tensor_val

        new_vae = comfy.sd.VAE(sd=new_sd)
        return (new_vae,)


class VAEMergeSDXLBlock:
    """
    2つのSDXL VAEをブロック単位でマージするノード
    ratio=1.0 で vae2 を使用、ratio=0.0 で vae1 を使用

    get_sd() はComfyUI内部のldm形式キーを返す:
      encoder.down_blocks.N.resnets.M.* -> encoder.down.N.block.M.*
      encoder.mid_block.resnets.0.*     -> encoder.mid.block_1.*
      encoder.mid_block.resnets.1.*     -> encoder.mid.block_2.*
      encoder.mid_block.attentions.0.*  -> encoder.mid.attn_1.*
      encoder.conv_norm_out.*           -> encoder.norm_out.*
      decoder.up_blocks.N.*             -> decoder.up.N.*
      decoder.mid_block.resnets.0.*     -> decoder.mid.block_1.*
      decoder.mid_block.resnets.1.*     -> decoder.mid.block_2.*
      decoder.mid_block.attentions.0.*  -> decoder.mid.attn_1.*
      decoder.conv_norm_out.*           -> decoder.norm_out.*
    """

    LAYER_KEYS = [
        # Quantization
        "quant_conv",
        "post_quant_conv",
        # Encoder
        "encoder.conv_in",
        *[f"encoder.down.{i}." for i in range(4)],
        "encoder.mid.attn_1.",
        "encoder.mid.block_1.",
        "encoder.mid.block_2.",
        "encoder.norm_out",
        "encoder.conv_out",
        # Decoder
        "decoder.conv_in",
        "decoder.mid.attn_1.",
        "decoder.mid.block_1.",
        "decoder.mid.block_2.",
        *[f"decoder.up.{i}." for i in range(4)],
        "decoder.norm_out",
        "decoder.conv_out",
    ]

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "vae1": ("VAE",),
            "vae2": ("VAE",),
        }
        argument = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
        for key in s.LAYER_KEYS:
            arg_dict[key] = argument
        return {"required": arg_dict}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "merge"
    CATEGORY = "advanced/model_merging/model_specific"
    DESCRIPTION = (
        "Block-wise merging for SDXL VAE. Ratio=0.0 keeps vae1, Ratio=1.0 uses vae2."
    )

    def merge(self, vae1, vae2, **kwargs):
        # kwargs のキーを LAYER_KEYS に復元（ComfyUI が "." を "_" に変換する場合に対応）
        kwarg_to_layer = {
            layer_key.replace(".", "_"): layer_key
            for layer_key in self.LAYER_KEYS
        }
        ratios = {}
        for k, v in kwargs.items():
            if k in kwarg_to_layer:
                ratios[kwarg_to_layer[k]] = v
            elif k in self.LAYER_KEYS:
                ratios[k] = v

        sd1 = vae1.get_sd()
        sd2 = vae2.get_sd()

        new_sd = {}

        for k in sd1.keys():
            if k not in sd2:
                new_sd[k] = sd1[k]
                continue

            ratio = 0.5
            matched_len = 0
            for arg_name, arg_value in ratios.items():
                if k.startswith(arg_name) and len(arg_name) > matched_len:
                    ratio = arg_value
                    matched_len = len(arg_name)

            new_sd[k] = sd1[k] * (1.0 - ratio) + sd2[k] * ratio

        # sd2 にしか存在しないキーはそのまま追加
        for k in sd2.keys():
            if k not in new_sd:
                new_sd[k] = sd2[k]

        new_vae = comfy.sd.VAE(sd=new_sd)
        return (new_vae,)


class VAEScaleFluxBlock:
    """
    FLUX1 VAEの特定の層をスケーリングするノード
    scale=1.0 でそのまま、scale=0.0 で完全に抑制

    FLUX1 AEのget_sd()はldm形式をそのまま返すため、
    SDXLのようなキーリマップは不要。
    ただしComfyUIがINPUT_TYPESのキーの"."を"_"に変換して
    kwargsに渡すため、逆引きテーブルで復元が必要。
    """

    LAYER_KEYS = [
        # Encoder
        "encoder.conv_in",
        "encoder.conv_out",
        "encoder.norm_out",
        *[f"encoder.down.{i}." for i in range(4)],
        *[f"encoder.down.{i}.block.{j}." for i in range(4) for j in range(3)],
        *[f"encoder.down.{i}.downsample." for i in range(3)],  # 0-2 のみ
        "encoder.mid.",
        "encoder.mid.block_1.",
        "encoder.mid.block_2.",
        "encoder.mid.attn_1.",
        # Decoder
        "decoder.conv_in",
        "decoder.conv_out",
        "decoder.norm_out",
        *[f"decoder.up.{i}." for i in range(4)],
        *[f"decoder.up.{i}.block.{j}." for i in range(4) for j in range(3)],
        *[f"decoder.up.{i}.upsample." for i in range(1, 4)],  # 1-3 のみ
        "decoder.mid.",
        "decoder.mid.block_1.",
        "decoder.mid.block_2.",
        "decoder.mid.attn_1.",
    ]

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"vae": ("VAE",)}
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        for key in s.LAYER_KEYS:
            arg_dict[key] = argument
        return {"required": arg_dict}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"
    DESCRIPTION = "Scale specific layers of FLUX1 VAE. Scale=1.0 keeps original, Scale=0.0 zeroes out the layer."

    def scale(self, vae, **kwargs):
        # ComfyUI が "." を "_" に変換する場合に備えて逆引きテーブルを作成
        kwarg_to_layer = {
            layer_key.replace(".", "_"): layer_key
            for layer_key in self.LAYER_KEYS
        }
        ratios = {}
        for k, v in kwargs.items():
            if k in kwarg_to_layer:
                ratios[kwarg_to_layer[k]] = v
            elif k in self.LAYER_KEYS:
                ratios[k] = v

        sd = vae.get_sd()
        new_sd = {}
        for tensor_key, tensor_val in sd.items():
            scale = 1.0
            matched_len = 0
            for arg_name, arg_value in ratios.items():
                if tensor_key.startswith(arg_name) and len(arg_name) > matched_len:
                    scale = arg_value
                    matched_len = len(arg_name)
            new_sd[tensor_key] = tensor_val * scale if scale != 1.0 else tensor_val

        new_vae = comfy.sd.VAE(sd=new_sd)
        return (new_vae,)


class VAEScaleFlux2Block:
    """
    FLUX2 VAEの特定の層をスケーリングするノード
    scale=1.0 でそのまま、scale=0.0 で完全に抑制

    get_sd() はComfyUI内部のldm形式キーを返す（SDXLと同様）:
      encoder.down_blocks.N.*          -> encoder.down.N.*
      encoder.mid_block.resnets.0.*    -> encoder.mid.block_1.*
      encoder.mid_block.resnets.1.*    -> encoder.mid.block_2.*
      encoder.mid_block.attentions.0.* -> encoder.mid.attn_1.*
      encoder.conv_norm_out.*          -> encoder.norm_out.*
      decoder.up_blocks.N.*            -> decoder.up.N.*
      decoder.mid_block.resnets.0.*    -> decoder.mid.block_1.*
      decoder.mid_block.resnets.1.*    -> decoder.mid.block_2.*
      decoder.mid_block.attentions.0.* -> decoder.mid.attn_1.*
      decoder.conv_norm_out.*          -> decoder.norm_out.*

    注意: bn.* テンソル（I64）はスケーリング対象外
    """

    LAYER_KEYS = [
        # Quantization
        "quant_conv",
        "post_quant_conv",
        # Encoder
        "encoder.conv_in",
        "encoder.conv_out",
        "encoder.norm_out",
        *[f"encoder.down.{i}." for i in range(4)],
        "encoder.mid.attn_1.",
        "encoder.mid.block_1.",
        "encoder.mid.block_2.",
        # Decoder
        "decoder.conv_in",
        "decoder.conv_out",
        "decoder.norm_out",
        "decoder.mid.attn_1.",
        "decoder.mid.block_1.",
        "decoder.mid.block_2.",
        *[f"decoder.up.{i}." for i in range(4)],
    ]

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {"vae": ("VAE",)}
        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01})
        for key in s.LAYER_KEYS:
            arg_dict[key] = argument
        return {"required": arg_dict}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "scale"
    CATEGORY = "advanced/model_merging/model_specific"
    DESCRIPTION = "Scale specific layers of FLUX2 VAE. Scale=1.0 keeps original, Scale=0.0 zeroes out the layer."

    # I64など浮動小数点演算できないdtypeを除外
    _SKIP_DTYPES = {torch.int32, torch.int64, torch.bool}

    def scale(self, vae, **kwargs):
        # ComfyUI が "." を "_" に変換する場合に備えて逆引きテーブルを作成
        kwarg_to_layer = {
            layer_key.replace(".", "_"): layer_key
            for layer_key in self.LAYER_KEYS
        }
        ratios = {}
        for k, v in kwargs.items():
            if k in kwarg_to_layer:
                ratios[kwarg_to_layer[k]] = v
            elif k in self.LAYER_KEYS:
                ratios[k] = v

        sd = vae.get_sd()
        new_sd = {}
        for tensor_key, tensor_val in sd.items():
            # I64 等の整数テンソルはスケーリング不可のためそのままコピー
            if tensor_val.dtype in self._SKIP_DTYPES:
                new_sd[tensor_key] = tensor_val
                continue

            scale = 1.0
            matched_len = 0
            for arg_name, arg_value in ratios.items():
                if tensor_key.startswith(arg_name) and len(arg_name) > matched_len:
                    scale = arg_value
                    matched_len = len(arg_name)

            new_sd[tensor_key] = tensor_val * scale if scale != 1.0 else tensor_val

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

        clip_sd = clip.get_sd()
        output_sd = {}

        # "qwen25_7b.transformer." プレフィックスを削除
        prefix_to_strip = "qwen25_7b.transformer."

        for k, v in clip_sd.items():
            if k.startswith(prefix_to_strip):
                new_key = k[len(prefix_to_strip):]
                output_sd[new_key] = v
            elif k.startswith("qwen25_7b."):
                new_key = k.replace("qwen25_7b.", "")
                output_sd[new_key] = v
            else:
                output_sd[k] = v

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=metadata)

        return {}


# ---- ノード登録用マッピング ----

NODE_CLASS_MAPPINGS = {
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
