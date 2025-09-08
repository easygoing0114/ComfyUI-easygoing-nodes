import sys
import os
import importlib.util
from pathlib import Path

# カスタムノードのディレクトリを取得
CUSTOM_NODE_DIR = Path(__file__).parent

def replace_module_with_custom(module_path, custom_module_path):
    """
    既存のモジュールをカスタムモジュールで置き換える
    
    Args:
        module_path: 置き換える元のモジュールパス（例: "comfy.sdxl_clip"）
        custom_module_path: カスタムモジュールファイルのパス
    """
    try:
        # カスタムモジュールを読み込み
        spec = importlib.util.spec_from_file_location(module_path, custom_module_path)
        custom_module = importlib.util.module_from_spec(spec)
        
        # sys.modulesに登録（既存のモジュールを上書き）
        sys.modules[module_path] = custom_module
        spec.loader.exec_module(custom_module)
        
        print(f"✓ Successfully replaced {module_path} with custom implementation")
        return True
        
    except Exception as e:
        print(f"✗ Failed to replace {module_path}: {str(e)}")
        return False

def apply_custom_modules():
    """
    修正版モジュールを適用
    """
    # 修正版ファイルのパス
    custom_sdxl_clip_path = CUSTOM_NODE_DIR / "modified_modules" / "sdxl_clip.py"
    custom_hidream_path = CUSTOM_NODE_DIR / "modified_modules" / "hidream.py"
    
    success = True
    
    # sdxl_clip.pyの置き換え
    if custom_sdxl_clip_path.exists():
        success &= replace_module_with_custom("comfy.sdxl_clip", custom_sdxl_clip_path)
    else:
        print(f"Info: {custom_sdxl_clip_path} not found - skipping sdxl_clip replacement")
    
    # hidream.pyの置き換え
    if custom_hidream_path.exists():
        success &= replace_module_with_custom("comfy.text_encoders.hidream", custom_hidream_path)
    else:
        print(f"Info: {custom_hidream_path} not found - skipping hidream replacement")
    
    return success

# モジュール置換を実行
print("Loading ComfyUI-easygoing-nodes with module replacements...")
try:
    apply_custom_modules()
    print("Module replacement process completed!")
except Exception as e:
    print(f"Error during module replacement: {str(e)}")

# 既存のノード定義をインポート
from .easygoing_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']