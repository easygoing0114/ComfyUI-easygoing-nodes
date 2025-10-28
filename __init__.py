import sys
import os
import importlib.util
from pathlib import Path

# カスタムノードのディレクトリを取得
CUSTOM_NODE_DIR = Path(__file__).parent

# 設定マネージャーをインポート
try:
    from .settings_manager import get_settings_manager
    from .web_api import register_api_routes
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

def replace_module_with_custom(module_path, custom_module_path):
    """既存のモジュールをカスタムモジュールで置き換える"""
    try:
        spec = importlib.util.spec_from_file_location(module_path, custom_module_path)
        custom_module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = custom_module
        spec.loader.exec_module(custom_module)
        return True
    except Exception:
        return False

def is_module_replacement_enabled(module_name):
    """設定に基づいてモジュール置換が有効かどうかを確認"""
    if not SETTINGS_AVAILABLE:
        return True
    
    try:
        settings_manager = get_settings_manager()
        return settings_manager.is_module_enabled(module_name)
    except Exception:
        return True

def apply_custom_modules():
    """修正版モジュールを設定に基づいて適用"""
    # 修正版ファイルのパス
    custom_sdxl_clip_path = CUSTOM_NODE_DIR / "modified_modules" / "sdxl_clip.py"
    custom_hidream_path = CUSTOM_NODE_DIR / "modified_modules" / "hidream.py"
    
    applied_modules = []
    skipped_modules = []
    
    # 設定を読み込んで表示
    if SETTINGS_AVAILABLE:
        try:
            settings_manager = get_settings_manager()
            settings = settings_manager.get_settings()
            print(f"EasygoingNodes settings loaded: {settings}")
        except Exception:
            pass
    
    # sdxl_clip.pyの置き換え
    if custom_sdxl_clip_path.exists():
        if is_module_replacement_enabled("sdxl_clip"):
            if replace_module_with_custom("comfy.sdxl_clip", custom_sdxl_clip_path):
                applied_modules.append("sdxl_clip")
        else:
            skipped_modules.append("sdxl_clip")
    
    # hidream.pyの置き換え
    if custom_hidream_path.exists():
        if is_module_replacement_enabled("hidream"):
            if replace_module_with_custom("comfy.text_encoders.hidream", custom_hidream_path):
                applied_modules.append("hidream")
        else:
            skipped_modules.append("hidream")
    
    # 結果のサマリーを表示
    if applied_modules:
        print(f"✓ Applied module replacements: {', '.join(applied_modules)}")
    if skipped_modules:
        print(f"⊘ Skipped module replacements: {', '.join(skipped_modules)}")
    
    return True

def setup_web_api():
    """ComfyUIのWebサーバーにAPIルートを登録"""
    if not SETTINGS_AVAILABLE:
        return
    
    try:
        import server
        if hasattr(server, 'PromptServer'):
            app = server.PromptServer.instance.app
            register_api_routes(app)
    except Exception:
        pass

# モジュール置換を実行
try:
    apply_custom_modules()
    setup_web_api()
except Exception:
    pass

# 既存のノード定義をインポート
try:
    from .easygoing_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Web拡張機能の定義
WEB_DIRECTORY = "./web"

__all__ = [
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS',
    'WEB_DIRECTORY'
]