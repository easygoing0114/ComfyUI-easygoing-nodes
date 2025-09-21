import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class EasygoingSettingsManager:
    """EasygoingNodesの設定を管理するクラス"""
    
    def __init__(self):
        self.settings_file = Path(__file__).parent / "settings.json"
        self._settings = None
        self.default_settings = {
            "enable_sdxl_clip": True,
            "enable_hidream": True
        }
        self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """設定ファイルから設定を読み込み"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    self._settings = {**self.default_settings, **loaded_settings}
            else:
                self._settings = self.default_settings.copy()
            return self._settings
        except Exception:
            self._settings = self.default_settings.copy()
            return self._settings
    
    def save_settings(self, settings: Optional[Dict[str, Any]] = None) -> bool:
        """設定をファイルに保存"""
        try:
            if settings is not None:
                self._settings.update(settings)
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception:
            return False
    
    def get_settings(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        if self._settings is None:
            self.load_settings()
        return self._settings.copy()
    
    def get_setting(self, key: str, default=None) -> Any:
        """特定の設定値を取得"""
        if self._settings is None:
            self.load_settings()
        return self._settings.get(key, default)
    
    def update_setting(self, key: str, value: Any) -> bool:
        """特定の設定値を更新"""
        if self._settings is None:
            self.load_settings()
        
        self._settings[key] = value
        return self.save_settings()
    
    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """複数の設定値を一括更新"""
        return self.save_settings(new_settings)
    
    def is_module_enabled(self, module_name: str) -> bool:
        """特定のモジュール置換が有効かどうかを確認"""
        key_mapping = {
            "sdxl_clip": "enable_sdxl_clip",
            "hidream": "enable_hidream"
        }
        
        setting_key = key_mapping.get(module_name, f"enable_{module_name}")
        return self.get_setting(setting_key, False)
    
    def get_enabled_modules(self) -> list:
        """有効化されているモジュールのリストを取得"""
        settings = self.get_settings()
        enabled_modules = []
        
        if settings.get("enable_sdxl_clip", False):
            enabled_modules.append("sdxl_clip")
        if settings.get("enable_hidream", False):
            enabled_modules.append("hidream")
            
        return enabled_modules

# グローバルインスタンス
_global_settings_manager = None

def get_settings_manager() -> EasygoingSettingsManager:
    """グローバル設定マネージャーインスタンスを取得"""
    global _global_settings_manager
    if _global_settings_manager is None:
        _global_settings_manager = EasygoingSettingsManager()
    return _global_settings_manager