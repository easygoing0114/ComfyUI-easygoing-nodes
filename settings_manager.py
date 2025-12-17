import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class EasygoingSettingsManager:
    """EasygoingNodesの設定を管理するクラス（SDXL CLIP専用）"""

    def __init__(self):
        self.settings_file = Path(__file__).parent / "settings.json"
        self._settings = None
        self.default_settings = {
            "enable_sdxl_clip": True
        }
        # 削除すべき古い設定キー
        self.deprecated_keys = ["enable_hidream"]
        self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """設定ファイルから設定を読み込み（古い設定を自動削除）"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)

                    # 古い設定キーを削除
                    cleaned_settings = {
                        k: v for k, v in loaded_settings.items()
                        if k not in self.deprecated_keys
                    }

                    # デフォルト設定とマージ
                    self._settings = {**self.default_settings, **cleaned_settings}

                    # 古い設定が含まれていた場合は、クリーンな設定で上書き保存
                    if any(k in loaded_settings for k in self.deprecated_keys):
                        self.save_settings()
            else:
                self._settings = self.default_settings.copy()
                self.save_settings()

            return self._settings
        except Exception:
            self._settings = self.default_settings.copy()
            return self._settings

    def save_settings(self, settings: Optional[Dict[str, Any]] = None) -> bool:
        """設定をファイルに保存"""
        try:
            if settings is not None:
                # 新しい設定から古いキーを除外
                cleaned_settings = {
                    k: v for k, v in settings.items()
                    if k not in self.deprecated_keys
                }
                self._settings.update(cleaned_settings)

            # 保存前にも古いキーを除外
            settings_to_save = {
                k: v for k, v in self._settings.items()
                if k not in self.deprecated_keys
            }

            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, indent=2, ensure_ascii=False)

            return True
        except Exception:
            return False

    def get_settings(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        if self._settings is None:
            self.load_settings()
        # 古いキーを除外して返す
        return {
            k: v for k, v in self._settings.items()
            if k not in self.deprecated_keys
        }

    def get_setting(self, key: str, default=None) -> Any:
        """特定の設定値を取得"""
        if self._settings is None:
            self.load_settings()
        # 古いキーの場合はデフォルト値を返す
        if key in self.deprecated_keys:
            return default
        return self._settings.get(key, default)

    def update_setting(self, key: str, value: Any) -> bool:
        """特定の設定値を更新"""
        if self._settings is None:
            self.load_settings()

        # 古いキーは更新しない
        if key in self.deprecated_keys:
            return False

        self._settings[key] = value
        return self.save_settings()

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """複数の設定値を一括更新"""
        return self.save_settings(new_settings)

    def reset_to_defaults(self) -> bool:
        """設定をデフォルトにリセット"""
        self._settings = self.default_settings.copy()
        return self.save_settings()

    def is_module_enabled(self, module_name: str) -> bool:
        """特定のモジュール置換が有効かどうかを確認"""
        if module_name == "sdxl_clip":
            return self.get_setting("enable_sdxl_clip", True)
        # 古いモジュール名の場合は無効を返す
        if module_name == "hidream":
            return False
        return False

    def get_enabled_modules(self) -> list:
        """有効化されているモジュールのリストを取得"""
        settings = self.get_settings()
        enabled_modules = []

        if settings.get("enable_sdxl_clip", True):
            enabled_modules.append("sdxl_clip")

        return enabled_modules

# グローバルインスタンス
_global_settings_manager = None

def get_settings_manager() -> EasygoingSettingsManager:
    """グローバル設定マネージャーインスタンスを取得"""
    global _global_settings_manager
    if _global_settings_manager is None:
        _global_settings_manager = EasygoingSettingsManager()
    return _global_settings_manager
