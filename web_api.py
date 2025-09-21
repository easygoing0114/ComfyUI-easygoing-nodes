import json
from pathlib import Path
from aiohttp import web
from .settings_manager import get_settings_manager

class EasygoingWebAPI:
    """
    EasygoingNodesのWeb APIエンドポイントを管理するクラス
    """
    
    def __init__(self):
        self.settings_manager = get_settings_manager()
    
    async def handle_get_settings(self, request):
        """設定を取得するAPIエンドポイント"""
        try:
            settings = self.settings_manager.get_settings()
            return web.json_response({
                "success": True,
                "settings": settings
            })
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def handle_post_settings(self, request):
        """設定を更新するAPIエンドポイント"""
        try:
            # リクエストボディからJSONデータを取得
            data = await request.json()
            
            # 設定を更新
            success = self.settings_manager.update_settings(data)
            
            if success:
                return web.json_response({
                    "success": True,
                    "message": "Settings updated successfully",
                    "settings": self.settings_manager.get_settings()
                })
            else:
                return web.json_response({
                    "success": False,
                    "error": "Failed to save settings"
                }, status=500)
                
        except json.JSONDecodeError:
            return web.json_response({
                "success": False,
                "error": "Invalid JSON in request body"
            }, status=400)
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def handle_reset_settings(self, request):
        """設定をデフォルトにリセットするAPIエンドポイント"""
        try:
            success = self.settings_manager.reset_to_defaults()
            
            if success:
                return web.json_response({
                    "success": True,
                    "message": "Settings reset to defaults",
                    "settings": self.settings_manager.get_settings()
                })
            else:
                return web.json_response({
                    "success": False,
                    "error": "Failed to reset settings"
                }, status=500)
                
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def handle_get_status(self, request):
        """現在のモジュール置換状態を取得するAPIエンドポイント"""
        try:
            enabled_modules = self.settings_manager.get_enabled_modules()
            settings = self.settings_manager.get_settings()
            
            return web.json_response({
                "success": True,
                "enabled_modules": enabled_modules,
                "all_settings": settings,
                "module_status": {
                    "sdxl_clip": {
                        "enabled": settings.get("enable_sdxl_clip", False),
                        "description": "Custom SDXL CLIP implementation"
                    },
                    "hidream": {
                        "enabled": settings.get("enable_hidream", False),
                        "description": "Custom HiDream text encoder implementation"
                    }
                }
            })
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

# グローバルAPIインスタンス
_web_api = None

def get_web_api():
    """グローバルWeb APIインスタンスを取得"""
    global _web_api
    if _web_api is None:
        _web_api = EasygoingWebAPI()
    return _web_api

def register_api_routes(app):
    """
    ComfyUIのWebアプリケーションにAPIルートを登録
    この関数は __init__.py から呼び出される
    """
    api = get_web_api()
    
    # APIルートを追加
    app.router.add_get("/easygoing_nodes/settings", api.handle_get_settings)
    app.router.add_post("/easygoing_nodes/settings", api.handle_post_settings)
    app.router.add_post("/easygoing_nodes/settings/reset", api.handle_reset_settings)
    app.router.add_get("/easygoing_nodes/status", api.handle_get_status)