#!/usr/bin/env python3
"""
ComfyUI-easygoing-nodes設定システムのテストスクリプト
"""

import sys
import json
from pathlib import Path

# スクリプトのディレクトリをパスに追加
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from settings_manager import EasygoingSettingsManager, get_settings_manager
    from web_api import EasygoingWebAPI
    print("✓ Successfully imported settings modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)

def test_settings_manager():
    """設定マネージャーのテスト"""
    print("\n=== Testing Settings Manager ===")
    
    # インスタンス作成
    manager = EasygoingSettingsManager()
    print(f"✓ Created settings manager: {manager}")
    
    # デフォルト設定の確認
    settings = manager.get_settings()
    print(f"✓ Default settings: {settings}")
    
    # 個別設定の取得
    sdxl_enabled = manager.get_setting("enable_sdxl_clip")
    hidream_enabled = manager.get_setting("enable_hidream")
    print(f"✓ SDXL CLIP enabled: {sdxl_enabled}")
    print(f"✓ HiDream enabled: {hidream_enabled}")
    
    # モジュール有効性の確認
    print(f"✓ SDXL CLIP module enabled: {manager.is_module_enabled('sdxl_clip')}")
    print(f"✓ HiDream module enabled: {manager.is_module_enabled('hidream')}")
    
    # 有効なモジュールのリスト
    enabled_modules = manager.get_enabled_modules()
    print(f"✓ Enabled modules: {enabled_modules}")
    
    # 設定更新のテスト
    print("\n--- Testing settings update ---")
    original_sdxl = manager.get_setting("enable_sdxl_clip")
    
    # 設定を変更
    new_value = not original_sdxl
    success = manager.update_setting("enable_sdxl_clip", new_value)
    if success:
        print(f"✓ Updated SDXL CLIP setting to: {new_value}")
        
        # 変更が反映されているか確認
        updated_value = manager.get_setting("enable_sdxl_clip")
        if updated_value == new_value:
            print("✓ Setting update confirmed")
        else:
            print(f"✗ Setting update failed: expected {new_value}, got {updated_value}")
        
        # 元に戻す
        manager.update_setting("enable_sdxl_clip", original_sdxl)
        print(f"✓ Restored original setting: {original_sdxl}")
    else:
        print("✗ Failed to update setting")
    
    # 一括更新のテスト
    print("\n--- Testing bulk update ---")
    original_settings = manager.get_settings()
    test_settings = {
        "enable_sdxl_clip": False,
        "enable_hidream": False
    }
    
    success = manager.update_settings(test_settings)
    if success:
        print("✓ Bulk update successful")
        updated_settings = manager.get_settings()
        print(f"✓ Updated settings: {updated_settings}")
        
        # 元に戻す
        manager.update_settings(original_settings)
        print("✓ Restored original settings")
    else:
        print("✗ Bulk update failed")

def test_web_api():
    """Web APIのテスト（モックリクエスト）"""
    print("\n=== Testing Web API ===")
    
    api = EasygoingWebAPI()
    print("✓ Created Web API instance")
    
    # 設定取得のテスト（実際のHTTPリクエストなしでロジックをテスト）
    try:
        settings = api.settings_manager.get_settings()
        print(f"✓ API can access settings: {settings}")
        
        enabled_modules = api.settings_manager.get_enabled_modules()
        print(f"✓ API can get enabled modules: {enabled_modules}")
        
    except Exception as e:
        print(f"✗ API test failed: {e}")

def test_global_instances():
    """グローバルインスタンスのテスト"""
    print("\n=== Testing Global Instances ===")
    
    # 設定マネージャーのグローバルインスタンス
    manager1 = get_settings_manager()
    manager2 = get_settings_manager()
    
    if manager1 is manager2:
        print("✓ Settings manager singleton working correctly")
    else:
        print("✗ Settings manager singleton failed")
    
    # 設定の一貫性確認
    settings1 = manager1.get_settings()
    settings2 = manager2.get_settings()
    
    if settings1 == settings2:
        print("✓ Settings consistency confirmed")
    else:
        print(f"✗ Settings inconsistency: {settings1} != {settings2}")

def test_file_operations():
    """ファイル操作のテスト"""
    print("\n=== Testing File Operations ===")
    
    # テスト用の設定ファイルパス
    test_settings_file = script_dir / "test_settings.json"
    
    # テスト用設定マネージャー
    class TestSettingsManager(EasygoingSettingsManager):
        def __init__(self):
            super().__init__()
            self.settings_file = test_settings_file
    
    manager = TestSettingsManager()
    
    # 設定を保存
    test_data = {
        "enable_sdxl_clip": True,
        "enable_hidream": False,
        "test_key": "test_value"
    }
    
    success = manager.save_settings(test_data)
    if success and test_settings_file.exists():
        print("✓ Settings file creation successful")
        
        # ファイル内容を確認
        with open(test_settings_file, 'r') as f:
            saved_data = json.load(f)
            
        if saved_data["enable_sdxl_clip"] == test_data["enable_sdxl_clip"]:
            print("✓ Settings file content correct")
        else:
            print("✗ Settings file content incorrect")
        
        # 設定を再読み込み
        manager.load_settings()
        reloaded_settings = manager.get_settings()
        
        if reloaded_settings["test_key"] == "test_value":
            print("✓ Settings reload successful")
        else:
            print("✗ Settings reload failed")
        
        # テストファイルを削除
        test_settings_file.unlink()
        print("✓ Test file cleaned up")
        
    else:
        print("✗ Settings file creation failed")

def main():
    """メインテスト関数"""
    print("ComfyUI-easygoing-nodes Settings System Test")
    print("=" * 50)
    
    try:
        test_settings_manager()
        test_web_api()
        test_global_instances()
        test_file_operations()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        print("\nTo test the full system:")
        print("1. Copy all files to your ComfyUI custom_nodes directory")
        print("2. Restart ComfyUI")
        print("3. Check the settings in ComfyUI's UI")
        print("4. Test API endpoints with curl or browser")
        
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()