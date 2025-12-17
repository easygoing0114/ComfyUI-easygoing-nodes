// EasygoingNodes Settings - SDXL CLIP Only Version
let settingsInitialized = false;
let initialSyncComplete = false;

// 設定更新関数
window.updateEasygoingSetting = async (key, value) => {
    try {
        const currentSdxl = app.ui.settings.getSettingValue?.("EasygoingNodes.enable_sdxl_clip", true);

        const settings = {
            enable_sdxl_clip: currentSdxl
        };
        settings[key] = value;

        const response = await fetch("/easygoing_nodes/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings)
        });

        if (response.ok && initialSyncComplete) {
            showRestartNotification(key, value);
        }
    } catch (error) {
        // Silent error handling
    }
};

// 通知システム（設定変更時のみ表示）
window.showRestartNotification = (setting, value) => {
    const existing = document.querySelector('.easygoing-notification');
    if (existing) existing.remove();

    const notification = document.createElement('div');
    notification.className = 'easygoing-notification';
    notification.style.cssText = `
        position: fixed; top: 20px; right: 20px; z-index: 10000;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 16px 20px; border-radius: 10px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 14px; max-width: 350px;
        animation: slideIn 0.4s ease-out;
    `;

    const moduleName = setting.replace('enable_', '').replace('_', ' ').toUpperCase();
    const statusText = value ? 'Enabled' : 'Disabled';
    const statusIcon = value ? '⚠️' : '❌';

    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="font-size: 24px;">${statusIcon}</div>
            <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 15px; margin-bottom: 4px;">
                    ${moduleName} ${statusText}
                </div>
                <div style="font-size: 13px; opacity: 0.9;">
                    Restart ComfyUI to apply changes
                </div>
            </div>
            <button onclick="this.parentElement.parentElement.remove()"
                    style="background: rgba(255,255,255,0.2); border: none; color: white;
                           width: 28px; height: 28px; border-radius: 50%; cursor: pointer;
                           font-size: 16px;">×</button>
        </div>
    `;

    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 5000);
};

// ComfyUI準備完了待機
function waitForComfyUI() {
    if (typeof app !== 'undefined' && app.ui?.settings?.addSetting && !settingsInitialized) {
        addEasygoingSettings();
    } else if (!settingsInitialized) {
        setTimeout(waitForComfyUI, 300);
    }
}

// 設定項目追加（SDXL CLIPのみ）
function addEasygoingSettings() {
    if (settingsInitialized) return;

    try {
        addStyles();

        app.ui.settings.addSetting({
            id: "EasygoingNodes.enable_sdxl_clip",
            name: "🔧 Enable SDXL CLIP replacement",
            type: "boolean",
            defaultValue: true,
            tooltip: "Enable custom SDXL CLIP implementation (restart required)",
            onChange: (value) => window.updateEasygoingSetting("enable_sdxl_clip", value)
        });

        settingsInitialized = true;
        setTimeout(syncInitialSettings, 1000);
    } catch (error) {
        // Silent error handling
    }
}

// 初期設定同期（通知なし）
async function syncInitialSettings() {
    try {
        const settings = {
            enable_sdxl_clip: app.ui.settings.getSettingValue?.("EasygoingNodes.enable_sdxl_clip", true)
        };

        await fetch("/easygoing_nodes/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings)
        });

        // 初回同期完了フラグを設定
        initialSyncComplete = true;
    } catch (error) {
        // Silent error handling
        initialSyncComplete = true;
    }
}

// CSS追加
function addStyles() {
    if (document.querySelector('#easygoing-styles')) return;

    const style = document.createElement('style');
    style.id = 'easygoing-styles';
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .easygoing-notification:hover {
            transform: translateY(-2px);
        }
    `;
    document.head.appendChild(style);
}

// 初期化開始
waitForComfyUI();
