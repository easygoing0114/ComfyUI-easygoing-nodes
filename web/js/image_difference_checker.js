import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ImageDifferenceChecker.RefreshButton",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "ImageDifferenceChecker") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const btn = this.addWidget(
                "button",
                "🔄 Refresh",
                "refresh",
                () => {
                    app.queuePrompt(0, 1);
                }
            );

            btn.serialize = false;
        };
    },
});