import { app } from "../../scripts/app.js";

// Read-only readout of the resolution chosen by ImageRes2ModelRes, drawn
// inside the node's own boundary (no widget, nothing clickable).
// Matches the node by its inputs, so the registered name doesn't matter.
const EXTRA_HEIGHT = 26;

app.registerExtension({
    name: "riversidenodes.ImageRes2ModelRes.display",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const req = nodeData?.input?.required;
        if (!req || !req.resize_longest_side || !req.model || !req.interpolation_mode) return;

        // Reserve a strip at the bottom of the node for the readout.
        const computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const size = computeSize.apply(this, arguments);
            size[1] += EXTRA_HEIGHT;
            return size;
        };

        // Workflows saved before this change have a smaller stored size.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);

            // Workflows saved before the megapixels widget existed stored 3 values
            // [model, interpolation_mode, resize_longest_side]. Widget values are
            // positional, so the old longest-side value would land in megapixels.
            const saved = info?.widgets_values;
            if (Array.isArray(saved) && saved.length === 3) {
                const mp = this.widgets?.find((w) => w.name === "megapixels");
                const longest = this.widgets?.find((w) => w.name === "resize_longest_side");
                if (mp && longest) {
                    mp.value = 0;
                    longest.value = saved[2];
                }
            }

            const min = this.computeSize();
            if (this.size[1] < min[1]) this.setSize([this.size[0], min[1]]);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const text = message?.text?.[0];
            if (text === undefined) return;
            this._resText = text;
            app.graph.setDirtyCanvas(true, true);
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags?.collapsed || !this._resText) return;
            ctx.save();
            ctx.font = "13px sans-serif";
            ctx.fillStyle = window.LiteGraph?.NODE_TEXT_COLOR ?? "#aaa";
            ctx.textAlign = "center";
            ctx.textBaseline = "alphabetic";
            ctx.fillText(`Output: ${this._resText}`, this.size[0] / 2, this.size[1] - 9);
            ctx.restore();
        };
    },
});
