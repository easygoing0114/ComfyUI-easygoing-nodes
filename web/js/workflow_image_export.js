import { app } from "../../../scripts/app.js";

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

async function waitForNodeImages() {
  const promises = [];
  for (const node of app.graph._nodes) {
    for (const img of node.imgs || []) {
      if (!img || (img.complete && img.naturalWidth !== 0)) continue;
      promises.push(
        typeof img.decode === "function"
          ? img.decode().catch(() => {})
          : new Promise((resolve) => {
              img.addEventListener("load", () => resolve(), { once: true });
              img.addEventListener("error", () => resolve(), { once: true });
            })
      );
    }
  }
  if (promises.length) await Promise.all(promises);
}

const NATIVE_WIDGET_TYPES = new Set([
  "number", "combo", "toggle", "button", "slider", "text", "customtext",
  "boolean", "int", "float", "string", "seed", "color", "$$canvas-image-preview",
]);

function isPreviewPseudoWidget(widget) {
  if (!widget || widget.element || widget.inputEl) return false;
  const type = widget.type;
  if (typeof type !== "string" || !type || NATIVE_WIDGET_TYPES.has(type)) return false;
  return type.startsWith("$$") || /preview/i.test(type);
}

function isTextLikeElement(el) {
  const tag = el?.tagName?.toLowerCase();
  if (tag === "textarea") return true;
  if (tag !== "input") return false;
  return ["text", "search", "url"].includes((el.type || "text").toLowerCase());
}

function getWidgetText(widget, isPseudo) {
  if (!isPseudo) {
    const el = widget.inputEl || widget.element;
    if (typeof el?.value === "string") return el.value;
  }
  const v = isPseudo && Array.isArray(widget.value) ? widget.value[0] : widget.value;
  return v == null ? "" : String(v);
}

function wrapText(ctx, text, maxWidth) {
  const lines = [];
  for (const rawLine of String(text).split(/\r\n|\r|\n/)) {
    let current = "";
    for (const word of rawLine.split(" ")) {
      const candidate = current ? current + " " + word : word;
      if (current && ctx.measureText(candidate).width > maxWidth) {
        lines.push(current);
        current = word;
      } else {
        current = candidate;
      }
    }
    lines.push(current);
  }
  return lines;
}

const isLinkDriven = (widget, node) =>
  Array.isArray(node.inputs) && node.inputs.some((i) => i && i.name === widget.name && i.link != null);

function drawDomWidgetFallback(ctx, widget, widgetWidth, y, height) {
  const el = widget.inputEl || widget.element;
  const isPseudo = isPreviewPseudoWidget(widget);
  const text = getWidgetText(widget, isPseudo);

  const margin = 8, padX = 6, padY = 6, lineHeight = 14;
  const maxWidth = widgetWidth - margin * 2 - padX * 2;
  ctx.font = "12px Arial";

  let boxHeight = height;
  if (el) {
    boxHeight = Math.max(
      boxHeight,
      el.getBoundingClientRect?.().height || 0,
      el.offsetHeight || 0,
      el.scrollHeight || 0
    );
  } else {
    boxHeight = Math.max(boxHeight, wrapText(ctx, text, maxWidth).length * lineHeight + padY * 2);
  }

  ctx.save();
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(margin, y, widgetWidth - margin * 2, boxHeight, 4);
  else ctx.rect(margin, y, widgetWidth - margin * 2, boxHeight);
  ctx.fillStyle = "#1a1a1a";
  ctx.fill();
  ctx.strokeStyle = "#4a4a4a";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.clip();

  ctx.fillStyle = text ? "#dddddd" : "#777777";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  const lines = wrapText(ctx, text, maxWidth);
  const maxLines = Math.max(1, Math.floor((boxHeight - padY * 2) / lineHeight));
  for (let i = 0; i < Math.min(lines.length, maxLines); i++) {
    ctx.fillText(lines[i], margin + padX, y + padY + i * lineHeight);
  }
  ctx.restore();
}

function drawDomWidgetFallbacks(ctx) {
  const [ox, oy] = app.canvas?.ds?.offset || [0, 0];

  for (const node of app.graph._nodes) {
    if (!node.widgets || node.flags?.collapsed) continue;
    const laidOut = node.widgets
      .filter((w) => typeof w.last_y === "number")
      .sort((a, b) => a.last_y - b.last_y);

    for (const widget of laidOut) {
      const isPseudo = isPreviewPseudoWidget(widget);
      if (!widget.element && !isPseudo) continue;
      if (isLinkDriven(widget, node)) continue;
      if (!isPseudo && !isTextLikeElement(widget.inputEl || widget.element)) continue;

      const next = laidOut[laidOut.indexOf(widget) + 1];
      const height = Math.max(20, (next ? next.last_y - widget.last_y : 24) - 4);

      ctx.save();
      ctx.translate(ox, oy);
      ctx.translate(node.pos[0], node.pos[1]);
      drawDomWidgetFallback(ctx, widget, node.size[0], widget.last_y, height);
      ctx.restore();
    }
  }
}

class EasygoingPngWorkflowImage {
  static HIRES_SCALE = 4;
  static MAX_CANVAS_DIMENSION = 16000;
  static MAX_CANVAS_PIXELS = 250_000_000;

  getBounds() {
    const bounds = app.graph._nodes.reduce(
      (p, n) => {
        const b = typeof n.getBounding === "function" ? n.getBounding() : [0, 0, n.size[0], n.size[1]];
        p[0] = Math.min(p[0], n.pos[0]);
        p[1] = Math.min(p[1], n.pos[1]);
        p[2] = Math.max(p[2], n.pos[0] + b[2]);
        p[3] = Math.max(p[3], n.pos[1] + b[3]);
        return p;
      },
      [99999, 99999, -99999, -99999]
    );
    bounds[0] -= 50;
    bounds[1] -= 80;
    bounds[2] += 50;
    bounds[3] += 20;
    return bounds;
  }

  computeSafeScale(bounds, targetScale) {
    const w = bounds[2] - bounds[0];
    const h = bounds[3] - bounds[1];
    const { MAX_CANVAS_DIMENSION: dim, MAX_CANVAS_PIXELS: px } = EasygoingPngWorkflowImage;
    return Math.max(Math.min(targetScale, dim / w, dim / h, Math.sqrt(px / (w * h))), 1);
  }

  createVirtualCanvas(width, height) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    Object.assign(canvas.style, {
      width: width + "px",
      height: height + "px",
      position: "fixed",
      top: "0px",
      left: "-999999px",
      pointerEvents: "none",
      zIndex: "-1",
    });
    document.body.appendChild(canvas);
    this._virtualCanvasEl = canvas;
    return canvas;
  }

  removeVirtualCanvas() {
    this._virtualCanvasEl?.remove();
    this._virtualCanvasEl = null;
  }

  // Background, links and nodes must share one canvas/context, otherwise the
  // frontend draws links on a separate bgcanvas at the wrong scale.
  swapCanvasTarget(virtualCanvas) {
    const lg = app.canvas;
    this._orig = {
      canvas: lg.canvas,
      ctx: lg.ctx,
      dsElement: lg.ds.element,
      bgcanvas: lg.bgcanvas,
      bgctx: lg.bgctx,
    };
    const vctx = virtualCanvas.getContext("2d");
    lg.canvas = virtualCanvas;
    lg.ctx = vctx;
    lg.ds.element = virtualCanvas;
    lg.bgcanvas = virtualCanvas;
    lg.bgctx = vctx;
  }

  restoreCanvasTarget() {
    const lg = app.canvas;
    const o = this._orig;
    if (!o) return;
    lg.canvas = o.canvas;
    lg.ctx = o.ctx;
    lg.ds.element = o.dsElement;
    lg.bgcanvas = o.bgcanvas;
    lg.bgctx = o.bgctx;
    this._orig = null;
  }

  // Pin the visible area to the export bounds so nothing is culled.
  installRenderAllPatch(bounds, pixelWidth, pixelHeight) {
    const lg = app.canvas;
    const ds = lg.ds;
    this._restoreFns = [];

    const override = (obj, key, value) => {
      const ownDesc = Object.getOwnPropertyDescriptor(obj, key);
      try {
        Object.defineProperty(obj, key, {
          value,
          writable: true,
          configurable: true,
          enumerable: ownDesc ? ownDesc.enumerable : false,
        });
      } catch (e) {
        console.warn(`[WorkflowImageExport] could not override ${key}`, e);
        return;
      }
      this._restoreFns.push(() => {
        try {
          if (ownDesc) Object.defineProperty(obj, key, ownDesc);
          else delete obj[key];
        } catch (e) {
          console.warn(`[WorkflowImageExport] could not restore ${key}`, e);
        }
      });
    };

    const setVisibleArea = (target) => {
      const va = target?.visible_area;
      if (!va) return;
      va[0] = bounds[0];
      va[1] = bounds[1];
      va[2] = bounds[2] - bounds[0];
      va[3] = bounds[3] - bounds[1];
    };

    override(ds, "computeVisibleArea", function () {
      setVisibleArea(this);
    });
    setVisibleArea(ds);

    override(lg, "computeVisibleNodes", function (nodes, out) {
      const result = out || [];
      result.length = 0;
      for (const node of nodes || this.graph?._nodes || app.graph._nodes) {
        node.updateArea?.();
        result.push(node);
      }
      return result;
    });
    lg.visible_nodes = app.graph._nodes.slice();

    // A viewport also stops the frontend from resetting the context transform.
    this._origViewport = lg.viewport;
    try {
      lg.viewport = [0, 0, pixelWidth, pixelHeight];
    } catch (e) {}
  }

  uninstallRenderAllPatch() {
    for (const fn of (this._restoreFns || []).reverse()) fn();
    this._restoreFns = null;
    try {
      app.canvas.viewport = this._origViewport;
    } catch (e) {}
    this._origViewport = undefined;
  }

  saveState() {
    const lg = app.canvas;
    this.state = {
      scale: lg.ds.scale,
      offsetX: lg.ds.offset[0],
      offsetY: lg.ds.offset[1],
      clearBackground: lg.clear_background,
      clearBackgroundColor: lg.clear_background_color,
      backgroundImage: lg.background_image,
      transform: lg.ctx?.getTransform?.() ?? null,
    };
  }

  restoreState() {
    const lg = app.canvas;
    const s = this.state;
    lg.ds.scale = s.scale;
    lg.ds.offset[0] = s.offsetX;
    lg.ds.offset[1] = s.offsetY;
    lg.clear_background = s.clearBackground;
    lg.clear_background_color = s.clearBackgroundColor;
    lg.background_image = s.backgroundImage;
    if (s.transform) lg.ctx?.setTransform?.(s.transform);
    lg.dirty_canvas = true;
    lg.dirty_bgcanvas = true;
  }

  setBackgroundFill() {
    const lg = app.canvas;
    const bgColor =
      this.state.clearBackgroundColor ||
      window.getComputedStyle(document.body).getPropertyValue("--bg-color").trim() ||
      "#202020";
    lg.clear_background = true;
    lg.clear_background_color = bgColor;
    lg.background_image = null;
    lg._pattern = null;
    this._bgColor = bgColor;
  }

  // ds.scale stays 1; upscaling is done only via ctx.setTransform.
  updateView(bounds, scale) {
    const lg = app.canvas;
    lg.ds.scale = 1;
    lg.ds.offset[0] = -bounds[0];
    lg.ds.offset[1] = -bounds[1];
    lg.ctx?.setTransform?.(scale, 0, 0, scale, 0, 0);
    lg.dirty_canvas = true;
    lg.dirty_bgcanvas = true;
  }

  async drawAndSettle(virtualCanvas) {
    await waitForNodeImages();

    const ctx = app.canvas.ctx;
    if (ctx) {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, virtualCanvas.width, virtualCanvas.height);
      ctx.restore();
    }

    app.canvas.draw(true, true);
    if (ctx) drawDomWidgetFallbacks(ctx);

    await nextFrame();
    await nextFrame();
  }

  async export(includeWorkflow, targetScale = 1) {
    this.saveState();

    const bounds = this.getBounds();
    const scale = this.computeSafeScale(bounds, targetScale);
    const w = Math.ceil((bounds[2] - bounds[0]) * scale);
    const h = Math.ceil((bounds[3] - bounds[1]) * scale);
    const virtualCanvas = this.createVirtualCanvas(w, h);

    try {
      this.swapCanvasTarget(virtualCanvas);
      this.setBackgroundFill();
      this.installRenderAllPatch(bounds, w, h);
      this.updateView(bounds, scale);

      await this.drawAndSettle(virtualCanvas);

      const blob = await this.getBlob(
        virtualCanvas,
        includeWorkflow ? JSON.stringify(app.graph.serialize()) : undefined
      );
      this.download(blob);
    } finally {
      this.uninstallRenderAllPatch();
      this.restoreCanvasTarget();
      this.restoreState();
      this.removeVirtualCanvas();
      app.canvas.setDirty(true, true);
    }
  }

  download(blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    Object.assign(a, { href: url, download: "workflow.png", style: "display: none" });
    document.body.append(a);
    a.click();
    setTimeout(() => {
      a.remove();
      URL.revokeObjectURL(url);
    }, 0);
  }

  n2b(n) {
    return new Uint8Array([(n >> 24) & 0xff, (n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]);
  }

  joinArrayBuffer(...bufs) {
    const result = new Uint8Array(bufs.reduce((total, buf) => total + buf.byteLength, 0));
    let offset = 0;
    for (const buf of bufs) {
      result.set(buf, offset);
      offset += buf.byteLength;
    }
    return result;
  }

  crc32(data) {
    const table =
      EasygoingPngWorkflowImage.crcTable ||
      (EasygoingPngWorkflowImage.crcTable = Array.from({ length: 256 }, (_, n) => {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        return c;
      }));
    let crc = -1;
    for (let i = 0; i < data.byteLength; i++) {
      crc = (crc >>> 8) ^ table[(crc ^ data[i]) & 0xff];
    }
    return (crc ^ -1) >>> 0;
  }

  async getBlob(virtualCanvas, workflow) {
    const composite = document.createElement("canvas");
    composite.width = virtualCanvas.width;
    composite.height = virtualCanvas.height;
    const ctx = composite.getContext("2d");
    ctx.fillStyle = this._bgColor || "#202020";
    ctx.fillRect(0, 0, composite.width, composite.height);
    ctx.drawImage(virtualCanvas, 0, 0);

    let blob = await new Promise((resolve) => composite.toBlob(resolve, "image/png"));

    if (workflow) {
      const buffer = await blob.arrayBuffer();
      const png = new Uint8Array(buffer);
      const data = new TextEncoder().encode(`tEXtworkflow\0${workflow}`);
      const chunk = this.joinArrayBuffer(this.n2b(data.byteLength - 4), data, this.n2b(this.crc32(data)));
      const sz = new DataView(buffer).getUint32(8) + 20;
      blob = new Blob([this.joinArrayBuffer(png.subarray(0, sz), chunk, png.subarray(sz))], {
        type: "image/png",
      });
    }
    return blob;
  }
}

app.registerExtension({
  name: "easygoing.WorkflowImageExport",
  getCanvasMenuItems() {
    const scale = EasygoingPngWorkflowImage.HIRES_SCALE;
    const run = (embed) => () => new EasygoingPngWorkflowImage().export(embed, scale);
    return [
      null,
      {
        content: "Workflow Image",
        submenu: {
          options: [
            { content: `Export PNG (embedded workflow, Hires x${scale})`, callback: run(true) },
            { content: `Export PNG (no embedded workflow, Hires x${scale})`, callback: run(false) },
          ],
        },
      },
    ];
  },
});