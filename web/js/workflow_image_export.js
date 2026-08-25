import { app } from "../../../scripts/app.js";

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function waitForNodeImages() {
  const promises = [];
  for (const node of app.graph._nodes) {
    const imgs = node.imgs;
    if (!imgs || !imgs.length) continue;
    for (const img of imgs) {
      if (!img) continue;
      if (img.complete && img.naturalWidth !== 0) continue;
      if (typeof img.decode === "function") {
        promises.push(img.decode().catch(() => {}));
      } else {
        promises.push(
          new Promise((resolve) => {
            img.addEventListener("load", () => resolve(), { once: true });
            img.addEventListener("error", () => resolve(), { once: true });
          })
        );
      }
    }
  }
  if (promises.length) {
    await Promise.all(promises);
  }
}

// A widget is DOM-backed if it has an `element` (ComfyUI frontend's own
// isDOMWidget check). `inputEl` is a deprecated alias for the same thing.
function isDomWidget(widget) {
  return !!(widget && widget.element);
}

// ComfyUI_frontend's "text preview" pseudo-widgets (e.g. PreviewAny's
// "textPreview", GetImageSize's "$$node-text-preview") have no
// `.element`/`.inputEl` -- they're painted via a separate overlay layer
// keyed off widget.type/widget.value, bypassing canvas draw() entirely.
// Their type names aren't consistently prefixed, so we treat any widget
// with no backing element and a type outside litegraph's known
// canvas-drawn primitives as a candidate pseudo-widget.
const KNOWN_CANVAS_NATIVE_WIDGET_TYPES = new Set([
  "number",
  "combo",
  "toggle",
  "button",
  "slider",
  "text",
  "customtext",
  "boolean",
  "int",
  "float",
  "string",
  "seed",
  "color",
  "$$canvas-image-preview",
]);

function isTextPreviewPseudoWidget(widget) {
  if (!widget || widget.element || widget.inputEl) return false;
  if (typeof widget.type !== "string" || !widget.type) return false;
  if (KNOWN_CANVAS_NATIVE_WIDGET_TYPES.has(widget.type)) return false;
  const looksInternal = widget.type.startsWith("$$") || /preview/i.test(widget.type);
  return looksInternal;
}

function getPseudoWidgetText(widget) {
  if (typeof widget.value === "string") return widget.value;
  if (Array.isArray(widget.value)) {
    // Some nodes store the display string as the sole element of a
    // tuple/array (mirrors the Python side's `{"ui": {"text": (value,)}}`).
    const first = widget.value[0];
    if (typeof first === "string") return first;
  }
  if (widget.value != null) return String(widget.value);
  return "";
}

// Only textarea / text-like input elements represent an editable text
// value worth rendering; other DOM widgets (audio players, upload
// buttons, etc.) are skipped rather than drawn as an empty placeholder.
function isTextLikeElement(el) {
  if (!el || typeof el.tagName !== "string") return false;
  const tag = el.tagName.toLowerCase();
  if (tag === "textarea") return true;
  if (tag === "input") {
    const inputType = (el.type || "text").toLowerCase();
    return inputType === "text" || inputType === "search" || inputType === "url";
  }
  return false;
}

function getWidgetText(widget) {
  if (widget.inputEl && typeof widget.inputEl.value === "string") return widget.inputEl.value;
  if (widget.element && typeof widget.element.value === "string") return widget.element.value;
  if (typeof widget.value === "string") return widget.value;
  if (widget.value != null) return String(widget.value);
  return "";
}

function wrapText(ctx, text, maxWidth) {
  const lines = [];
  const rawLines = String(text).split(/\r\n|\r|\n/);
  for (const rawLine of rawLines) {
    if (rawLine === "") {
      lines.push("");
      continue;
    }
    let current = "";
    for (const word of rawLine.split(" ")) {
      const candidate = current ? current + " " + word : word;
      if (ctx.measureText(candidate).width > maxWidth && current) {
        lines.push(current);
        current = word;
      } else {
        current = candidate;
      }
    }
    if (current) lines.push(current);
  }
  return lines;
}

// A widget is link-driven (its real value comes from a connected input,
// not from the widget itself) if there's an input slot with the same
// name that has an active link.
function isWidgetHidden(widget, node) {
  if (!widget) return true;
  if (!Array.isArray(node.inputs)) return false;
  return node.inputs.some((inp) => inp && inp.name === widget.name && inp.link != null);
}

function drawDomWidgetFallback(ctx, node, widget, widgetWidth, y, height) {
  const el = widget.inputEl || widget.element;
  const isPseudo = isTextPreviewPseudoWidget(widget);

  if (isWidgetHidden(widget, node)) return;
  if (!isPseudo && !isTextLikeElement(el)) return;

  const text = isPseudo ? getPseudoWidgetText(widget) : getWidgetText(widget);

  // The `height` litegraph passes into draw() for a DOM widget is only
  // its internal placeholder/layout height, not the actual rendered size
  // of the textarea (which is sized by CSS, independent of litegraph's
  // canvas layout). Using that placeholder height here under-counts
  // maxLines and silently clips multi-line text down to (often) a single
  // line. The textarea's own rendered box is the source of truth for how
  // tall this widget actually looks on screen, so prefer that.
  let boxHeight = height;
  if (el) {
    if (typeof el.getBoundingClientRect === "function") {
      const rect = el.getBoundingClientRect();
      if (rect.height > boxHeight) boxHeight = rect.height;
    }
    if (typeof el.offsetHeight === "number" && el.offsetHeight > boxHeight) {
      boxHeight = el.offsetHeight;
    }
    if (typeof el.scrollHeight === "number" && el.scrollHeight > boxHeight) {
      boxHeight = el.scrollHeight;
    }
  } else if (isPseudo) {
    // No backing element to measure; estimate height from the text
    // itself so multi-line preview values aren't clipped to one line.
    const padY = 6;
    const lineHeight = 14;
    const maxWidth = widgetWidth - 8 * 2 - 6 * 2;
    ctx.font = "12px Arial";
    const estimatedLines = wrapText(ctx, text || "", maxWidth).length || 1;
    const estimatedHeight = estimatedLines * lineHeight + padY * 2;
    if (estimatedHeight > boxHeight) boxHeight = estimatedHeight;
  }

  const margin = 8;

  ctx.save();
  ctx.beginPath();
  const radius = 4;
  if (ctx.roundRect) {
    ctx.roundRect(margin, y, widgetWidth - margin * 2, boxHeight, radius);
  } else {
    ctx.rect(margin, y, widgetWidth - margin * 2, boxHeight);
  }
  ctx.fillStyle = "#1a1a1a";
  ctx.fill();
  ctx.strokeStyle = "#4a4a4a";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.clip();
  ctx.fillStyle = text ? "#dddddd" : "#777777";
  ctx.font = "12px Arial";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";

  const padX = 6;
  const padY = 6;
  const lineHeight = 14;
  const maxWidth = widgetWidth - margin * 2 - padX * 2;
  const lines = wrapText(ctx, text || "", maxWidth);
  const maxLines = Math.max(1, Math.floor((boxHeight - padY * 2) / lineHeight));

  for (let i = 0; i < Math.min(lines.length, maxLines); i++) {
    ctx.fillText(lines[i], margin + padX, y + padY + i * lineHeight);
  }

  ctx.restore();
}

// Collects the DOM-backed (and pseudo-DOM) text widgets that need a
// manual fallback paint for this export. litegraph's normal draw() pass
// leaves DOM widgets blank on canvas but still records each one's
// on-screen position via `widget.last_y`; we read that back afterwards
// rather than patching any widget's draw() method.
function collectDomWidgetsToRender() {
  const targets = [];

  for (const node of app.graph._nodes) {
    if (!node.widgets || node.flags?.collapsed) continue;
    for (const widget of node.widgets) {
      const isPseudo = isTextPreviewPseudoWidget(widget);
      if (!isDomWidget(widget) && !isPseudo) continue;
      if (isWidgetHidden(widget, node)) continue;

      const el = widget.inputEl || widget.element;
      if (!isPseudo && !isTextLikeElement(el)) continue;
      if (typeof widget.last_y !== "number") continue;

      targets.push({ node, widget });
    }
  }

  return targets;
}

// Paints the fallback box for each collected widget directly, using
// litegraph's own recorded layout (widget.last_y, node.size) for
// position and width. Runs after the normal lgCanvas.draw() pass, as a
// pure additional paint step -- it never reads or writes widget.draw.
function drawDomWidgetFallbacks(ctx, targets) {
  // litegraph's draw() positions each node via ds.offset (graph pan, set
  // in updateView()) plus a per-node translate to node.pos; widget.last_y
  // is recorded relative to both. Both transforms are undone by the time
  // draw() returns, so we reapply them here. ds.scale is left out: the
  // exporter pins it to 1 and scales via ctx.setTransform() instead.
  const dsOffset = app.canvas?.ds?.offset || [0, 0];

  // Use the gap to the next widget on the same node (by last_y order) as
  // a layout-height estimate; drawDomWidgetFallback() refines this using
  // the element's actual rendered size when available.
  for (const { node, widget } of targets) {
    const siblings = (node.widgets || [])
      .filter((w) => typeof w.last_y === "number")
      .sort((a, b) => a.last_y - b.last_y);
    const idx = siblings.indexOf(widget);
    const next = idx >= 0 ? siblings[idx + 1] : null;
    const fallbackGap = next ? next.last_y - widget.last_y : 24;
    const height = Math.max(20, fallbackGap - 4);

    const widgetWidth = node.size[0];
    const y = widget.last_y;

    ctx.save();
    ctx.translate(dsOffset[0], dsOffset[1]);
    ctx.translate(node.pos[0], node.pos[1]);
    drawDomWidgetFallback(ctx, node, widget, widgetWidth, y, height);
    ctx.restore();
  }
}

class EasygoingPngWorkflowImage {
  extension = "png";

  static HIRES_SCALE = 4;
  static MAX_CANVAS_DIMENSION = 16000;
  static MAX_CANVAS_PIXELS = 250_000_000;

  getBounds() {
    const bounds = app.graph._nodes.reduce(
      (p, n) => {
        if (n.pos[0] < p[0]) p[0] = n.pos[0];
        if (n.pos[1] < p[1]) p[1] = n.pos[1];
        const b = typeof n.getBounding === "function" ? n.getBounding() : [0, 0, n.size[0], n.size[1]];
        const r = n.pos[0] + b[2];
        const bo = n.pos[1] + b[3];
        if (r > p[2]) p[2] = r;
        if (bo > p[3]) p[3] = bo;
        return p;
      },
      [99999, 99999, -99999, -99999]
    );

    // Margin so links don't get clipped at the edges
    bounds[0] -= 50;
    bounds[1] -= 80;
    bounds[2] += 50;
    bounds[3] += 20;

    return bounds;
  }

  // Clamp scale so the canvas doesn't exceed browser pixel limits
  computeSafeScale(bounds, targetScale) {
    const w = bounds[2] - bounds[0];
    const h = bounds[3] - bounds[1];

    const dimLimit = EasygoingPngWorkflowImage.MAX_CANVAS_DIMENSION;
    const pxLimit = EasygoingPngWorkflowImage.MAX_CANVAS_PIXELS;

    const dimScaleLimit = Math.min(dimLimit / w, dimLimit / h);
    const pxScaleLimit = Math.sqrt(pxLimit / (w * h));

    const scale = Math.min(targetScale, dimScaleLimit, pxScaleLimit);
    return Math.max(scale, 1);
  }

  // Real (but offscreen) canvas, so litegraph's DOM-dependent calculations
  // (getBoundingClientRect etc.) still work correctly.
  createVirtualCanvas(pixelWidth, pixelHeight) {
    const canvas = document.createElement("canvas");
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;

    canvas.style.width = pixelWidth + "px";
    canvas.style.height = pixelHeight + "px";
    canvas.style.position = "fixed";
    canvas.style.top = "0px";
    canvas.style.left = "-999999px";
    canvas.style.pointerEvents = "none";
    canvas.style.zIndex = "-1";

    document.body.appendChild(canvas);
    this._virtualCanvasEl = canvas;

    return canvas;
  }

  removeVirtualCanvas() {
    if (this._virtualCanvasEl && this._virtualCanvasEl.parentNode) {
      this._virtualCanvasEl.parentNode.removeChild(this._virtualCanvasEl);
    }
    this._virtualCanvasEl = null;
  }

  swapCanvasTarget(virtualCanvas) {
    const lgCanvas = app.canvas;

    this._originalCanvasEl = lgCanvas.canvas;
    this._originalCtx = lgCanvas.ctx;
    this._originalDsElement = lgCanvas.ds.element;

    const vctx = virtualCanvas.getContext("2d");

    lgCanvas.canvas = virtualCanvas;
    lgCanvas.ctx = vctx;
    lgCanvas.ds.element = virtualCanvas;

    return vctx;
  }

  restoreCanvasTarget() {
    const lgCanvas = app.canvas;
    lgCanvas.canvas = this._originalCanvasEl;
    lgCanvas.ctx = this._originalCtx;
    lgCanvas.ds.element = this._originalDsElement;
    this._originalCanvasEl = null;
    this._originalCtx = null;
    this._originalDsElement = null;
  }

  saveState() {
    const lgCanvas = app.canvas;
    this.state = {
      scale: lgCanvas.ds.scale,
      offsetX: lgCanvas.ds.offset[0],
      offsetY: lgCanvas.ds.offset[1],
      clearBackground: lgCanvas.clear_background,
      clearBackgroundColor: lgCanvas.clear_background_color,
      backgroundImage: lgCanvas.background_image,
      transform: lgCanvas.ctx?.getTransform ? lgCanvas.ctx.getTransform() : null,
    };
  }

  restoreState() {
    const lgCanvas = app.canvas;
    lgCanvas.ds.scale = this.state.scale;
    lgCanvas.ds.offset[0] = this.state.offsetX;
    lgCanvas.ds.offset[1] = this.state.offsetY;
    lgCanvas.clear_background = this.state.clearBackground;
    lgCanvas.clear_background_color = this.state.clearBackgroundColor;
    lgCanvas.background_image = this.state.backgroundImage;
    if (this.state.transform && lgCanvas.ctx?.setTransform) {
      lgCanvas.ctx.setTransform(this.state.transform);
    }
    lgCanvas.dirty_canvas = true;
    lgCanvas.dirty_bgcanvas = true;
  }

  setBackgroundFill() {
    const lgCanvas = app.canvas;
    lgCanvas.clear_background = true;
    const bgColor =
      this.state.clearBackgroundColor ||
      window.getComputedStyle(document.body).getPropertyValue("--bg-color").trim() ||
      "#202020";
    lgCanvas.clear_background_color = bgColor;
    lgCanvas.background_image = null;
    lgCanvas._pattern = null;
    this._bgColor = bgColor;
  }

  // ds.scale stays at 1; upscaling happens only via ctx.setTransform.
  updateView(bounds, scale) {
    const lgCanvas = app.canvas;

    lgCanvas.ds.scale = 1;
    lgCanvas.ds.offset[0] = -bounds[0];
    lgCanvas.ds.offset[1] = -bounds[1];

    if (lgCanvas.ctx?.setTransform) {
      lgCanvas.ctx.setTransform(scale, 0, 0, scale, 0, 0);
    }

    lgCanvas.dirty_canvas = true;
    lgCanvas.dirty_bgcanvas = true;
  }

  async drawAndSettle(virtualCanvas) {
    await waitForNodeImages();

    const lgCanvas = app.canvas;
    const ctx = lgCanvas.ctx;
    if (ctx) {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, virtualCanvas.width, virtualCanvas.height);
      ctx.restore();
    }

    // DOM-backed widgets live outside the canvas and won't be captured
    // by draw(); paint a fallback box for them afterwards.
    lgCanvas.draw(true, true);
    const domWidgetTargets = collectDomWidgetsToRender();
    if (domWidgetTargets.length && ctx) {
      drawDomWidgetFallbacks(ctx, domWidgetTargets);
    }

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
      this.updateView(bounds, scale);

      await this.drawAndSettle(virtualCanvas);

      const blob = await this.getBlob(
        virtualCanvas,
        includeWorkflow ? JSON.stringify(app.graph.serialize()) : undefined
      );

      this.download(blob);
    } finally {
      this.restoreCanvasTarget();
      this.restoreState();
      this.removeVirtualCanvas();
      app.canvas.setDirty(true, true);
    }
  }

  download(blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    Object.assign(a, {
      href: url,
      download: "workflow." + this.extension,
      style: "display: none",
    });
    document.body.append(a);
    a.click();
    setTimeout(function () {
      a.remove();
      window.URL.revokeObjectURL(url);
    }, 0);
  }

  n2b(n) {
    return new Uint8Array([(n >> 24) & 0xff, (n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]);
  }

  joinArrayBuffer(...bufs) {
    const result = new Uint8Array(bufs.reduce((totalSize, buf) => totalSize + buf.byteLength, 0));
    bufs.reduce((offset, buf) => {
      result.set(buf, offset);
      return offset + buf.byteLength;
    }, 0);
    return result;
  }

  crc32(data) {
    const crcTable =
      EasygoingPngWorkflowImage.crcTable ||
      (EasygoingPngWorkflowImage.crcTable = (() => {
        let c;
        const crcTable = [];
        for (let n = 0; n < 256; n++) {
          c = n;
          for (let k = 0; k < 8; k++) {
            c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
          }
          crcTable[n] = c;
        }
        return crcTable;
      })());

    let crc = 0 ^ -1;
    for (let i = 0; i < data.byteLength; i++) {
      crc = (crc >>> 8) ^ crcTable[(crc ^ data[i]) & 0xff];
    }
    return (crc ^ -1) >>> 0;
  }

  async canvasToBlob(canvas) {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/png");
    });
  }

  async getBlob(virtualCanvas, workflow) {
    // Composite onto an opaque background to avoid transparency
    const bgColor = this._bgColor || "#202020";

    const compositeCanvas = document.createElement("canvas");
    compositeCanvas.width = virtualCanvas.width;
    compositeCanvas.height = virtualCanvas.height;
    const ctx = compositeCanvas.getContext("2d");

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, compositeCanvas.width, compositeCanvas.height);
    ctx.drawImage(virtualCanvas, 0, 0);

    let blob = await this.canvasToBlob(compositeCanvas);

    if (workflow) {
      const buffer = await blob.arrayBuffer();
      const typedArr = new Uint8Array(buffer);
      const view = new DataView(buffer);
      const data = new TextEncoder().encode(`tEXtworkflow\0${workflow}`);
      const chunk = this.joinArrayBuffer(this.n2b(data.byteLength - 4), data, this.n2b(this.crc32(data)));
      const sz = view.getUint32(8) + 20;
      const result = this.joinArrayBuffer(typedArr.subarray(0, sz), chunk, typedArr.subarray(sz));
      blob = new Blob([result], { type: "image/png" });
    }

    return blob;
  }
}

app.registerExtension({
  name: "easygoing.WorkflowImageExport",
  setup() {
    // Add a menu entry by wrapping getCanvasMenuOptions on the shared
    // prototype, always calling through to the original implementation.
    if (typeof LGraphCanvas?.prototype?.getCanvasMenuOptions !== "function") {
      console.warn("[WorkflowImageExport] LGraphCanvas.getCanvasMenuOptions not found; menu entry not added.");
      return;
    }
    const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
      const options = orig.apply(this, arguments);

      options.push(null, {
        content: "Workflow Image",
        submenu: {
          options: [
            {
              content: `Export PNG (embedded workflow, Hires x${EasygoingPngWorkflowImage.HIRES_SCALE})`,
              callback: () => {
                new EasygoingPngWorkflowImage().export(true, EasygoingPngWorkflowImage.HIRES_SCALE);
              },
            },
            {
              content: `Export PNG (no embedded workflow, Hires x${EasygoingPngWorkflowImage.HIRES_SCALE})`,
              callback: () => {
                new EasygoingPngWorkflowImage().export(false, EasygoingPngWorkflowImage.HIRES_SCALE);
              },
            },
          ],
        },
      });

      return options;
    };
  },
});