// workflowImageExport.js
//
// Note: ds.scale must stay at 1. Upscaling is done via ctx.setTransform(),
// not ds.scale, since changing ds.scale triggers zoom-dependent logic in
// litegraph and third-party extensions that breaks link rendering.

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

// ---------------------------------------------------------------------------
// DOM widgets (e.g. the multiline "text" widget used by CLIP Text Encode)
// are backed by a real HTMLTextAreaElement/HTMLInputElement that litegraph
// positions on top of the canvas with CSS. lgCanvas.draw() only paints the
// <canvas> 2D context, so it never picks up whatever is inside that
// overlaid DOM element. When the canvas is temporarily swapped for the
// virtual (offscreen) one during export, those widgets end up rendered as
// blank space.
//
// To fix this we temporarily force every DOM widget on the graph into a
// manual "draw as text on canvas" mode for the duration of the capture,
// then restore its normal behavior afterwards.
// ---------------------------------------------------------------------------

function isDomWidget(widget) {
  return !!(widget && (widget.element || widget.inputEl || widget.type === "customtext" || widget.type === "textarea"));
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

// Draws a DOM-backed widget's current text value directly onto the 2D
// context, mimicking litegraph's normal widget box styling closely enough
// to be legible in the exported image.
function drawDomWidgetFallback(ctx, node, widget, widgetWidth, y, height) {
  const text = getWidgetText(widget);
  const margin = 8;

  ctx.save();
  ctx.beginPath();
  const radius = 4;
  if (ctx.roundRect) {
    ctx.roundRect(margin, y, widgetWidth - margin * 2, height, radius);
  } else {
    ctx.rect(margin, y, widgetWidth - margin * 2, height);
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
  const maxLines = Math.max(1, Math.floor((height - padY * 2) / lineHeight));

  for (let i = 0; i < Math.min(lines.length, maxLines); i++) {
    ctx.fillText(lines[i], margin + padX, y + padY + i * lineHeight);
  }

  ctx.restore();
}

// Monkeypatches every DOM widget currently in the graph so that, instead of
// relying on its real (offscreen, un-swapped) DOM element, it paints its
// text value onto whatever canvas context it's given. Returns a restore
// function that undoes the patch.
function forceDomWidgetsToCanvasDraw() {
  const patched = [];

  for (const node of app.graph._nodes) {
    if (!node.widgets) continue;
    for (const widget of node.widgets) {
      if (!isDomWidget(widget)) continue;

      const originalDraw = widget.draw;
      const originalComputeSize = widget.computeSize ? widget.computeSize.bind(widget) : null;

      widget.draw = function (ctx, node, widgetWidth, y, height) {
        drawDomWidgetFallback(ctx, node, widget, widgetWidth, y, height);
      };

      patched.push({ widget, originalDraw, originalComputeSize });
    }
  }

  return function restore() {
    for (const { widget, originalDraw } of patched) {
      widget.draw = originalDraw;
    }
  };
}

class PngWorkflowImage {
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

    const dimLimit = PngWorkflowImage.MAX_CANVAS_DIMENSION;
    const pxLimit = PngWorkflowImage.MAX_CANVAS_PIXELS;

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

    // DOM-backed widgets (multiline text boxes, etc.) live outside the
    // canvas and won't be captured by draw(). Temporarily force them to
    // paint their text onto the 2D context instead, for this draw only.
    const restoreDomWidgets = forceDomWidgetsToCanvasDraw();
    try {
      lgCanvas.draw(true, true);
    } finally {
      restoreDomWidgets();
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
      PngWorkflowImage.crcTable ||
      (PngWorkflowImage.crcTable = (() => {
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
    const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
      const options = orig.apply(this, arguments);

      options.push(null, {
        content: "Workflow Image",
        submenu: {
          options: [
            {
              content: "Export PNG (embedded workflow)",
              callback: () => {
                new PngWorkflowImage().export(true);
              },
            },
            {
              content: `Export PNG (embedded workflow, Hires x${PngWorkflowImage.HIRES_SCALE})`,
              callback: () => {
                new PngWorkflowImage().export(true, PngWorkflowImage.HIRES_SCALE);
              },
            },
            {
              content: "Export PNG (no embedded workflow)",
              callback: () => {
                new PngWorkflowImage().export(false);
              },
            },
            {
              content: `Export PNG (no embedded workflow, Hires x${PngWorkflowImage.HIRES_SCALE})`,
              callback: () => {
                new PngWorkflowImage().export(false, PngWorkflowImage.HIRES_SCALE);
              },
            },
          ],
        },
      });

      return options;
    };
  },
});