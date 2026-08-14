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

// ComfyUI frontend's own DOM widget detection
// (src/scripts/domWidget.ts, isDOMWidget) checks only `element`.
// `inputEl` is a deprecated legacy alias for the same DOM element and is
// never set without `element` also being set, so checking it separately
// only widens the match without adding coverage. Checking `widget.type`
// is unreliable: node authors are free to use "customtext"/"textarea" as
// a *canvas* widget type, and doing so previously caused non-DOM widgets
// (e.g. toggle/combo widgets on custom nodes) to be misidentified as DOM
// widgets, which made drawDomWidgetFallback paint an empty black box over
// them instead of letting their normal canvas draw() run.
function isDomWidget(widget) {
  return !!(widget && widget.element);
}

// ComfyUI_frontend's "text preview" pseudo-widgets -- e.g. PreviewAny
// ("Preview as Text", widget name "textPreview", widget.type ===
// "textPreview") and GetImageSize's live progress-text overlay
// (widget name "progressText", widget.type === "$$node-text-preview",
// populated via send_progress_text / PromptServer) -- are NOT
// registered through node.addDOMWidget(). They never get a
// `.element`/`.inputEl` at all; ComfyUI_frontend paints them through
// a separate overlay layer keyed off widget.type and widget.value
// directly, bypassing the normal canvas draw() path entirely.
//
// These pseudo-widget type names are NOT consistently prefixed --
// "textPreview" has no "$$" prefix while "$$node-text-preview" does
// -- so we can't key off a naming convention. Instead we treat any
// widget with no backing element and a type that isn't one of
// litegraph's known *canvas-drawn* primitive widget types (number,
// combo, toggle, button, text/customtext for plain text inputs
// without an element, etc.) as a candidate, then require it to
// actually carry displayable value content. This stays conservative:
// ordinary canvas-native widgets (number/combo/toggle/button/slider)
// are explicitly excluded so we never accidentally intercept their
// normal draw().
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
  // Require an explicit draw()/computeSize() override or a "$$"-style
  // internal type name -- both are signals this widget renders itself
  // outside litegraph's normal per-type switch (ComfyWidgets registry)
  // rather than just being an unrecognized-but-still-canvas-drawn type.
  const looksInternal = widget.type.startsWith("$$") || /preview/i.test(widget.type);
  return looksInternal;
}

function getPseudoWidgetText(widget) {
  if (typeof widget.value === "string") return widget.value;
  if (Array.isArray(widget.value)) {
    // GetImageSize/PreviewAny sometimes store the display string as
    // the sole element of a tuple/array (mirrors the Python side's
    // `{"ui": {"text": (value,)}}` convention).
    const first = widget.value[0];
    if (typeof first === "string") return first;
  }
  if (widget.value != null) return String(widget.value);
  return "";
}

// Only textarea / text-like input elements represent an editable text
// value we can meaningfully render as a text box. Other DOM widgets
// (e.g. audio players, upload buttons, custom previews) have an
// `element` too, but drawing them as an empty bordered text box is
// misleading, so we skip those instead of drawing a placeholder.
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

// Since ComfyUI_frontend v1.16, widgets and their equivalent input
// sockets simply coexist on a node -- there is no more "conversion"
// step, and widget.type never becomes "converted-widget"
// (Comfy.WidgetInputs' convertWidgetToInput is now just a deprecated
// no-op stub). So a widget like `text` on CLIP Text Encode (with
// Offload) being driven by a Concatenate Text node still shows up in
// node.widgets, completely indistinguishable by `type` from a normal,
// freely-editable widget like `value` on Text (Multiline).
//
// litegraph's own internal visibility/layout logic has a known
// mismatch here too (Comfy-Org/ComfyUI_frontend#10276): computeSize()
// calls isWidgetVisible(), while the actual widget layout pass only
// filters on `hidden` -- so even litegraph itself can reserve blank
// space for a widget whose real content isn't shown. We can't rely on
// isWidgetVisible()/hidden/computedDisabled for the same reason noted
// below (they reflect live-canvas/zoom state, not link-driven state,
// and previously misfired on Text (Multiline)'s `value` widget).
//
// The one thing that's actually true and stable regardless of canvas
// state is the link itself: a widget is link-driven if there is an
// input slot with the same name AND that slot has an active link
// (inp.link !== null/undefined). This mirrors how ComfyUI_frontend's
// own Vue widget layer determines this (see
// isInputConnected(getWidgetInputIndex(widget)) in NodeWidgets.vue,
// Comfy-Org/ComfyUI_frontend#5692). Matching on name alone (without
// checking `link`) is what caused the earlier false positive on
// Text (Multiline): that node happened to have a same-named, but
// unconnected, input slot.
function isWidgetHidden(widget, node) {
  if (!widget) return true;
  if (!Array.isArray(node.inputs)) return false;
  return node.inputs.some((inp) => inp && inp.name === widget.name && inp.link != null);
}

function drawDomWidgetFallback(ctx, node, widget, widgetWidth, y, height) {
  const el = widget.inputEl || widget.element;
  const isPseudo = isTextPreviewPseudoWidget(widget);

  if (isWidgetHidden(widget, node)) {
    console.log("[WorkflowImageExport] skip (link-driven):", node.title, widget.name, {
      inputs: Array.isArray(node.inputs) ? node.inputs.map((i) => ({ name: i && i.name, link: i && i.link })) : null,
    });
    return;
  }

  if (!isPseudo && !isTextLikeElement(el)) {
    console.log("[WorkflowImageExport] skip (not text-like element):", node.title, widget.name, {
      type: widget.type,
      tagName: el && el.tagName,
      hasElement: !!widget.element,
      hasInputEl: !!widget.inputEl,
    });
    return;
  }

  const text = isPseudo ? getPseudoWidgetText(widget) : getWidgetText(widget);
  const inputsInfo = Array.isArray(node.inputs)
    ? node.inputs.map((inp) => ({
        name: inp && inp.name,
        hasWidget: !!(inp && inp.widget),
        widgetIsSameRef: !!(inp && inp.widget && inp.widget === widget),
        widgetName: inp && inp.widget && inp.widget.name,
        link: inp && inp.link,
      }))
    : null;
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
    // No backing element to measure. These Vue overlay widgets size
    // themselves to their text content client-side; litegraph's
    // passed-in `height` is just its own layout placeholder and is
    // often too short (e.g. 1 line) for multi-line preview text.
    // Estimate from the actual text so multi-line values (Preview as
    // Text on a long prompt) aren't clipped to one line.
    const padY = 6;
    const lineHeight = 14;
    const maxWidth = widgetWidth - 8 * 2 - 6 * 2;
    ctx.font = "12px Arial";
    const estimatedLines = wrapText(ctx, text || "", maxWidth).length || 1;
    const estimatedHeight = estimatedLines * lineHeight + padY * 2;
    if (estimatedHeight > boxHeight) boxHeight = estimatedHeight;
  }

  console.log("[WorkflowImageExport] drawing:", node.title, widget.name, {
    textLength: text.length,
    text: text.slice(0, 80),
    tagName: el ? el.tagName : "(none - pseudo-widget)",
    elementValue: el && typeof el.value === "string" ? el.value.slice(0, 80) : el && el.value,
    widgetValue: typeof widget.value === "string" ? widget.value.slice(0, 80) : widget.value,
    passedInHeight: height,
    resolvedBoxHeight: boxHeight,
    inputsInfo,
  });
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

function forceDomWidgetsToCanvasDraw() {
  const patched = [];

  for (const node of app.graph._nodes) {
    if (!node.widgets) continue;
    for (const widget of node.widgets) {
      const isPseudo = isTextPreviewPseudoWidget(widget);

      if (!isDomWidget(widget) && !isPseudo) {
        console.log("[WorkflowImageExport] not a DOM widget:", node.title, widget.name, widget.type);
        continue;
      }
      // Link-driven widgets (a same-named input slot with an active
      // link): leave them alone. Their real value comes over the link,
      // not from typing into the DOM element, so patching draw() on them
      // would only ever show stale/empty content.
      if (isWidgetHidden(widget, node)) {
        console.log("[WorkflowImageExport] patch-skip (link-driven):", node.title, widget.name);
        continue;
      }
      // Non text-holding DOM widgets (audio players, upload buttons,
      // previews, etc) are left with their normal draw() so they don't
      // lose their own rendering to an unconditional no-op/fallback.
      // Pseudo-widgets (Preview as Text / Get Image Size's progress
      // text) have no element to check here -- they're identified by
      // widget.type instead, already confirmed via isPseudo above.
      const el = widget.inputEl || widget.element;
      if (!isPseudo && !isTextLikeElement(el)) {
        console.log("[WorkflowImageExport] patch-skip (not text-like):", node.title, widget.name, {
          tagName: el && el.tagName,
        });
        continue;
      }

      console.log(
        "[WorkflowImageExport] patching widget:",
        node.title,
        widget.name,
        isPseudo ? widget.type : el.tagName
      );
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