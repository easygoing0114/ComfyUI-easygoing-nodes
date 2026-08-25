import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from comfy_api.latest import io, ui

# ---------------------------------------------------------------------------
# Font loading helpers
# ---------------------------------------------------------------------------

# Candidate font files tried in order; falls back to PIL's built-in default
# font if none of them are available on the host system.
_LABEL_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "Arial.ttf",
    "LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
_MONO_FONT_CANDIDATES = (
    "DejaVuSansMono.ttf",
    "DejaVuSansMono-Regular.ttf",
    "Courier New.ttf",
    "cour.ttf",
    "LiberationMono-Regular.ttf",
    "C:/Windows/Fonts/cour.ttf",
)


def _load_font(candidates: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont:
    """Try each candidate font filename in turn, falling back to PIL's
    built-in default bitmap font if none can be loaded."""
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _get_label_font(size: int) -> ImageFont.FreeTypeFont:
    return _load_font(_LABEL_FONT_CANDIDATES, size)


def _get_mono_font(size: int) -> ImageFont.FreeTypeFont:
    return _load_font(_MONO_FONT_CANDIDATES, size)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute a simple global (non-windowed) SSIM between two float32
    images in [0, 1], using luminance derived from Rec. 601 coefficients."""
    c1, c2 = 0.0001, 0.0009
    g1 = 0.299 * img1[..., 0] + 0.587 * img1[..., 1] + 0.114 * img1[..., 2]
    g2 = 0.299 * img2[..., 0] + 0.587 * img2[..., 1] + 0.114 * img2[..., 2]
    mu1, mu2 = np.mean(g1), np.mean(g2)
    var1, var2 = np.var(g1), np.var(g2)
    covar = np.mean(g1 * g2) - mu1 * mu2
    return (2 * mu1 * mu2 + c1) * (2 * covar + c2) / (
        (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2) + 1e-8
    )


# ---------------------------------------------------------------------------
# Tone tables (Markdown for the STRING output, ASCII for the report image)
# ---------------------------------------------------------------------------

_TONE_CHANNELS = ("Red", "Green", "Blue")


def _channel_tone_stats(np_img: np.ndarray) -> list[tuple[str, float, float, float]]:
    """Return (channel_name, sum, avg, mean_fraction) for each RGB channel
    of a float32 image in [0, 1], followed by an extra "Brightness" row
    using Rec. 601 luminance weights (0.299R + 0.587G + 0.114B), matching
    the luminance definition used by calculate_ssim()."""
    stats = []
    for i, name in enumerate(_TONE_CHANNELS):
        channel_255 = np_img[:, :, i] * 255
        stats.append((
            name,
            float(np.sum(channel_255)),
            float(np.mean(channel_255)),
            float(np.mean(np_img[:, :, i])),
        ))

    brightness = (
        0.299 * np_img[:, :, 0] + 0.587 * np_img[:, :, 1] + 0.114 * np_img[:, :, 2]
    )
    brightness_255 = brightness * 255.0
    stats.append((
        "Brightness",
        float(np.sum(brightness_255)),
        float(np.mean(brightness_255)),
        float(np.mean(brightness)),
    ))
    return stats


def build_tone_table_markdown(np1: np.ndarray, np2: np.ndarray) -> str:
    """Return per-channel tone statistics for both images as two
    independent Markdown tables (used in the STRING result_text output)."""

    def make_table(np_img: np.ndarray, title: str) -> str:
        header = f"**{title}**\n\n| Channel | Sum | Avg | % |\n|---------|-----|-----|---|"
        rows = [
            f"| {name} | {total:,.0f} | {avg:.1f} | {frac * 100:.1f}% |"
            for name, total, avg, frac in _channel_tone_stats(np_img)
        ]
        return header + "\n" + "\n".join(rows)

    return make_table(np1, "Image 1") + "\n\n" + make_table(np2, "Image 2")


def build_tone_table_ascii(np_img: np.ndarray) -> str:
    """Return the tone statistics of a single image as a monospace ASCII
    box-drawing table (used when rendering the composited report image)."""
    rows = [
        (name, f"{total:,.0f}", f"{avg:.1f}", f"{frac * 100:.1f}%")
        for name, total, avg, frac in _channel_tone_stats(np_img)
    ]

    col_headers = ("Channel", "Sum", "Avg", "%")
    col_w = [
        max(len(col_headers[c]), max(len(r[c]) for r in rows))
        for c in range(4)
    ]

    def hline(left: str, mid: str, right: str, fill: str = "─") -> str:
        segments = [fill * (w + 2) for w in col_w]
        return left + mid.join(segments) + right

    def row_line(a: str, b: str, c: str, d: str) -> str:
        return f"│ {a:<{col_w[0]}} │ {b:>{col_w[1]}} │ {c:>{col_w[2]}} │ {d:>{col_w[3]}} │"

    lines = [
        hline("┌", "┬", "┐"),
        row_line(*col_headers),
        hline("├", "┼", "┤"),
    ]
    lines.extend(row_line(*row) for row in rows)
    lines.append(hline("└", "┴", "┘"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report image rendering
# ---------------------------------------------------------------------------

def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def draw_tone_curve_graph(
    np_img: np.ndarray,
    width: int,
    bg_color: tuple[int, int, int],
    scale: float,
) -> Image.Image:
    """Render a filled RGB histogram ("tone curve") for one image, with
    the Rec. 601 brightness histogram drawn last (i.e. on top, in gray)."""
    height = int(width * 0.71)
    graph_canvas = Image.new("RGB", (width, height), color=bg_color)
    overlay = Image.new("RGBA", (width, height), bg_color + (0,))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(graph_canvas)

    colors = ((220, 50, 50), (50, 180, 50), (50, 50, 220))
    brightness_color = (170, 170, 170)

    margin = int(10 * scale)
    graph_w = width - (margin * 2)
    graph_h = height - (margin * 2) - int(20 * scale)
    base_y = height - margin

    def plot_channel(channel_255: np.ndarray, color: tuple[int, int, int], fill_alpha: int) -> None:
        hist, _ = np.histogram(channel_255, bins=256, range=(0, 256))
        hist_norm = hist / (hist.max() + 1e-5) * graph_h

        points = [
            (margin + (x / 255) * graph_w, base_y - hist_norm[x]) for x in range(256)
        ]
        poly_points = [(margin, base_y)] + points + [(margin + graph_w, base_y)]

        draw_overlay.polygon(poly_points, fill=color + (fill_alpha,))
        draw.line(points, fill=color, width=max(1, int(scale // 2)))

    for i in range(3):
        plot_channel(np_img[:, :, i] * 255, colors[i], 51)

    # Rec. 601 brightness, drawn last so it sits on top of the RGB curves.
    brightness_255 = (
        0.299 * np_img[:, :, 0] + 0.587 * np_img[:, :, 1] + 0.114 * np_img[:, :, 2]
    ) * 255
    plot_channel(brightness_255, brightness_color, 60)

    graph_canvas.paste(overlay, (0, 0), overlay)
    return graph_canvas


def draw_tone_table_ascii_panel(
    np_img: np.ndarray,
    width: int,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    scale: float,
    stat_font_size: int,
) -> Image.Image:
    """Render the ASCII tone table for one image as a standalone panel."""
    ascii_str = build_tone_table_ascii(np_img)
    lines = ascii_str.split("\n")

    font = _get_mono_font(stat_font_size)
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    char_bbox = dummy_draw.textbbox((0, 0), "W", font=font)
    char_h = char_bbox[3] - char_bbox[1]
    line_h = int(char_h * 1.6)

    pad_x = int(12 * scale)
    content_h = line_h * len(lines)
    canvas_h = max(int(width * 0.35), content_h + int(20 * scale))
    canvas = Image.new("RGB", (width, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)

    start_y = (canvas_h - content_h) // 2
    for i, line in enumerate(lines):
        draw.text((pad_x, start_y + i * line_h), line, font=font, fill=text_color)

    return canvas


def build_report_image(
    np1: np.ndarray,
    np2: np.ndarray,
    color_diff_np: np.ndarray,
    brightness_diff_np: np.ndarray,
    mae_value: float,
    mae_similarity: float,
    ssim_value: float,
    ssim_similarity: float,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    show_original: bool,
    show_diff: bool,
    show_tone: bool,
    scale: float,
) -> Image.Image:
    """Compose the full report image.

    Section order (each optional except MAE/SSIM, which is always shown):
        1. Input images (Image 1 / Image 2)
        2. Difference maps (Color / Brightness [Rec.601])
        3. MAE & SSIM metrics
        4. Tone curve (RGB + Rec.601 brightness histogram) per image
        5. Tone table (ASCII) per image
    """
    pad = int(24 * scale)
    gap = int(16 * scale)
    label_h = int(28 * scale)
    metrics_h = int(72 * scale)
    label_size = int(14 * scale)
    stat_size = int(12 * scale)

    h, w = np1.shape[:2]
    label_font = _get_label_font(label_size)
    metrics_font = _get_label_font(label_size)
    graph_h = int(w * 0.71)

    # Pre-compute the ASCII table panel height (needed for canvas sizing).
    ascii_panel_h = 0
    if show_tone:
        ascii_panel_h = draw_tone_table_ascii_panel(
            np1, w, bg_color, text_color, scale, stat_size
        ).height

    # --- Canvas height ---
    total_h = pad
    if show_original:
        total_h += label_h + h
    if show_diff:
        if show_original:
            total_h += gap
        total_h += label_h + h
    if show_original or show_diff:
        total_h += gap
    total_h += metrics_h
    if show_tone:
        total_h += gap + label_h + graph_h
        total_h += gap + ascii_panel_h
    total_h += pad

    total_w = pad + w + gap + w + pad
    canvas = Image.new("RGB", (total_w, total_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)

    def paste_panel(img_obj, label: str, x: int, y: int, panel_w: int) -> None:
        text_w = draw.textlength(label, font=label_font)
        draw.text(
            (x + (panel_w - text_w) / 2, y + (label_h - label_size) // 2),
            label, font=label_font, fill=text_color,
        )
        target_y = y + label_h
        pil_img = _np_to_pil(img_obj) if isinstance(img_obj, np.ndarray) else img_obj
        canvas.paste(pil_img, (x, target_y))

    curr_y = pad
    left_x = pad
    right_x = pad + w + gap

    # 1) Input images
    if show_original:
        paste_panel(np1, "Image 1", left_x, curr_y, w)
        paste_panel(np2, "Image 2", right_x, curr_y, w)
        curr_y += label_h + h

    # 2) Difference maps
    if show_diff:
        if show_original:
            curr_y += gap
        paste_panel(color_diff_np, "Color Difference", left_x, curr_y, w)
        paste_panel(brightness_diff_np, "Brightness Difference (Rec.601)", right_x, curr_y, w)
        curr_y += label_h + h

    # 3) MAE & SSIM (always shown)
    if show_original or show_diff:
        curr_y += gap
    mae_text = f"MAE: {mae_value:.1f} (Similarity: {mae_similarity:.1f}%)"
    ssim_text = f"SSIM: {ssim_value:.3f} (Similarity: {ssim_similarity:.1f}%)"
    mae_w = draw.textlength(mae_text, font=metrics_font)
    ssim_w = draw.textlength(ssim_text, font=metrics_font)
    metrics_y = curr_y + (metrics_h - label_size) // 2
    draw.text((left_x + w // 2 - mae_w / 2, metrics_y), mae_text, font=metrics_font, fill=text_color)
    draw.text((right_x + w // 2 - ssim_w / 2, metrics_y), ssim_text, font=metrics_font, fill=text_color)
    curr_y += metrics_h

    # 4) Tone curve
    if show_tone:
        curr_y += gap
        graph1 = draw_tone_curve_graph(np1, w, bg_color, scale)
        graph2 = draw_tone_curve_graph(np2, w, bg_color, scale)
        paste_panel(graph1, "Image 1 Tone Curve", left_x, curr_y, w)
        paste_panel(graph2, "Image 2 Tone Curve", right_x, curr_y, w)
        curr_y += label_h + graph_h

        # 5) Tone table (ASCII)
        curr_y += gap
        table1 = draw_tone_table_ascii_panel(np1, w, bg_color, text_color, scale, stat_size)
        table2 = draw_tone_table_ascii_panel(np2, w, bg_color, text_color, scale, stat_size)
        canvas.paste(table1, (left_x, curr_y))
        canvas.paste(table2, (right_x, curr_y))

    return canvas


# ---------------------------------------------------------------------------
# Node definition (V3)
# ---------------------------------------------------------------------------

class ImageDifferenceChecker(io.ComfyNode):

    NODE_ID_LEGACY = "ImageDifferenceChecker"
    NODE_ID_INPUT_ORDER = (
        "image1",
        "image2",
        "ui_scale",
        "dark_mode",
        "show_original_image",
        "show_difference_map",
        "show_tone_analysis",
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasygoingNodes_ImageDifferenceChecker",
            display_name="Image Difference Checker",
            category="image/analysis",
            description=(
                "Compares two same-resolution images and reports their "
                "differences: a color diff map, a Rec.601 brightness diff "
                "map, a composited visual report (MAE/SSIM + optional tone "
                "analysis), and a Markdown text summary."
            ),
            inputs=[
                io.Image.Input("image1", tooltip="First image to compare."),
                io.Image.Input("image2", tooltip="Second image to compare."),
                io.Float.Input(
                    "ui_scale",
                    default=3.2,
                    min=1.0,
                    max=8.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Scale factor for the composited report image "
                    "(labels, padding, graphs).",
                ),
                io.Boolean.Input(
                    "dark_mode",
                    default=True,
                    tooltip="Use a dark background for the report image.",
                ),
                io.Boolean.Input(
                    "show_original_image",
                    default=True,
                    tooltip="Include the two input images in the report.",
                ),
                io.Boolean.Input(
                    "show_difference_map",
                    default=True,
                    tooltip="Include the color/grayscale difference maps "
                    "in the report.",
                ),
                io.Boolean.Input(
                    "show_tone_analysis",
                    default=False,
                    tooltip="Include per-channel tone-curve graphs and tone "
                    "tables in the report and in the text summary.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="color_diff_map"),
                io.Image.Output(display_name="brightness_diff_map"),
                io.Image.Output(display_name="result_image"),
                io.String.Output(display_name="result_text"),
            ],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(
        cls,
        image1: torch.Tensor,
        image2: torch.Tensor,
        ui_scale: float,
        dark_mode: bool,
        show_original_image: bool,
        show_difference_map: bool,
        show_tone_analysis: bool,
    ) -> io.NodeOutput:
        img1, img2 = image1[0], image2[0]
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if h1 != h2 or w1 != w2:
            raise ValueError(f"Resolutions mismatch: {w1}x{h1} vs {w2}x{h2}")

        np1 = img1.cpu().numpy().astype(np.float32)
        np2 = img2.cpu().numpy().astype(np.float32)

        # Difference maps.
        color_diff_np = np.abs(np1 - np2)[:, :, :3]
        brightness1 = 0.299 * np1[:, :, 0] + 0.587 * np1[:, :, 1] + 0.114 * np1[:, :, 2]
        brightness2 = 0.299 * np2[:, :, 0] + 0.587 * np2[:, :, 1] + 0.114 * np2[:, :, 2]
        brightness_diff = np.clip(np.abs(brightness1 - brightness2), 0, 1)
        brightness_diff_np = np.stack([brightness_diff, brightness_diff, brightness_diff], axis=-1)

        # Metrics.
        mae_value = float(np.mean(np.abs(np1 - np2)) * 255)
        mae_similarity = (1.0 - mae_value / 255.0) * 100.0
        ssim_value = calculate_ssim(np1, np2)
        ssim_similarity = ssim_value * 100.0

        # Text summary.
        result_text = (
            f"### Difference\n\n"
            f"MAE:  {mae_value:.8f}  (Similarity: {mae_similarity:.1f}%)  \n"
            f"SSIM: {ssim_value:.8f}  (Similarity: {ssim_similarity:.1f}%)"
        )
        if show_tone_analysis:
            tone_md = build_tone_table_markdown(np1, np2)
            result_text += f"\n\n\n### Tone Analysis\n\n{tone_md}"

        # Composited report image.
        scale = max(1.0, ui_scale)
        bg_color = (18, 27, 18) if dark_mode else (255, 255, 255)
        text_color = (198, 204, 210) if dark_mode else (0, 0, 0)
        report_pil = build_report_image(
            np1, np2, color_diff_np, brightness_diff_np,
            mae_value, mae_similarity, ssim_value, ssim_similarity,
            bg_color, text_color,
            show_original_image, show_difference_map, show_tone_analysis,
            scale,
        )
        report_tensor = torch.from_numpy(
            np.array(report_pil).astype(np.float32) / 255.0
        ).unsqueeze(0)

        color_diff_tensor = torch.from_numpy(color_diff_np).unsqueeze(0)
        brightness_diff_tensor = torch.from_numpy(brightness_diff_np).unsqueeze(0)

        return io.NodeOutput(
            color_diff_tensor,
            brightness_diff_tensor,
            report_tensor,
            result_text,
            ui=ui.PreviewImage(report_tensor, cls=cls),
        )

NODE_LIST = [
    ImageDifferenceChecker,
]