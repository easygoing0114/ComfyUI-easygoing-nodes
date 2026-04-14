import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
import folder_paths

class ImageDifferenceChecker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "ui_scale": ("FLOAT", {
                    "default": 2.8,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "dark_mode": ("BOOLEAN", {"default": True}),
                "show_difference_map": ("BOOLEAN", {"default": True}),
                "show_tone_analysis": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("color_diff_map", "grayscale_diff_map", "result_image", "result_text")
    FUNCTION = "compare_images"
    CATEGORY = "image/analysis"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # ------------------------------------------------------------------ #
    #  result_text 用 Markdown 表（Image1 / Image2 を別々に生成）          #
    # ------------------------------------------------------------------ #
    def _build_tone_table_markdown(self, np1, np2):
        """
        Image1 と Image2 のトーン情報をそれぞれ独立した Markdown 表として返す。
        """
        channels = ["Red", "Green", "Blue"]

        def make_table(np_img, title):
            header = f"**{title}**\n\n| Channel | Sum | Avg | % |\n|---------|-----|-----|---|"
            rows = []
            for i, ch in enumerate(channels):
                c = np_img[:, :, i] * 255
                pct = f"{np.mean(np_img[:, :, i]) * 100:.1f}%"
                rows.append(f"| {ch} | {np.sum(c):,.0f} | {np.mean(c):.1f} | {pct} |")
            return header + "\n" + "\n".join(rows)

        return make_table(np1, "Image 1") + "\n\n" + make_table(np2, "Image 2")

    # ------------------------------------------------------------------ #
    #  プレビュー画像用 ASCII 表（1枚の画像に対して生成）                    #
    # ------------------------------------------------------------------ #
    def _build_tone_table_ascii(self, np_img):
        """
        1枚の画像のトーン情報を ASCII 表形式の文字列として返す（プレビュー描画用）。
        列: Channel / Sum / Avg
        """
        channels = ["Red", "Green", "Blue"]
        rows = []
        for i, ch in enumerate(channels):
            c = np_img[:, :, i] * 255
            pct = f"{np.mean(np_img[:, :, i]) * 100:.1f}%"
            rows.append((ch, f"{np.sum(c):,.0f}", f"{np.mean(c):.1f}", pct))

        col_headers = ["Channel", "Sum", "Avg", "%"]
        col_w = [
            max(len(col_headers[0]), max(len(r[0]) for r in rows)),
            max(len(col_headers[1]), max(len(r[1]) for r in rows)),
            max(len(col_headers[2]), max(len(r[2]) for r in rows)),
            max(len(col_headers[3]), max(len(r[3]) for r in rows)),
        ]

        def hline(left, mid, right, fill="─"):
            return (left + fill * (col_w[0] + 2) + mid + fill * (col_w[1] + 2)
                    + mid + fill * (col_w[2] + 2) + mid + fill * (col_w[3] + 2) + right)

        def row_line(a, b, c, d):
            return f"│ {a:<{col_w[0]}} │ {b:>{col_w[1]}} │ {c:>{col_w[2]}} │ {d:>{col_w[3]}} │"

        lines = [
            hline("┌", "┬", "┐"),
            row_line(*col_headers),
            hline("├", "┼", "┤"),
        ]
        for ch, s, a, p in rows:
            lines.append(row_line(ch, s, a, p))
        lines.append(hline("└", "┴", "┘"))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  メイン処理                                                           #
    # ------------------------------------------------------------------ #
    def compare_images(self, image1: torch.Tensor, image2: torch.Tensor,
                       ui_scale: float, dark_mode: bool,
                       show_difference_map: bool, show_tone_analysis: bool):

        _SCALE = max(1.0, ui_scale)
        _PAD = int(24 * _SCALE)
        _GAP = int(16 * _SCALE)
        _LABEL_H = int(28 * _SCALE)
        _METRICS_H = int(72 * _SCALE)
        _LABEL_SIZE = int(14 * _SCALE)
        _STAT_SIZE = int(12 * _SCALE)

        def _get_font(size: int):
            for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf", "C:/Windows/Fonts/arial.ttf"):
                try:
                    return ImageFont.truetype(name, size)
                except (IOError, OSError):
                    pass
            return ImageFont.load_default()

        def _get_mono_font(size: int):
            for name in (
                "DejaVuSansMono.ttf",
                "DejaVuSansMono-Regular.ttf",
                "Courier New.ttf",
                "cour.ttf",
                "LiberationMono-Regular.ttf",
                "C:/Windows/Fonts/cour.ttf",
            ):
                try:
                    return ImageFont.truetype(name, size)
                except (IOError, OSError):
                    pass
            return ImageFont.load_default()

        img1 = image1[0]
        img2 = image2[0]
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 != h2 or w1 != w2:
            raise ValueError(f"Resolutions mismatch: {w1}x{h1} vs {w2}x{h2}")

        np1 = img1.cpu().numpy().astype(np.float32)
        np2 = img2.cpu().numpy().astype(np.float32)

        # 差分計算
        color_diff_np = np.abs(np1 - np2)[:, :, :3]
        sum_img1 = np1[:, :, 0] + np1[:, :, 1] + np1[:, :, 2]
        sum_img2 = np2[:, :, 0] + np2[:, :, 1] + np2[:, :, 2]
        g = np.clip(np.abs(sum_img1 - sum_img2) / 3.0, 0, 1)
        gray_diff_np = np.stack([g, g, g], axis=-1)

        # メトリクス
        mae_value = float(np.mean(np.abs(np1 - np2)) * 255)
        mae_similarity = (1.0 - mae_value / 255.0) * 100.0
        ssim_value = self._calculate_ssim(np1, np2)
        ssim_similarity = ssim_value * 100.0

        # result_text 生成
        mae_line  = f"MAE:  {mae_value:.1f}  (Similarity: {mae_similarity:.1f}%)"
        ssim_line = f"SSIM: {ssim_value:.3f}  (Similarity: {ssim_similarity:.1f}%)"
        result_text = f"### Difference\n\n{mae_line}  \n{ssim_line}"

        if show_tone_analysis:
            tone_md = self._build_tone_table_markdown(np1, np2)
            result_text += f"\n\n\n### Tone Analysis\n\n{tone_md}"

        bg_color = (27, 18, 18) if dark_mode else (255, 255, 255)
        text_color = (198, 204, 210) if dark_mode else (0, 0, 0)

        result_pil = self._build_result_image_detailed(
            np1, np2, color_diff_np, gray_diff_np,
            mae_value, mae_similarity, ssim_value, ssim_similarity,
            bg_color, text_color, show_difference_map, show_tone_analysis,
            _SCALE, _PAD, _GAP, _LABEL_H, _METRICS_H, _LABEL_SIZE, _STAT_SIZE,
            _get_font, _get_mono_font
        )

        result_np = np.array(result_pil).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        temp_dir = folder_paths.get_temp_directory()
        filename = f"diff_checker_{random.randint(0, 1000000)}.png"
        result_pil.save(os.path.join(temp_dir, filename), compress_level=4)

        return {
            "ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]},
            "result": (torch.from_numpy(color_diff_np).unsqueeze(0),
                       torch.from_numpy(gray_diff_np).unsqueeze(0),
                       result_tensor, result_text),
        }

    # ------------------------------------------------------------------ #
    #  トーンカーブ描画（RGB ヒストグラム）                                  #
    # ------------------------------------------------------------------ #
    def _draw_analysis_graph(self, np_img, width, bg_color, text_color, _SCALE, _STAT_SIZE):
        height = int(width * 0.71)
        graph_canvas = Image.new("RGB", (width, height), color=bg_color)
        overlay = Image.new("RGBA", (width, height), bg_color + (0,))
        draw_ov = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(graph_canvas)

        channels = ["Red", "Green", "Blue"]
        colors = [(220, 50, 50), (50, 180, 50), (50, 50, 220)]

        margin = int(10 * _SCALE)
        g_w = width - (margin * 2)
        g_h = height - (margin * 2) - int(20 * _SCALE)
        base_y = height - margin

        for i, col_name in enumerate(channels):
            c_data = np_img[:, :, i] * 255
            hist, _ = np.histogram(c_data, bins=256, range=(0, 256))
            hist_norm = hist / (hist.max() + 1e-5) * g_h

            points = [(margin + (x / 255) * g_w, base_y - hist_norm[x]) for x in range(256)]
            poly_points = [(margin, base_y)] + points + [(margin + g_w, base_y)]

            draw_ov.polygon(poly_points, fill=colors[i] + (51,))
            draw.line(points, fill=colors[i], width=max(1, int(_SCALE // 2)))

        graph_canvas.paste(overlay, (0, 0), overlay)
        return graph_canvas

    # ------------------------------------------------------------------ #
    #  ASCII トーン表描画                                                   #
    # ------------------------------------------------------------------ #
    def _draw_tone_table_ascii(self, np_img, width, bg_color, text_color,
                               _SCALE, _STAT_SIZE, _get_mono_font):
        """ASCII 表を PIL Image として描画して返す。"""
        ascii_str = self._build_tone_table_ascii(np_img)
        lines = ascii_str.split("\n")

        font = _get_mono_font(_STAT_SIZE)
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        char_bbox = dummy_draw.textbbox((0, 0), "W", font=font)
        char_h = char_bbox[3] - char_bbox[1]
        line_h = int(char_h * 1.6)

        pad_x = int(12 * _SCALE)
        content_h = line_h * len(lines)
        canvas_h = max(int(width * 0.35), content_h + int(20 * _SCALE))
        canvas = Image.new("RGB", (width, canvas_h), color=bg_color)
        draw = ImageDraw.Draw(canvas)

        start_y = (canvas_h - content_h) // 2
        for i, line in enumerate(lines):
            draw.text((pad_x, start_y + i * line_h), line, font=font, fill=text_color)

        return canvas

    # ------------------------------------------------------------------ #
    #  結果画像の構築                                                        #
    #  表示順: Input Image → Difference Map → MAE & SSIM                  #
    #           → Tone Curve → Tone Table                                  #
    # ------------------------------------------------------------------ #
    def _build_result_image_detailed(self, np1, np2, color_diff_np, gray_diff_np,
                                     mae_v, mae_s, ssim_v, ssim_s,
                                     bg_color, text_color, show_diff, show_tone,
                                     _SCALE, _PAD, _GAP, _LABEL_H, _METRICS_H,
                                     _LABEL_SIZE, _STAT_SIZE, _get_font, _get_mono_font):

        h, w = np1.shape[:2]
        f_label   = _get_font(_LABEL_SIZE)
        f_metrics = _get_font(_LABEL_SIZE)
        graph_h   = int(w * 0.71)

        # ASCII 表パネルの高さを事前計算
        ascii_panel_h = 0
        if show_tone:
            dummy = self._draw_tone_table_ascii(
                np1, w, bg_color, text_color,
                _SCALE, _STAT_SIZE, _get_mono_font
            )
            ascii_panel_h = dummy.height

        # キャンバス高さの計算
        # 1) Input images
        total_h = _PAD + _LABEL_H + h
        # 2) Difference maps（任意）
        if show_diff:
            total_h += _GAP + _LABEL_H + h
        # 3) MAE & SSIM（常に表示）
        total_h += _GAP + _METRICS_H
        # 4) Tone curve（任意）
        if show_tone:
            total_h += _GAP + _LABEL_H + graph_h
        # 5) Tone table（任意）
        if show_tone:
            total_h += _GAP + _LABEL_H + ascii_panel_h
        total_h += _PAD

        total_w = _PAD + w + _GAP + w + _PAD
        canvas = Image.new("RGB", (total_w, total_h), color=bg_color)
        draw = ImageDraw.Draw(canvas)

        def np_to_pil(arr):
            return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

        def paste_panel(img_obj, label, x, y, panel_w, panel_h):
            tw = draw.textlength(label, font=f_label)
            draw.text(
                (x + (panel_w - tw) / 2, y + (_LABEL_H - _LABEL_SIZE) // 2),
                label, font=f_label, fill=text_color
            )
            target_y = y + _LABEL_H
            if isinstance(img_obj, np.ndarray):
                canvas.paste(np_to_pil(img_obj), (x, target_y))
            else:
                canvas.paste(img_obj, (x, target_y))

        curr_y = _PAD

        # ── 1) Input images ──────────────────────────────────────────────
        paste_panel(np1, "Image 1", _PAD, curr_y, w, h)
        paste_panel(np2, "Image 2", _PAD + w + _GAP, curr_y, w, h)
        curr_y += _LABEL_H + h

        # ── 2) Difference maps ───────────────────────────────────────────
        if show_diff:
            curr_y += _GAP
            paste_panel(color_diff_np, "Color Difference", _PAD, curr_y, w, h)
            paste_panel(gray_diff_np, "Grayscale Difference", _PAD + w + _GAP, curr_y, w, h)
            curr_y += _LABEL_H + h

        # ── 3) MAE & SSIM ────────────────────────────────────────────────
        curr_y += _GAP
        left_x  = _PAD + w // 2
        right_x = _PAD + w + _GAP + w // 2

        mae_text  = f"MAE: {mae_v:.1f} (Similarity: {mae_s:.1f}%)"
        ssim_text = f"SSIM: {ssim_v:.3f} (Similarity: {ssim_s:.1f}%)"

        tw_mae  = draw.textlength(mae_text,  font=f_metrics)
        tw_ssim = draw.textlength(ssim_text, font=f_metrics)

        draw.text(
            (left_x  - tw_mae  / 2, curr_y + (_METRICS_H - _LABEL_SIZE) // 2),
            mae_text,  font=f_metrics, fill=text_color
        )
        draw.text(
            (right_x - tw_ssim / 2, curr_y + (_METRICS_H - _LABEL_SIZE) // 2),
            ssim_text, font=f_metrics, fill=text_color
        )
        curr_y += _METRICS_H

        # ── 4) Tone curve ────────────────────────────────────────────────
        if show_tone:
            curr_y += _GAP
            graph1 = self._draw_analysis_graph(
                np1, w, bg_color, text_color, _SCALE, _STAT_SIZE
            )
            graph2 = self._draw_analysis_graph(
                np2, w, bg_color, text_color, _SCALE, _STAT_SIZE
            )
            paste_panel(graph1, "Image 1 Tone Curve", _PAD, curr_y, w, graph_h)
            paste_panel(graph2, "Image 2 Tone Curve", _PAD + w + _GAP, curr_y, w, graph_h)
            curr_y += _LABEL_H + graph_h

        # ── 5) Tone table（ASCII）────────────────────────────────────────
        if show_tone:
            curr_y += _GAP
            table1 = self._draw_tone_table_ascii(
                np1, w, bg_color, text_color,
                _SCALE, _STAT_SIZE, _get_mono_font
            )
            table2 = self._draw_tone_table_ascii(
                np2, w, bg_color, text_color,
                _SCALE, _STAT_SIZE, _get_mono_font
            )
            paste_panel(table1, "Image 1 Tone Table", _PAD, curr_y, w, ascii_panel_h)
            paste_panel(table2, "Image 2 Tone Table", _PAD + w + _GAP, curr_y, w, ascii_panel_h)
            curr_y += _LABEL_H + ascii_panel_h

        return canvas

    # ------------------------------------------------------------------ #
    #  SSIM 計算                                                            #
    # ------------------------------------------------------------------ #
    def _calculate_ssim(self, img1, img2):
        C1, C2 = 0.0001, 0.0009
        g1 = 0.299 * img1[..., 0] + 0.587 * img1[..., 1] + 0.114 * img1[..., 2]
        g2 = 0.299 * img2[..., 0] + 0.587 * img2[..., 1] + 0.114 * img2[..., 2]
        mu1, mu2 = np.mean(g1), np.mean(g2)
        s1, s2 = np.var(g1), np.var(g2)
        s12 = np.mean(g1 * g2) - mu1 * mu2
        return (2 * mu1 * mu2 + C1) * (2 * s12 + C2) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2) + 1e-8)


NODE_CLASS_MAPPINGS = {"ImageDifferenceChecker": ImageDifferenceChecker}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageDifferenceChecker": "Image Difference Checker"}