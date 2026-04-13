"""
nodes パッケージ
各カテゴリのノードを個別ファイルで管理し、NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS を統合してエクスポートする。

  color_nodes.py                  : 色調節ノード
  save_nodes.py                   : セーブ用ノード
  merge_nodes.py                  : マージ用ノード
  text_encode_nodes.py            : テキストエンコードノード
  image_difference_checker_nodes.py: 画像差分チェッカーノード
"""

from .color_nodes import (
    NODE_CLASS_MAPPINGS as COLOR_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COLOR_DISPLAY_MAPPINGS,
)
from .save_nodes import (
    NODE_CLASS_MAPPINGS as SAVE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SAVE_DISPLAY_MAPPINGS,
)
from .merge_nodes import (
    NODE_CLASS_MAPPINGS as MERGE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MERGE_DISPLAY_MAPPINGS,
)
from .text_encode_nodes import (
    NODE_CLASS_MAPPINGS as TEXT_ENCODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as TEXT_ENCODE_DISPLAY_MAPPINGS,
)
from .image_difference_checker_nodes import (
    NODE_CLASS_MAPPINGS as IMAGE_DIFF_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as IMAGE_DIFF_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    **COLOR_CLASS_MAPPINGS,
    **SAVE_CLASS_MAPPINGS,
    **MERGE_CLASS_MAPPINGS,
    **TEXT_ENCODE_CLASS_MAPPINGS,
    **IMAGE_DIFF_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **COLOR_DISPLAY_MAPPINGS,
    **SAVE_DISPLAY_MAPPINGS,
    **MERGE_DISPLAY_MAPPINGS,
    **TEXT_ENCODE_DISPLAY_MAPPINGS,
    **IMAGE_DIFF_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]