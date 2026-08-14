from .color_nodes import NODE_LIST as COLOR_NODE_LIST
from .image_difference_checker_nodes import NODE_LIST as IMAGE_DIFF_NODE_LIST
from .merge_nodes import NODE_LIST as MERGE_NODE_LIST
from .model_load_save_nodes import NODE_LIST as MODEL_LOAD_SAVE_NODE_LIST
from .save_image_nodes import NODE_LIST as SAVE_IMAGE_NODE_LIST
from .text_encode_nodes import NODE_LIST as TEXT_ENCODE_NODE_LIST

NODE_LIST = [
    *COLOR_NODE_LIST,
    *IMAGE_DIFF_NODE_LIST,
    *MERGE_NODE_LIST,
    *MODEL_LOAD_SAVE_NODE_LIST,
    *SAVE_IMAGE_NODE_LIST,
    *TEXT_ENCODE_NODE_LIST,
]

RENAMED_NODES = [
    node_cls for node_cls in NODE_LIST if hasattr(node_cls, "NODE_ID_LEGACY")
]

__all__ = ["NODE_LIST", "RENAMED_NODES"]