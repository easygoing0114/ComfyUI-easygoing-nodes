try:
    from comfy_api.latest import ComfyAPI, ComfyExtension, io
    from .nodes import NODE_LIST, RENAMED_NODES

    _comfy_api = ComfyAPI()

    class EasygoingNodesExtension(ComfyExtension):
        async def on_load(self) -> None:

            for node_cls in RENAMED_NODES:
                await _comfy_api.node_replacement.register(io.NodeReplace(
                    new_node_id=node_cls.define_schema().node_id,
                    old_node_id=node_cls.NODE_ID_LEGACY,
                    old_widget_ids=list(node_cls.NODE_ID_INPUT_ORDER),
                ))

        async def get_node_list(self):
            return NODE_LIST

    async def comfy_entrypoint() -> EasygoingNodesExtension:
        return EasygoingNodesExtension()

    V3_AVAILABLE = True
except ImportError as e:

    import traceback
    print(
        f"EasygoingNodes: failed to import V3 node API or node modules "
        f"({e.__class__.__name__}: {e}). Falling back to legacy (no nodes "
        f"registered on this ComfyUI version if V3 is genuinely unavailable; "
        f"otherwise this indicates a bug in nodes/ that should be fixed, "
        f"see traceback below)."
    )
    traceback.print_exc()

    V3_AVAILABLE = False
    NODE_LIST = []

# Web extension definition.
WEB_DIRECTORY = "./web"

if V3_AVAILABLE:
    __all__ = [
        'WEB_DIRECTORY',
        'comfy_entrypoint',
    ]
else:
    print(
        "EasygoingNodes: comfy_api.latest (V3 node API) not found. "
        "This version of the pack requires a ComfyUI build with V3 node support; "
        "nodes will not be registered on this ComfyUI version."
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

    __all__ = [
        'NODE_CLASS_MAPPINGS',
        'NODE_DISPLAY_NAME_MAPPINGS',
        'WEB_DIRECTORY',
    ]