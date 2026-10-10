"""Compatibility imports for the standalone mask-editor submodule."""

from pathlib import Path
import sys


_SUBMODULE = Path(__file__).resolve().parents[1] / "mask-editor"
if not (_SUBMODULE / "mask_editor" / "__init__.py").is_file():
    raise ImportError("mask-editor is missing; run git submodule update --init mask-editor")
if str(_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE))

from mask_editor import MaskDocument, MaskEditor, editor_pages, launch_editor  # noqa: E402,F401
