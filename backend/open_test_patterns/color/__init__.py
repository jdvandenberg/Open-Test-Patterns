"""Color engine for Open Test Patterns.

Thin, well-documented wrappers around :mod:`colour` (colour-science) that expose
exactly the operations the pattern generators need: color-space definitions,
matrix conversions, chromatic adaptation, and transfer functions.
"""

from .colorspaces import (
    COLOR_SPACES,
    ColorSpaceInfo,
    RGB_to_RGB,
    RGB_to_XYZ,
    XYZ_to_RGB,
    get_colorspace,
    list_colorspaces,
)
from .transfer import (
    TRANSFER_FUNCTIONS,
    TransferFunctionInfo,
    encode,
    full_to_legal,
    get_transfer_function,
    legal_to_full,
    list_transfer_functions,
)

__all__ = [
    "COLOR_SPACES",
    "ColorSpaceInfo",
    "RGB_to_RGB",
    "RGB_to_XYZ",
    "XYZ_to_RGB",
    "get_colorspace",
    "list_colorspaces",
    "TRANSFER_FUNCTIONS",
    "TransferFunctionInfo",
    "encode",
    "full_to_legal",
    "get_transfer_function",
    "legal_to_full",
    "list_transfer_functions",
]
