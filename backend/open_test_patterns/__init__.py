"""Open Test Patterns.

A modern, extensible generator of color-accurate test patterns for display
calibration and HDR / wide-gamut verification.
"""

import warnings as _warnings

# colour-science emits a ColourUsageWarning when optional plotting backends
# (Matplotlib) are absent. We intentionally do not depend on Matplotlib, so this
# is expected noise; silence it here, before any submodule imports `colour`.
_warnings.filterwarnings(
    "ignore",
    message=r'.*"Matplotlib" related API features are not available.*',
)

__version__ = "0.1.0"
__author__ = "JD Vandenberg"

__all__ = ["__version__", "__author__"]
