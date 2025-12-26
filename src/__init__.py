"""
Volatility Surface Modeling Library

A comprehensive implementation of volatility surface models including:
- Dupire Local Volatility Model
- Heston Stochastic Volatility Model
- SABR Model

With arbitrage-free constraints, interpolation methods, and visualizations.
"""

from .volatility_surface import VolatilitySurface
from .dupire import DupireLocalVolatility
from .heston import HestonModel
from .sabr import SABRModel
from .arbitrage import ArbitrageChecker
from .interpolation import SurfaceInterpolator
from .visualization import VolatilitySurfaceVisualizer

__version__ = "1.0.0"
__all__ = [
    "VolatilitySurface",
    "DupireLocalVolatility",
    "HestonModel",
    "SABRModel",
    "ArbitrageChecker",
    "SurfaceInterpolator",
    "VolatilitySurfaceVisualizer",
]
