"""CartoPalette — Context-Aware Color Palette Generation for Thematic Cartography."""
__version__ = "0.1.0"


def __getattr__(name):
    if name == "CartoPalette":
        from cartopalette.core.inference import CartoPalette
        return CartoPalette
    raise AttributeError(f"module 'cartopalette' has no attribute {name!r}")
