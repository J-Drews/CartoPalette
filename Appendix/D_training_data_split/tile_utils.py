"""
Utility functions for converting between lat/lon and tile coordinates.
Based on the Slippy Map tilenames convention (OSM standard).
"""

import math


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert latitude/longitude to tile x, y coordinates at a given zoom level.

    Args:
        lat: Latitude in degrees (-85.0511 to 85.0511)
        lon: Longitude in degrees (-180 to 180)
        zoom: Zoom level (0-20)

    Returns:
        Tuple of (x, y) tile coordinates
    """
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    # Clamp to valid range
    x = max(0, min(x, n - 1))
    y = max(0, min(y, n - 1))

    return x, y


def tile_to_latlon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Convert tile coordinates to latitude/longitude (top-left corner of tile).

    Args:
        x: Tile x coordinate
        y: Tile y coordinate
        zoom: Zoom level

    Returns:
        Tuple of (lat, lon) in degrees
    """
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def get_patch_tiles(center_x: int, center_y: int, zoom: int, grid_size: int = 3) -> list[tuple[int, int]]:
    """Get tile coordinates for a grid_size x grid_size patch centered on (center_x, center_y).

    Args:
        center_x: Center tile x coordinate
        center_y: Center tile y coordinate
        zoom: Zoom level
        grid_size: Size of the grid (default 3 for 3x3)

    Returns:
        List of (x, y) tile coordinates, row by row, top-left to bottom-right
    """
    n = 2 ** zoom
    offset = grid_size // 2

    tiles = []
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            tx = (center_x + dx) % n  # Wrap around horizontally
            ty = center_y + dy
            # Clamp y (no wrapping vertically)
            ty = max(0, min(ty, n - 1))
            tiles.append((tx, ty))

    return tiles
