"""
Test plot shapefile generation via interactive placement on a raster.

Lets the user place rectangular test regions on a raster by clicking.
Supports configurable plot dimensions, grid overlay, and snap-to-grid.
Output is a shapefile of test region polygons for use with the split
generator's geographic test split.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import box

logger = logging.getLogger('data_processor')


class TestPlotGenerator:
    """Creates test plot polygons interactively on a raster."""

    def __init__(self, config: Dict):
        self.plot_width = config.get('test_plots', {}).get('plot_width', 50.0)
        self.plot_height = config.get('test_plots', {}).get('plot_height', 50.0)
        self.grid_size = config.get('test_plots', {}).get('grid_size', 50.0)

    def create_plot(self, center_x: float, center_y: float,
                    width: float, height: float) -> box:
        """Create a rectangular plot polygon centered at (center_x, center_y)."""
        return box(
            center_x - width / 2, center_y - height / 2,
            center_x + width / 2, center_y + height / 2,
        )

    def snap_to_grid(self, x: float, y: float, grid_size: float,
                     origin_x: float, origin_y: float) -> Tuple[float, float]:
        """Snap coordinates to the center of the grid cell they fall in."""
        col = int((x - origin_x) / grid_size)
        row = int((y - origin_y) / grid_size)
        snapped_x = origin_x + (col + 0.5) * grid_size
        snapped_y = origin_y + (row + 0.5) * grid_size
        return snapped_x, snapped_y

    def plots_to_geodataframe(self, plots: List[dict], crs=None) -> gpd.GeoDataFrame:
        """Convert list of plot dicts to a GeoDataFrame.

        Each dict has keys: 'geometry', 'width', 'height'.
        """
        if not plots:
            return gpd.GeoDataFrame(columns=['width', 'height', 'geometry'], crs=crs)

        return gpd.GeoDataFrame(
            [{'width': p['width'], 'height': p['height']} for p in plots],
            geometry=[p['geometry'] for p in plots],
            crs=crs,
        )

    def save(self, plots_gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
        """Save test plots GeoDataFrame to shapefile."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plots_gdf.to_file(output_path)
        logger.info(f"Saved {len(plots_gdf)} test plot(s): {output_path}")
        return output_path
