"""
Point density raster generation from LiDAR point clouds (points per m²).
"""

import logging
import numpy as np
from typing import Dict

logger = logging.getLogger('data_processor')


class DensityGenerator:

    def __init__(self, config: Dict):
        self.resolution = config['chm']['resolution']
        self.pixel_area = self.resolution ** 2

    def create_density(self, points: np.ndarray, grid_x: np.ndarray,
                      grid_y: np.ndarray) -> np.ndarray:
        """Create point density raster (points per square meter)."""
        logger.info("Creating density raster...")
        x, y = points[:, 0], points[:, 1]

        density = np.zeros(grid_x.shape, dtype=np.float32)

        pixel_i = ((y - np.min(grid_y)) / self.resolution).astype(int)
        pixel_j = ((x - np.min(grid_x)) / self.resolution).astype(int)

        valid = (
            (pixel_i >= 0) & (pixel_i < grid_x.shape[0]) &
            (pixel_j >= 0) & (pixel_j < grid_x.shape[1])
        )
        pixel_i = pixel_i[valid]
        pixel_j = pixel_j[valid]

        # Vectorized counting via flat index bincount
        flat_indices = pixel_i * grid_x.shape[1] + pixel_j
        counts = np.bincount(flat_indices, minlength=density.size)
        density = counts[:density.size].reshape(density.shape).astype(np.float32)

        density = density / self.pixel_area

        logger.info(f"Density raster created ({len(x):,} points)")
        return density
