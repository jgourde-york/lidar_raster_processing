"""
LiDAR intensity raster generation from first-return points, normalized to [0, 1].
"""

import logging
import numpy as np
from typing import Dict

logger = logging.getLogger('data_processor')


class IntensityGenerator:

    def __init__(self, config: Dict):
        self.resolution = config['chm']['resolution']

    def create_intensity(self, points: np.ndarray, grid_x: np.ndarray,
                        grid_y: np.ndarray) -> np.ndarray:
        """Create intensity raster from first returns, averaged per pixel and normalized to [0, 1]."""
        logger.info("Creating intensity raster...")
        x, y = points[:, 0], points[:, 1]
        intensity = points[:, 4]
        return_num = points[:, 5]

        # First returns provide the most consistent intensity values
        first_returns = return_num == 1
        x_first = x[first_returns]
        y_first = y[first_returns]
        intensity_first = intensity[first_returns]
        logger.debug(f"Using {np.sum(first_returns):,} first-return points")

        pixel_i = ((y_first - np.min(grid_y)) / self.resolution).astype(int)
        pixel_j = ((x_first - np.min(grid_x)) / self.resolution).astype(int)

        valid = (
            (pixel_i >= 0) & (pixel_i < grid_x.shape[0]) &
            (pixel_j >= 0) & (pixel_j < grid_x.shape[1])
        )
        pixel_i = pixel_i[valid]
        pixel_j = pixel_j[valid]
        intensity_valid = intensity_first[valid]

        # Vectorized mean intensity per pixel via flat index aggregation
        flat_indices = pixel_i * grid_x.shape[1] + pixel_j
        n_pixels = grid_x.shape[0] * grid_x.shape[1]
        sum_grid = np.bincount(flat_indices, weights=intensity_valid, minlength=n_pixels)
        count_grid = np.bincount(flat_indices, minlength=n_pixels)

        intensity_grid = np.full(n_pixels, np.nan, dtype=np.float32)
        has_data = count_grid[:n_pixels] > 0
        intensity_grid[has_data] = (sum_grid[:n_pixels][has_data] / count_grid[:n_pixels][has_data]).astype(np.float32)
        intensity_grid = intensity_grid.reshape(grid_x.shape)

        # Normalize to [0, 1]
        valid_mask = ~np.isnan(intensity_grid)
        if np.any(valid_mask):
            valid_intensities = intensity_grid[valid_mask]
            min_val = np.min(valid_intensities)
            max_val = np.max(valid_intensities)

            if max_val > min_val:
                intensity_grid = (intensity_grid - min_val) / (max_val - min_val)
            else:
                intensity_grid[valid_mask] = 0.5

        intensity_grid[np.isnan(intensity_grid)] = 0.0

        logger.info(f"Intensity raster created ({np.sum(first_returns):,} first returns)")
        return intensity_grid
