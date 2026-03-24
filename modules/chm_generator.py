"""
CHM, DSM, and DTM raster generation from LiDAR point clouds.
"""

import logging
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, gaussian_filter
from typing import Dict, Tuple

logger = logging.getLogger('data_processor')


class CHMGenerator:

    # Maximum distance (meters) to fill DTM gaps via nearest-neighbor interpolation
    MAX_GAP_FILL_DISTANCE = 10.0

    def __init__(self, config: Dict):
        self.resolution = config['chm']['resolution']
        self.min_height = config['chm']['min_height']
        self.max_height = config['chm']['max_height']
        self.dtm_method = config['chm']['dtm_method']
        self.use_ground_class = config['chm']['use_ground_class']
        self.seed = config['chm'].get('seed', 42)

        smoothing = config['chm'].get('smoothing', {})
        self.smoothing_enabled = smoothing.get('enabled', False)
        self.smoothing_sigma = smoothing.get('sigma', 2)
        self.smoothing_rescale = smoothing.get('rescale', 'p95')
        self.smoothing_rescale_percentile = smoothing.get('rescale_percentile', 95)

    def _extract_ground_points(self, points: np.ndarray,
                               max_points: int = 50000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract and subsample ground points from a point cloud.

        Uses ground-classified points (class 2) if available, otherwise
        falls back to 10th percentile method.
        """
        x, y, z, classification = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

        if self.use_ground_class:
            ground_mask = classification == 2
            num_ground = np.sum(ground_mask)
            if num_ground > 10:
                logger.debug(f"Using {num_ground:,} ground-classified points")
                ground_x = x[ground_mask]
                ground_y = y[ground_mask]
                ground_z = z[ground_mask]
            else:
                logger.warning("Insufficient ground-classified points, using 10th percentile")
                ground_x, ground_y, ground_z = self._percentile_ground(x, y, z)
        else:
            logger.debug("Using 10th percentile for ground estimation")
            ground_x, ground_y, ground_z = self._percentile_ground(x, y, z)

        if len(ground_x) > max_points:
            logger.debug(f"Subsampling ground points to {max_points:,}")
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(ground_x), max_points, replace=False)
            ground_x = ground_x[indices]
            ground_y = ground_y[indices]
            ground_z = ground_z[indices]

        return ground_x, ground_y, ground_z

    def create_dtm(self, points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        """Create Digital Terrain Model by interpolating ground points to a regular grid."""
        logger.info("Creating DTM...")

        num_pixels = grid_x.size
        max_points = min(num_pixels // 4, 50000)
        ground_x, ground_y, ground_z = self._extract_ground_points(
            points, max_points=max_points
        )

        logger.debug("Interpolating ground points to grid")
        points_2d = np.column_stack([ground_x, ground_y])
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        dtm_flat = griddata(
            points_2d, ground_z, grid_points,
            method=self.dtm_method, fill_value=np.nan
        )
        dtm = dtm_flat.reshape(grid_x.shape)

        # Fill small gaps (within ~10m of data) with nearest neighbor
        if np.any(np.isnan(dtm)):
            logger.debug("Filling small gaps with nearest neighbor")
            data_mask = ~np.isnan(dtm)
            max_gap_pixels = int(self.MAX_GAP_FILL_DISTANCE / self.resolution)
            dilated_mask = binary_dilation(data_mask, iterations=max_gap_pixels)
            small_gaps_mask = np.isnan(dtm) & dilated_mask

            if np.sum(small_gaps_mask) > 0:
                dtm_filled = griddata(
                    points_2d, ground_z, grid_points,
                    method='nearest'
                ).reshape(grid_x.shape)
                dtm[small_gaps_mask] = dtm_filled[small_gaps_mask]

            dtm[np.isnan(dtm)] = 0.0

        logger.info(f"DTM created ({len(ground_x):,} ground points)")
        return dtm

    def _percentile_ground(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          percentile: float = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate ground elevation using per-cell percentile method (1m cells)."""
        cell_size = 1.0
        x_min, x_max = np.min(x), np.max(x)
        y_min = np.min(y)

        x_idx = ((x - x_min) / cell_size).astype(np.int32)
        y_idx = ((y - y_min) / cell_size).astype(np.int32)

        nx = int(np.ceil((x_max - x_min) / cell_size)) + 1
        cell_idx = y_idx * nx + x_idx

        unique_cells, _, counts = np.unique(cell_idx, return_inverse=True, return_counts=True)

        sort_idx = np.argsort(cell_idx)
        z_sorted = z[sort_idx]

        cumsum = np.zeros(len(counts) + 1, dtype=np.int64)
        cumsum[1:] = np.cumsum(counts)

        ground_z_vals = np.zeros(len(unique_cells))
        for i in range(len(unique_cells)):
            start, end = cumsum[i], cumsum[i + 1]
            ground_z_vals[i] = np.percentile(z_sorted[start:end], percentile)

        cell_y_idx = unique_cells // nx
        cell_x_idx = unique_cells % nx
        ground_x = x_min + (cell_x_idx + 0.5) * cell_size
        ground_y = y_min + (cell_y_idx + 0.5) * cell_size

        return ground_x, ground_y, ground_z_vals

    def interpolate_ground_elevation(self, points: np.ndarray,
                                     query_xy: np.ndarray) -> np.ndarray:
        """Interpolate ground elevation at arbitrary XY locations for 3D point
        cloud height normalization (subtracting ground from each point's Z).

        Unlike create_dtm() which produces a 2D raster grid, this returns
        elevations at scattered query points. Processes in chunks of 500K
        to manage memory.
        """
        ground_x, ground_y, ground_z = self._extract_ground_points(points)
        ground_2d = np.column_stack([ground_x, ground_y])

        chunk_size = 500_000
        n_points = len(query_xy)
        ground_elev = np.empty(n_points, dtype=np.float64)

        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            chunk_xy = query_xy[start:end]

            chunk_elev = griddata(
                ground_2d, ground_z, chunk_xy,
                method=self.dtm_method, fill_value=np.nan
            )

            nan_mask = np.isnan(chunk_elev)
            if np.any(nan_mask):
                chunk_elev[nan_mask] = griddata(
                    ground_2d, ground_z, chunk_xy[nan_mask],
                    method='nearest'
                )

            ground_elev[start:end] = chunk_elev

        return ground_elev

    def create_dsm(self, points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        """Create Digital Surface Model (max elevation per pixel)."""
        logger.info("Creating DSM...")
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        dsm = np.full(grid_x.shape, -np.inf, dtype=np.float32)

        logger.debug(f"DSM from {len(x):,} points")
        pixel_i = ((y - grid_y[0, 0]) / self.resolution).astype(int)
        pixel_j = ((x - grid_x[0, 0]) / self.resolution).astype(int)

        valid = (
            (pixel_i >= 0) & (pixel_i < grid_x.shape[0]) &
            (pixel_j >= 0) & (pixel_j < grid_x.shape[1])
        )
        pixel_i = pixel_i[valid]
        pixel_j = pixel_j[valid]
        z_valid = z[valid]

        flat_indices = pixel_i * grid_x.shape[1] + pixel_j
        np.maximum.at(dsm.ravel(), flat_indices, z_valid)

        dsm[dsm == -np.inf] = 0.0

        logger.info(f"DSM created ({len(x):,} points)")
        return dsm

    def create_chm(self, dsm: np.ndarray, dtm: np.ndarray) -> np.ndarray:
        """Create Canopy Height Model (DSM - DTM), clipped to [min_height, max_height].

        If smoothing is enabled, applies Gaussian smoothing with percentile-based
        height rescaling to compensate for zero-dilution.
        """
        chm = dsm - dtm
        chm = np.clip(chm, 0, self.max_height)
        chm[chm < self.min_height] = 0

        if self.smoothing_enabled:
            chm = self.smooth_chm(chm)

        return chm

    def smooth_chm(self, chm: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing with percentile-based height rescaling.

        Standard Gaussian smoothing treats zero (no-data) pixels as real values,
        which suppresses canopy heights. This method rescales after smoothing so
        that a target percentile of the height distribution is preserved.
        """
        sigma = self.smoothing_sigma
        smoothed = gaussian_filter(chm.astype(np.float64), sigma=sigma).astype(np.float32)

        if self.smoothing_rescale == 'none':
            logger.info(f"Applied Gaussian smoothing (sigma={sigma}, no rescale)")
            return smoothed

        nz_orig = chm > 0
        nz_smooth = smoothed > 0.1

        if not nz_orig.any() or not nz_smooth.any():
            logger.warning("No non-zero pixels for rescaling, returning unscaled smoothing")
            return smoothed

        pct = self.smoothing_rescale_percentile
        p_orig = np.percentile(chm[nz_orig], pct)
        p_smooth = np.percentile(smoothed[nz_smooth], pct)

        if p_smooth > 0:
            scale = p_orig / p_smooth
            smoothed = np.clip(smoothed * scale, 0, self.max_height)
            logger.info(f"Applied Gaussian smoothing (sigma={sigma}, "
                        f"p{pct} rescale factor={scale:.3f})")
        else:
            logger.warning("Smoothed percentile is zero, skipping rescale")

        return smoothed
