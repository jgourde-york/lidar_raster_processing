"""
Point cloud height normalization: subtract interpolated ground elevation from Z.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import laspy

from .chm_generator import CHMGenerator

logger = logging.getLogger('data_processor')


class PointCloudNormalizer:
    """Height-normalizes LAS/LAZ point clouds using DTM interpolation."""

    def __init__(self, config: Dict):
        self.chm_generator = CHMGenerator(config)

    def normalize(self, las_path: Path, output_dir: Path) -> Path:
        """Height-normalize a point cloud by subtracting ground elevation.

        Reads the original LAS/LAZ, interpolates DTM at each point, subtracts
        ground elevation from Z, filters to canopy height range, and writes a
        new file preserving all original metadata.
        """
        logger.info(f"Normalizing: {las_path.name}")

        las = laspy.read(str(las_path))
        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z)
        num_points = len(x)
        logger.info(f"Points: {num_points:,}")
        logger.debug(f"Z range (raw): [{z.min():.2f}, {z.max():.2f}] m")

        try:
            classification = np.array(las.classification)
        except AttributeError:
            classification = np.zeros(num_points, dtype=np.uint8)

        # Pad to Nx7 — only x, y, z, classification (cols 0-3) are used by
        # ground extraction, but the interface expects 7 columns
        points_array = np.column_stack([
            x, y, z, classification,
            np.zeros(num_points),
            np.zeros(num_points),
            np.zeros(num_points)
        ])

        logger.info("Computing ground elevation at each point")
        query_xy = np.column_stack([x, y])
        ground_elev = self.chm_generator.interpolate_ground_elevation(points_array, query_xy)

        normalized_z = z - ground_elev
        logger.debug(f"Z range (normalized): [{normalized_z.min():.2f}, {normalized_z.max():.2f}] m")

        min_h = self.chm_generator.min_height
        max_h = self.chm_generator.max_height
        canopy_mask = (normalized_z >= min_h) & (normalized_z <= max_h)
        removed = num_points - np.sum(canopy_mask)
        logger.debug(f"Filtering: keeping points with height [{min_h}, {max_h}] m")
        logger.info(f"Removed {removed:,} points ({100*removed/num_points:.1f}%), {np.sum(canopy_mask):,} remaining")

        las.points = las.points[canopy_mask]
        las.z = normalized_z[canopy_mask]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{las_path.stem}_normalized{las_path.suffix}"

        logger.debug(f"Writing: {output_path}")
        las.write(str(output_path))
        logger.info("Saved normalized point cloud")

        return output_path
