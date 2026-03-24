"""
Raster and LAS/LAZ file I/O: loading, saving, and resampling.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import laspy
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import zoom

logger = logging.getLogger('data_processor')


class RasterIO:
    """Handles reading/writing of LAS point clouds and GeoTIFF rasters."""

    def __init__(self, config: Dict):
        self.config = config

    def load_las_file(self, las_path: str) -> Tuple[np.ndarray, Dict]:
        """Load LAS/LAZ file, returning (Nx7 points array, metadata dict).

        Points columns: [x, y, z, classification, intensity, return_num, num_returns]
        """
        las_path = Path(las_path)
        if not las_path.exists():
            raise FileNotFoundError(f"LAS file not found: {las_path}")

        las = laspy.read(las_path)

        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z)

        try:
            classification = np.array(las.classification)
        except AttributeError:
            classification = np.zeros(len(x), dtype=np.uint8)

        try:
            intensity = np.array(las.intensity)
        except AttributeError:
            intensity = np.zeros(len(x), dtype=np.uint16)

        try:
            return_num = np.array(las.return_number)
            num_returns = np.array(las.number_of_returns)
        except AttributeError:
            return_num = np.ones(len(x), dtype=np.uint8)
            num_returns = np.ones(len(x), dtype=np.uint8)

        points = np.column_stack([x, y, z, classification, intensity, return_num, num_returns])

        metadata = {
            'crs': self._extract_crs(las) or f"EPSG:{self.config['crs']['fallback_epsg']}",
            'bounds': {
                'min_x': float(np.min(x)),
                'max_x': float(np.max(x)),
                'min_y': float(np.min(y)),
                'max_y': float(np.max(y)),
                'min_z': float(np.min(z)),
                'max_z': float(np.max(z))
            },
            'num_points': len(x),
            'las_version': f"{las.header.version.major}.{las.header.version.minor}",
            'point_format': las.header.point_format.id
        }

        return points, metadata

    def _extract_crs(self, las) -> Optional[str]:
        """Extract CRS from LAS header VLRs, or None if unavailable."""
        try:
            if hasattr(las, 'header') and hasattr(las.header, 'parse_crs'):
                crs = las.header.parse_crs()
                if crs is not None:
                    return str(crs)
        except Exception:
            pass
        return None

    def load_rasters_from_disk(self, raster_path: Path,
                               band: Optional[int] = None,
                               layer_name_override: Optional[str] = None,
                               ) -> Tuple[Dict[str, np.ndarray], Dict, str, str]:
        """Load pre-existing GeoTIFF raster(s) and extract metadata.

        Accepts either:
        - A single GeoTIFF file path -> loads as single layer
        - A site/resolution directory -> discovers all layers

        Returns (rasters_dict, metadata, site_name, plot_name).
        """
        raster_path = Path(raster_path)
        band_idx = band if band is not None else 1

        if raster_path.is_file() and raster_path.suffix.lower() in ('.tif', '.tiff'):
            layer_name = layer_name_override or raster_path.parent.name
            site_name = raster_path.parent.parent.parent.name
            plot_name = raster_path.stem
            with rasterio.open(raster_path) as src:
                rasters = {layer_name: src.read(band_idx).astype(np.float32)}
                metadata = {
                    'crs': str(src.crs) if src.crs else f"EPSG:{self.config['crs']['fallback_epsg']}",
                    'bounds': {
                        'min_x': src.bounds.left,
                        'max_x': src.bounds.right,
                        'min_y': src.bounds.bottom,
                        'max_y': src.bounds.top,
                    }
                }

            band_msg = f", band {band_idx}" if band_idx != 1 else ""
            logger.info(f"Loaded raster: {raster_path.name} ({layer_name}{band_msg})")
            logger.debug(f"Shape: {rasters[layer_name].shape}, "
                        f"Bounds: [{metadata['bounds']['min_x']:.2f}, {metadata['bounds']['max_x']:.2f}] x "
                        f"[{metadata['bounds']['min_y']:.2f}, {metadata['bounds']['max_y']:.2f}]")
            return rasters, metadata, site_name, plot_name

        elif raster_path.is_dir():
            # Expected structure: {raster_path}/{layer}/{plot}.tif
            layer_names = ['chm', 'dsm', 'dtm', 'intensity', 'density']
            rasters = {}
            metadata = {}
            site_name = raster_path.parent.name
            plot_name = None

            for layer in layer_names:
                layer_dir = raster_path / layer
                if not layer_dir.exists():
                    continue

                for tif_file in sorted(layer_dir.glob('*.tif')):
                    with rasterio.open(tif_file) as src:
                        rasters[layer] = src.read(1).astype(np.float32)
                        if not metadata:
                            metadata = {
                                'crs': str(src.crs) if src.crs else f"EPSG:{self.config['crs']['fallback_epsg']}",
                                'bounds': {
                                    'min_x': src.bounds.left,
                                    'max_x': src.bounds.right,
                                    'min_y': src.bounds.bottom,
                                    'max_y': src.bounds.top,
                                }
                            }
                        if plot_name is None:
                            plot_name = tif_file.stem
                    logger.info(f"Loaded {layer}: {tif_file.name}")
                    break  # Only first TIF per layer

            if not rasters:
                raise FileNotFoundError(f"No GeoTIFF rasters found in: {raster_path}")

            logger.info(f"Loaded {len(rasters)} layer(s): {', '.join(rasters.keys())}")
            return rasters, metadata, site_name or 'unknown', plot_name or 'unknown'

        else:
            raise FileNotFoundError(f"Raster path not found or unsupported: {raster_path}")

    def save_rasters(self, rasters: Dict[str, np.ndarray], metadata: Dict,
                    rasters_base: Path, plot_name: str,
                    suffix: str = "", rotation_angle: float = 0.0,
                    resolution_override: Optional[float] = None):
        """Save each raster layer as a single-band GeoTIFF.

        Output: {rasters_base}/{layer_name}/{plot_name}{suffix}.tif
        """
        resolution = resolution_override or self.config['chm']['resolution']
        origin = (metadata['bounds']['min_x'], metadata['bounds']['max_y'])
        crs = metadata.get('crs', f"EPSG:{self.config['crs']['fallback_epsg']}")
        transform = Affine.translation(origin[0], origin[1]) * Affine.scale(resolution, -resolution)

        for layer_name, raster in rasters.items():
            layer_dir = rasters_base / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)
            output_path = layer_dir / f"{plot_name}{suffix}.tif"

            with rasterio.open(
                output_path, 'w', driver='GTiff',
                height=raster.shape[0], width=raster.shape[1],
                count=1, dtype=self.config['output']['dtype'],
                crs=crs, transform=transform,
                compress=self.config['output']['compression'],
                nodata=self.config['output']['nodata']
            ) as dst:
                dst.write(raster.astype(np.float32), 1)
                dst.set_band_description(1, layer_name)
                dst.update_tags(rotation_angle=rotation_angle)

            logger.debug(f"Saved {layer_name} to {output_path}")

        logger.info(f"Saved {len(rasters)} rasters for {plot_name}{suffix}")

    def resample_raster(self, raster: np.ndarray, src_res: float, target_res: float,
                        method: str = 'bilinear') -> np.ndarray:
        """Resample a 2D raster array to a different resolution."""
        scale = src_res / target_res
        order = {'nearest': 0, 'bilinear': 1, 'cubic': 3}.get(method, 1)
        return zoom(raster, scale, order=order)
