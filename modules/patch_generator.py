"""
Patch extraction from full raster layers with rotation optimization,
coverage filtering, georeferencing, and label integration.
"""

import logging
import numpy as np
from scipy.ndimage import rotate
from sklearn.decomposition import PCA
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from shapely import affinity
from shapely.geometry import box
from shapely.ops import unary_union
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('data_processor')


class PatchGenerator:

    def __init__(self, config: Dict):
        self.config = config
        self.patch_size = config['patches']['size']
        self.patch_width = self.patch_size
        self.patch_height = self.patch_size
        self.min_overlap = config['patches']['min_overlap']
        self.min_coverage = config['patches']['min_coverage']
        self.rotation_enabled = config['rotation']['enabled']

    def compute_optimal_rotation(self, raster: np.ndarray) -> float:
        """Compute rotation angle to align data's principal axis with nearest cardinal direction."""
        if not self.rotation_enabled:
            return 0.0

        data_mask = raster > 0
        y_coords, x_coords = np.where(data_mask)

        if len(x_coords) < 10:
            return 0.0

        coords = np.column_stack([x_coords, y_coords])
        pca = PCA(n_components=2)
        pca.fit(coords)

        pc1 = pca.components_[0]
        pca_angle = np.degrees(np.arctan2(pc1[1], pc1[0]))
        logger.debug(f"PCA principal axis angle: {pca_angle:.2f}")

        # Normalize to [0, 180) since 180° = 0° for alignment
        normalized = pca_angle % 180
        dist_to_horizontal = min(abs(normalized - 0), abs(normalized - 180))
        dist_to_vertical = abs(normalized - 90)

        if dist_to_horizontal < dist_to_vertical:
            target = 0
        else:
            target = 90 if normalized < 90 else -90

        # Negate because scipy.ndimage.rotate uses opposite convention
        rotation_angle = -(target - pca_angle)

        while rotation_angle > 90:
            rotation_angle -= 180
        while rotation_angle < -90:
            rotation_angle += 180

        logger.debug(f"Target alignment: {target}, rotation needed: {rotation_angle:.2f}")
        return rotation_angle

    def rotate_rasters(self, rasters: Dict[str, np.ndarray], angle: float) -> Tuple[Dict[str, np.ndarray], Tuple[int, int, int, int]]:
        """Rotate all raster layers by angle and crop to remove padding.

        Returns (rotated_rasters, crop_bounds) where crop_bounds is
        (row_min, row_max, col_min, col_max).
        """
        if angle == 0:
            shape = rasters[list(rasters.keys())[0]].shape
            return rasters, (0, shape[0], 0, shape[1])

        rotated = {}
        for key, raster in rasters.items():
            rotated[key] = rotate(raster, angle, reshape=True, order=1, cval=0.0)

        reference_raster = list(rotated.values())[0]
        crop_bounds = self._find_data_bounds(reference_raster)

        cropped = {}
        for key, raster in rotated.items():
            row_min, row_max, col_min, col_max = crop_bounds
            cropped[key] = raster[row_min:row_max, col_min:col_max]

        return cropped, crop_bounds

    def rotate_labels(self, labels_gdf: gpd.GeoDataFrame,
                     original_shape: Tuple[int, int],
                     angle: float,
                     crop_bounds: Tuple[int, int, int, int],
                     resolution: Tuple[float, float],
                     origin: Tuple[float, float]) -> gpd.GeoDataFrame:
        """Rotate label geometries around the raster center to match rotated rasters.

        Crop offset is NOT applied here - it's handled by the metadata origin
        update in process_data.py.
        """
        if angle == 0 or labels_gdf is None or len(labels_gdf) == 0:
            return labels_gdf

        # scipy.ndimage.rotate rotates around the array center
        original_height, original_width = original_shape
        center_x_world = origin[0] + (original_width / 2) * resolution[0]
        center_y_world = origin[1] - (original_height / 2) * resolution[1]

        logger.debug(f"Rotating labels by {angle:.2f} around center ({center_x_world:.1f}, {center_y_world:.1f})")

        rotated_gdf = labels_gdf.copy()
        rotated_gdf['geometry'] = labels_gdf.geometry.apply(
            lambda geom: affinity.rotate(geom, angle, origin=(center_x_world, center_y_world))
        )

        return rotated_gdf

    def _find_data_bounds(self, raster: np.ndarray) -> tuple:
        """Find bounding box of non-zero data in a raster."""
        data_mask = raster > 0
        rows_with_data = np.any(data_mask, axis=1)
        cols_with_data = np.any(data_mask, axis=0)

        row_indices = np.where(rows_with_data)[0]
        col_indices = np.where(cols_with_data)[0]

        if len(row_indices) == 0 or len(col_indices) == 0:
            return 0, raster.shape[0], 0, raster.shape[1]

        return row_indices[0], row_indices[-1] + 1, col_indices[0], col_indices[-1] + 1

    def create_support_map(self, rasters: Dict[str, np.ndarray],
                          resolution: float = 5.0) -> np.ndarray:
        """Create coarse binary support map for quick coverage estimation."""
        reference_raster = list(rasters.values())[0]

        downsample_factor = max(1, int(resolution / self.config['chm']['resolution']))
        coarse_shape = (
            reference_raster.shape[0] // downsample_factor,
            reference_raster.shape[1] // downsample_factor
        )

        support_map = np.zeros(coarse_shape, dtype=bool)

        for i in range(coarse_shape[0]):
            for j in range(coarse_shape[1]):
                i_start = i * downsample_factor
                i_end = min((i + 1) * downsample_factor, reference_raster.shape[0])
                j_start = j * downsample_factor
                j_end = min((j + 1) * downsample_factor, reference_raster.shape[1])

                block = reference_raster[i_start:i_end, j_start:j_end]
                if np.any(block != 0) and not np.all(np.isnan(block)):
                    support_map[i, j] = True

        return support_map

    def compute_optimal_overlap(self, raster_shape: Tuple[int, int],
                                support_map: np.ndarray) -> float:
        """Test overlaps from min_overlap to 50% and select the one maximizing valid patches."""
        height, width = raster_shape

        min_overlap_pct = int(self.min_overlap * 100)
        candidate_overlaps = []

        start_pct = (min_overlap_pct // 5) * 5
        if start_pct < min_overlap_pct:
            start_pct += 5

        for pct in range(start_pct, 51, 5):
            candidate_overlaps.append(pct / 100.0)

        if self.min_overlap not in candidate_overlaps:
            candidate_overlaps.insert(0, self.min_overlap)

        best_overlap = self.min_overlap
        max_patches = 0

        logger.debug(f"Computing optimal overlap for raster shape {raster_shape}")
        logger.debug(f"Testing overlaps from {self.min_overlap*100:.0f}% to 50%")

        for overlap in candidate_overlaps:
            step_y = int(self.patch_height * (1 - overlap))
            step_x = int(self.patch_width * (1 - overlap))

            patch_count = 0
            for i in range(0, height - self.patch_height + 1, step_y):
                for j in range(0, width - self.patch_width + 1, step_x):
                    if self._check_coverage(i, j, support_map):
                        patch_count += 1

            logger.debug(f"Overlap {overlap*100:.0f}%: {patch_count} valid patches")

            if patch_count > max_patches:
                max_patches = patch_count
                best_overlap = overlap

        logger.info(f"Selected optimal overlap: {best_overlap*100:.0f}% ({max_patches} patches)")
        return best_overlap

    def generate_patch_locations(self, raster_shape: Tuple[int, int],
                                 support_map: Optional[np.ndarray] = None,
                                 use_optimal_overlap: bool = True) -> List[Tuple[int, int]]:
        """Generate grid of patch locations with optional adaptive overlap."""
        height, width = raster_shape

        if use_optimal_overlap and support_map is not None:
            overlap = self.compute_optimal_overlap(raster_shape, support_map)
        else:
            overlap = self.min_overlap

        step_y = int(self.patch_height * (1 - overlap))
        step_x = int(self.patch_width * (1 - overlap))

        locations = []
        for i in range(0, height - self.patch_height + 1, step_y):
            for j in range(0, width - self.patch_width + 1, step_x):
                if support_map is not None:
                    if not self._check_coverage(i, j, support_map):
                        continue
                locations.append((i, j))

        return locations

    def _check_coverage(self, row: int, col: int, support_map: np.ndarray) -> bool:
        """Check if patch at given location meets minimum coverage threshold."""
        downsample_factor = max(1, int(5.0 / self.config['chm']['resolution']))

        row_start = max(0, row // downsample_factor)
        row_end = min(support_map.shape[0], (row + self.patch_height) // downsample_factor)
        col_start = max(0, col // downsample_factor)
        col_end = min(support_map.shape[1], (col + self.patch_width) // downsample_factor)

        if row_end <= row_start or col_end <= col_start:
            return False

        patch_support = support_map[row_start:row_end, col_start:col_end]
        return np.sum(patch_support) / patch_support.size >= self.min_coverage

    def extract_patch(self, rasters: Dict[str, np.ndarray], row: int, col: int) -> Dict[str, np.ndarray]:
        """Extract a single patch from all raster layers, padding edges if needed."""
        patches = {}
        for key, raster in rasters.items():
            patch = raster[row:row + self.patch_height, col:col + self.patch_width]
            if patch.shape != (self.patch_height, self.patch_width):
                padded = np.zeros((self.patch_height, self.patch_width), dtype=patch.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            patches[key] = patch
        return patches

    def create_geotransform(self, row: int, col: int, resolution: Tuple[float, float],
                           origin: Tuple[float, float]) -> Affine:
        """Create georeferencing Affine transform for a patch.

        Origin is (min_x, max_y) — the top-left corner in world coords.
        """
        patch_x = origin[0] + col * resolution[0]
        patch_y = origin[1] - row * resolution[1]
        return Affine.translation(patch_x, patch_y) * Affine.scale(resolution[0], -resolution[1])

    def clip_labels(self, labels_gdf: gpd.GeoDataFrame, patch_bounds: box,
                   transform: Affine) -> Optional[gpd.GeoDataFrame]:
        """Clip labels to patch bounds and transform to pixel coordinates."""
        clipped = labels_gdf[labels_gdf.intersects(patch_bounds)].copy()

        if len(clipped) == 0:
            return None

        clipped['geometry'] = clipped.geometry.apply(
            lambda geom: self._transform_geometry(geom, transform)
        )
        return clipped

    def _transform_geometry(self, geom, transform: Affine):
        """Transform geometry from world coordinates to pixel coordinates."""
        inv_transform = ~transform

        if geom.geom_type == 'Point':
            px, py = inv_transform * (geom.x, geom.y)
            return type(geom)(px, py)
        elif geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords)
            # Handle both 2D and 3D (PolygonZ) coordinates
            transformed_coords = [inv_transform * (c[0], c[1]) for c in coords]
            return type(geom)(transformed_coords)
        else:
            return geom

    def save_patch(self, patches: Dict[str, np.ndarray], rasters_base: Path,
                  filename: str, transform: Affine, crs: str,
                  labels: Optional[gpd.GeoDataFrame] = None,
                  rotation_angle: float = 0.0):
        """Save patch as single-band GeoTIFFs per layer, plus optional label shapefile."""
        patch_size_str = f"{self.patch_size}px"

        for layer_name, patch_data in patches.items():
            patch_dir = rasters_base / layer_name / patch_size_str
            patch_dir.mkdir(parents=True, exist_ok=True)

            with rasterio.open(
                patch_dir / f"{filename}.tif", 'w', driver='GTiff',
                height=self.patch_height, width=self.patch_width,
                count=1, dtype=np.float32,
                crs=crs, transform=transform,
                compress=self.config['output']['compression'],
                nodata=self.config['output']['nodata']
            ) as dst:
                dst.write(patch_data.astype(np.float32), 1)
                dst.set_band_description(1, layer_name)
                dst.update_tags(rotation_angle=rotation_angle)

        if labels is not None and self.config['output']['save_annotations']:
            label_dir = rasters_base / 'labels' / patch_size_str
            label_dir.mkdir(parents=True, exist_ok=True)
            # Labels are in pixel coordinates after clip_labels but retain
            # the original CRS from the input shapefile. If the input had no
            # CRS (no .prj file), set it from the raster to avoid warnings.
            if labels.crs is None:
                labels = labels.set_crs(crs)
            labels.to_file(label_dir / f"{filename}.shp")

    def generate_patches(self, rasters: Dict[str, np.ndarray], metadata: Dict,
                        rasters_base: Path, site_name: str, plot_name: str,
                        labels_gdf: Optional[gpd.GeoDataFrame] = None,
                        rotation_angle: float = 0.0,
                        aoi_gdf: Optional[gpd.GeoDataFrame] = None,
                        resolution_override: Optional[tuple] = None) -> List[str]:
        """Generate all patches from rasters and save as single-band GeoTIFFs per layer.

        Args:
            aoi_gdf: Optional AOI GeoDataFrame. If provided, only patches inside
                     the AOI get labels; all patches are saved.
            resolution_override: Optional (res_x, res_y) tuple to override config
                     resolution (used by --from-rasters when raster resolution
                     differs from config).

        Returns list of dicts with 'filename', 'bounds', 'in_aoi', 'has_labels'.
        """
        support_map = self.create_support_map(rasters)
        raster_shape = list(rasters.values())[0].shape
        locations = self.generate_patch_locations(raster_shape, support_map)

        # Pre-compute AOI union geometry for fast containment checks
        aoi_geom = None
        if aoi_gdf is not None:
            aoi_geom = unary_union(aoi_gdf.geometry)
            logger.info(f"AOI filter active: {aoi_geom.geom_type} with area {aoi_geom.area:.1f} m²")

        logger.info(f"Testing {len(locations)} candidate patch locations")

        if resolution_override is not None:
            resolution = resolution_override
        else:
            res = self.config['chm']['resolution']
            resolution = (res, res)
        origin = (metadata['bounds']['min_x'], metadata['bounds']['max_y'])
        crs = metadata.get('crs', f"EPSG:{self.config['crs']['fallback_epsg']}")

        patch_names = []
        skipped_aoi = 0

        for idx, (row, col) in enumerate(locations):
            patches = self.extract_patch(rasters, row, col)
            transform = self.create_geotransform(row, col, resolution, origin)

            patch_bounds = box(
                origin[0] + col * resolution[0],
                origin[1] - (row + self.patch_height) * resolution[1],
                origin[0] + (col + self.patch_width) * resolution[0],
                origin[1] - row * resolution[1]
            )

            # Check if patch is inside AOI (used for label clipping and split assignment)
            in_aoi = aoi_geom is None or aoi_geom.contains(patch_bounds)

            # Only clip labels for patches inside the AOI
            patch_labels = None
            if in_aoi and labels_gdf is not None:
                patch_labels = self.clip_labels(labels_gdf, patch_bounds, transform)

            if site_name == plot_name:
                filename = f"{plot_name}_{idx:04d}"
            else:
                filename = self.config['output']['naming_pattern'].format(
                    site=site_name, plot=plot_name, index=idx, x=col, y=row
                )

            # Save ALL patches (labels only for AOI patches)
            self.save_patch(patches, rasters_base, filename,
                          transform, crs, patch_labels, rotation_angle)
            patch_names.append({
                'filename': filename,
                'bounds': patch_bounds,
                'in_aoi': in_aoi,
                'has_labels': patch_labels is not None and len(patch_labels) > 0,
            })
            if not in_aoi:
                skipped_aoi += 1

            if self.config['processing']['show_progress'] and (idx + 1) % 100 == 0:
                logger.debug(f"Processed {idx + 1}/{len(locations)} patches")

        if skipped_aoi > 0:
            logger.info(f"{skipped_aoi} patches outside AOI (saved without labels)")
        logger.info(f"Saved {len(patch_names)} patches ({len(rasters)} layers each)")
        return patch_names
