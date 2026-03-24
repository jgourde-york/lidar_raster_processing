"""
AOI (Area of Interest) shapefile generation from label geometries.

Creates a polygon mask covering labeled regions so that only patches
with actual labels are included in training splits. Useful for partially
labeled datasets where some areas of a raster have no ground truth.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

logger = logging.getLogger('data_processor')


class AOIGenerator:
    """Generates AOI polygons from label shapefiles using buffered union."""

    def __init__(self, config: Dict):
        self.buffer_distance = config.get('aoi', {}).get('buffer_distance', 5.0)
        self.max_gap_area = config.get('aoi', {}).get('max_gap_area', 500.0)

    def generate(self, labels_gdf: gpd.GeoDataFrame,
                 buffer_distance: Optional[float] = None,
                 max_gap_area: Optional[float] = None,
                 crs=None) -> gpd.GeoDataFrame:
        """Generate an AOI polygon from label geometries using buffered union.

        Buffers each label, unions them, fills interior holes, then erodes
        back to produce a tight boundary around labeled regions. Small edge
        gaps (concavities) below max_gap_area are auto-filled.

        Args:
            labels_gdf: GeoDataFrame of label polygons.
            buffer_distance: Override for buffer distance (meters).
            max_gap_area: Override for max edge gap area to auto-fill (m²).
            crs: CRS to assign if labels have none (e.g. from the raster).

        Returns:
            GeoDataFrame with a single AOI polygon.
        """
        if len(labels_gdf) == 0:
            raise ValueError("No label geometries provided")

        buf = buffer_distance if buffer_distance is not None else self.buffer_distance

        aoi = self._buffered_union(labels_gdf, buf)

        # Auto-fill small edge gaps (concavities smaller than threshold)
        gap_threshold = max_gap_area if max_gap_area is not None else self.max_gap_area
        if gap_threshold > 0:
            aoi = self._fill_small_gaps(aoi, gap_threshold)

        logger.info(f"AOI generated: {aoi.area:.0f} m² (buffer={buf}m)")

        aoi_gdf = gpd.GeoDataFrame(
            {'buffer_m': [buf]},
            geometry=[aoi],
            crs=labels_gdf.crs or crs,
        )
        return aoi_gdf

    def _buffered_union(self, labels_gdf: gpd.GeoDataFrame,
                        buffer_distance: float):
        """Buffer each label, union, fill interior holes, then erode back.

        Tight fit that follows label clusters. Buffer distance controls
        how far apart labels can be before becoming separate AOI regions.
        Interior holes (treeless gaps within the labeled extent) are filled
        so the AOI doesn't pinch through narrow gaps.
        """
        buffered = labels_gdf.geometry.buffer(buffer_distance)
        merged = unary_union(buffered)

        # Fill interior holes — keeps the outer boundary but removes
        # any interior gaps (e.g. treeless areas between label clusters)
        if merged.geom_type == 'Polygon':
            merged = Polygon(merged.exterior)
        elif merged.geom_type == 'MultiPolygon':
            merged = MultiPolygon([Polygon(p.exterior) for p in merged.geoms])

        # Erode back to remove the buffer inflation while keeping merged regions
        eroded = merged.buffer(-buffer_distance * 0.5)

        # Re-buffer slightly so patch footprints at the edges are included
        result = eroded.buffer(buffer_distance * 0.5)

        return result

    def _fill_small_gaps(self, aoi, max_gap_area: float):
        """Fill small edge concavities by comparing against the convex hull.

        Gaps are the difference between the convex hull and the AOI. Any gap
        polygon smaller than max_gap_area is merged back into the AOI.
        """
        hull = aoi.convex_hull
        gaps = hull.difference(aoi)

        if gaps.is_empty:
            return aoi

        gap_list = self._extract_polygons(gaps)
        small_gaps = [g for g in gap_list if g.area <= max_gap_area]
        if small_gaps:
            filled = unary_union([aoi] + small_gaps)
            logger.debug(f"Auto-filled {len(small_gaps)} small gap(s) "
                         f"(threshold: {max_gap_area} m²)")
            return filled

        return aoi

    def get_gaps(self, aoi_gdf: gpd.GeoDataFrame) -> List[Polygon]:
        """Return edge gap polygons (convex hull minus AOI) for interactive filling."""
        aoi = aoi_gdf.geometry.iloc[0]
        hull = aoi.convex_hull
        gaps = hull.difference(aoi)

        if gaps.is_empty:
            return []

        return self._extract_polygons(gaps)

    def fill_gaps(self, aoi_gdf: gpd.GeoDataFrame,
                  gaps_to_fill: List[Polygon]) -> gpd.GeoDataFrame:
        """Merge specific gap polygons into the AOI."""
        aoi = aoi_gdf.geometry.iloc[0]
        filled = unary_union([aoi] + gaps_to_fill)

        result = aoi_gdf.copy()
        result.geometry = [filled]
        return result

    def unfill_gaps(self, aoi_gdf: gpd.GeoDataFrame,
                    gaps_to_remove: List[Polygon]) -> gpd.GeoDataFrame:
        """Remove previously filled gap polygons from the AOI."""
        aoi = aoi_gdf.geometry.iloc[0]
        removed = aoi.difference(unary_union(gaps_to_remove))

        result = aoi_gdf.copy()
        result.geometry = [removed]
        return result

    def _extract_polygons(self, geom) -> List[Polygon]:
        """Extract Polygon geometries from any geometry type."""
        if geom.geom_type == 'Polygon':
            return [geom]
        elif geom.geom_type == 'MultiPolygon':
            return list(geom.geoms)
        else:
            return [g for g in geom.geoms if g.geom_type == 'Polygon']

    def save(self, aoi_gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
        """Save AOI GeoDataFrame to shapefile."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aoi_gdf.to_file(output_path)
        logger.info(f"Saved AOI: {output_path}")
        return output_path
