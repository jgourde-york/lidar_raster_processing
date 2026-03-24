"""
Train/val/test split assignment and splits CSV generation.
"""

import csv
import logging
from pathlib import Path
from typing import Optional, List, Dict, Callable, Tuple

import numpy as np
import geopandas as gpd
from shapely.ops import unary_union

logger = logging.getLogger('data_processor')


class SplitGenerator:
    """Assigns train/val/test splits and writes splits CSV files."""

    def __init__(self, dataset_config: Dict):
        self.dataset_config = dataset_config

    def assign_splits(self, patches: List[Dict],
                      test_regions_gdf: Optional[gpd.GeoDataFrame] = None,
                      use_three_way_ratio: bool = False) -> List[Dict]:
        """Assign train/val/test splits to patches.

        If test_regions_gdf is provided, patches whose center falls inside a
        test region are assigned to 'test'; remaining are split into train/val.

        If use_three_way_ratio is True and no test regions are provided, uses
        train/val/test ratios directly (for pipelines where patches may lack
        geographic bounds).
        """
        seed = self.dataset_config.get('seed', 42)
        rng = np.random.default_rng(seed)
        ratios = self.dataset_config['split_ratios']

        test_geom = None
        if test_regions_gdf is not None and len(test_regions_gdf) > 0:
            test_geom = unary_union(test_regions_gdf.geometry)

        if test_geom is not None:
            # Geographic test split
            train_frac = ratios['train'] / (ratios['train'] + ratios['val'])
            test_patches = []
            trainval_patches = []

            for p in patches:
                if 'bounds' in p and test_geom.contains(p['bounds'].centroid):
                    p['split'] = 'test'
                    test_patches.append(p)
                else:
                    trainval_patches.append(p)

            n = len(trainval_patches)
            indices = rng.permutation(n)
            n_train = int(n * train_frac)
            for i, idx in enumerate(indices):
                trainval_patches[idx]['split'] = 'train' if i < n_train else 'val'

            return test_patches + trainval_patches

        if use_three_way_ratio:
            # Standard ratio-based split (train/val/test from config)
            train_ratio = ratios['train']
            val_ratio = ratios['val']

            def assign_by_ratio(group):
                n = len(group)
                indices = rng.permutation(n)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                for i, idx in enumerate(indices):
                    if i < n_train:
                        group[idx]['split'] = 'train'
                    elif i < n_train + n_val:
                        group[idx]['split'] = 'val'
                    else:
                        group[idx]['split'] = 'test'

            labeled = [p for p in patches if p.get('has_labels')]
            unlabeled = [p for p in patches if not p.get('has_labels')]
            if labeled:
                assign_by_ratio(labeled)
            if unlabeled:
                assign_by_ratio(unlabeled)
            return labeled + unlabeled

        # No test regions, two-way split (train/val only)
        train_frac = ratios['train'] / (ratios['train'] + ratios['val'])
        n = len(patches)
        indices = rng.permutation(n)
        n_train = int(n * train_frac)
        for i, idx in enumerate(indices):
            patches[idx]['split'] = 'train' if i < n_train else 'val'
        return patches

    def write_splits_csv(self, patches: List[Dict], csv_path: Path,
                         get_paths_fn: Callable) -> None:
        """Write splits CSV from assigned patches.

        Args:
            patches: List of dicts with 'split' already assigned.
            csv_path: Output CSV path.
            get_paths_fn: Callable(patch_dict) -> (patch_file_str, label_file_str, site, plot)
        """
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['patch_file', 'label_file', 'split', 'site', 'plot'])
            writer.writeheader()
            for p in patches:
                patch_file, label_file, site, plot = get_paths_fn(p)
                writer.writerow({
                    'patch_file': patch_file,
                    'label_file': label_file,
                    'split': p['split'],
                    'site': site,
                    'plot': plot,
                })

        n_train = sum(1 for p in patches if p['split'] == 'train')
        n_val = sum(1 for p in patches if p['split'] == 'val')
        n_test = sum(1 for p in patches if p['split'] == 'test')
        logger.info(f"Generated splits CSV: {csv_path}")
        logger.info(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    def generate_las_splits(self, patch_info: List[Dict], rasters_dir: Path,
                            resolution_str: str, patch_size_str: str,
                            test_regions_gdf: Optional[gpd.GeoDataFrame] = None):
        """Generate splits CSV for LAS-pipeline patches."""
        # Filter to AOI patches only (matches raster pipeline behavior)
        aoi_patches = [p for p in patch_info if p.get('in_aoi', True)]
        non_aoi_count = len(patch_info) - len(aoi_patches)
        if non_aoi_count > 0:
            logger.info(f"  {non_aoi_count} patches outside AOI excluded from splits")

        all_patches = self.assign_splits(
            aoi_patches, test_regions_gdf,
            use_three_way_ratio=(test_regions_gdf is None))

        csv_path = Path('data/splits') / f"split_{resolution_str}_{patch_size_str}.csv"

        def get_paths(p):
            site = p['site']
            name = p['patch_name']
            patch_file = str(rasters_dir / site / resolution_str / 'chm' / patch_size_str / f"{name}.tif")
            label_file = ''
            if p['has_labels']:
                label_file = str(rasters_dir / site / resolution_str / 'labels' / patch_size_str / f"{name}.shp")
            return patch_file, label_file, site, p['plot']

        self.write_splits_csv(all_patches, csv_path, get_paths)

    def generate_raster_splits(self, patch_infos: List[Dict], raster_path: Path,
                               rasters_dir: Path, patch_size: int,
                               test_regions_gdf: Optional[gpd.GeoDataFrame] = None,
                               site_name_override: Optional[str] = None):
        """Generate splits CSV for raster-pipeline patches.

        Auto-detects resolution from patch bounds. Excludes non-AOI patches
        from splits.
        """
        site_name = site_name_override or raster_path.parent.name

        # Derive resolution from first patch bounds
        first_info = patch_infos[0]
        b = first_info['bounds'].bounds  # (minx, miny, maxx, maxy)
        patch_width_m = b[2] - b[0]
        detected_res = round(patch_width_m / patch_size, 4)
        resolution_str = f"{detected_res}m"
        patch_size_str = f"{patch_size}px"
        rasters_base = rasters_dir / site_name / resolution_str

        # Filter to AOI patches only
        aoi_patches = [info for info in patch_infos if info.get('in_aoi', True)]
        non_aoi_count = len(patch_infos) - len(aoi_patches)
        if non_aoi_count > 0:
            logger.info(f"  {non_aoi_count} patches outside AOI excluded from splits")

        all_patches = self.assign_splits(aoi_patches, test_regions_gdf)

        csv_path = Path('data/splits') / f"split_{resolution_str}_{patch_size_str}.csv"

        def get_paths(p):
            name = p['filename']
            patch_file = str(rasters_base / 'chm' / patch_size_str / f"{name}.tif")
            label_file = ''
            if p['has_labels']:
                label_file = str(rasters_base / 'labels' / patch_size_str / f"{name}.shp")
            return patch_file, label_file, site_name, site_name

        self.write_splits_csv(all_patches, csv_path, get_paths)
