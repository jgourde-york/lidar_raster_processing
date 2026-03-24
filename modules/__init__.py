"""
Data Processor Modules

Modular components for geospatial data processing:
- CHMGenerator: CHM/DSM/DTM raster generation from point clouds
- IntensityGenerator: Intensity raster generation
- DensityGenerator: Point density raster generation
- PatchGenerator: Raster to patch extraction with rotation and labels
- RasterIO: LAS/GeoTIFF loading, saving, and resampling
- PointCloudNormalizer: Height normalization of LAS/LAZ files
- SplitGenerator: Train/val/test split assignment and CSV output
- AOIGenerator: AOI shapefile generation from label geometries
- TestPlotGenerator: Interactive test plot region placement
"""

from .chm_generator import CHMGenerator
from .intensity_generator import IntensityGenerator
from .density_generator import DensityGenerator
from .patch_generator import PatchGenerator
from .raster_io import RasterIO
from .normalizer import PointCloudNormalizer
from .split_generator import SplitGenerator
from .aoi_generator import AOIGenerator
from .test_plot_generator import TestPlotGenerator

__all__ = [
    'CHMGenerator', 'IntensityGenerator', 'DensityGenerator',
    'PatchGenerator', 'RasterIO', 'PointCloudNormalizer', 'SplitGenerator',
    'AOIGenerator', 'TestPlotGenerator',
]
__version__ = '1.2.0'
