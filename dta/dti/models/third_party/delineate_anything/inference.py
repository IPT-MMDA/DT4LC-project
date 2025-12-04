"""Delineate-Anything inference wrapper.

Generates YAML configuration and runs the delineate.py script from the
downloaded model repository. Auto-detects data source (Sentinel/Planet/Maxar)
to configure appropriate filtering thresholds.

See: https://github.com/Lavreniuk/Delineate-Anything
"""

from __future__ import annotations

import base64
from datetime import datetime
import io
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import yaml

from dta.dti.utils.data_source import detect_data_source, get_filtering_thresholds

# Try to import visualization dependencies
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_BATCH_SIZE = 4  # Conservative default for GPU memory
DEFAULT_BANDS = [1, 2, 3]  # RGB ordering


def _create_field_boundaries_visualization(
    raster_path: str,
    gdf: Any,
    title: str = "Detected Field Boundaries",
) -> str | None:
    """Create visualization of detected field boundaries as filled polygons.

    Creates a segmentation-style visualization with:
    - Yellow/gold: Detected agricultural field polygons
    - Blue: Non-field areas (background)

    Args:
        raster_path: Path to input raster
        gdf: GeoDataFrame with field boundary polygons
        title: Title for the visualization

    Returns:
        Base64 encoded PNG image, or None if dependencies unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    try:
        import numpy as np
        import rasterio
        from rasterio.features import rasterize

        with rasterio.open(raster_path) as src:
            height, width = src.height, src.width
            transform = src.transform

            # Create a raster mask from the field polygons
            # 1 = field, 0 = background
            if len(gdf) > 0:
                # Ensure CRS match
                if gdf.crs != src.crs:
                    gdf = gdf.to_crs(src.crs)

                shapes = [(geom, 1) for geom in gdf.geometry]
                field_mask = rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
            else:
                field_mask = np.zeros((height, width), dtype=np.uint8)

            # Create RGB visualization
            # Yellow/gold for fields: RGB(255, 200, 50)
            # Blue for background: RGB(65, 105, 225) - royal blue
            viz = np.zeros((height, width, 3), dtype=np.uint8)

            # Background (non-field) - blue
            viz[field_mask == 0] = [65, 105, 225]

            # Fields - yellow/gold
            viz[field_mask == 1] = [255, 200, 50]

            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(viz)
            ax.set_title(f"{title} ({len(gdf)} fields)", fontsize=14, fontweight="bold")
            ax.axis("off")

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#FFC832", edgecolor="black", label=f"Fields: {len(gdf)}"),
                Patch(facecolor="#4169E1", edgecolor="black", label="Non-field"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            buf.seek(0)

            return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        logger.warning(f"Failed to create field boundary visualization: {e}")
        return None


def _validate_input(raster_path: str) -> tuple[bool, str]:
    """Validate input raster file.

    Args:
        raster_path: Path to input raster

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(raster_path)

    if not path.exists():
        return False, f"File not found: {raster_path}"

    if path.suffix.lower() not in (".tif", ".tiff", ".geotiff"):
        return False, f"Unsupported format: {path.suffix}. Expected GeoTIFF (.tif, .tiff)"

    return True, ""


def _generate_conf_yaml(
    bands: list[int],
    batch_size: int,
    minimum_area_m2: int,
    minimum_hole_area_m2: int,
    model_variant: str = "small",
) -> dict[str, Any]:
    """Generate delineation configuration.

    Args:
        bands: RGB band indices (1-based)
        batch_size: Inference batch size
        minimum_area_m2: Minimum field area threshold
        minimum_hole_area_m2: Minimum hole area threshold
        model_variant: Model size ("small" or "large")

    Returns:
        Configuration dictionary matching conf_sample.yaml structure
    """
    return {
        "model": [model_variant],  # Must be list - inference.py iterates over it
        "method": "main",
        "super_resolution": None,
        "treat_as_vrt": False,
        # execution_args set by batch mode, but include for completeness
        "execution_args": {
            "src_folder": "",
            "temp_folder": "",
            "output_path": "",
            "keep_temp": False,
            "mask_filepath": None,
        },
        "mask_info": {
            "range": 24,
            "filter_classes": [1, 10, 12, 23],
            "clip_classes": [0, 13, 14],
        },
        "background_info": {
            "background_classes_from_mask": [],
            "additional_source": None,
        },
        "data_loader": {
            "skip": False,
            "bands": bands,
            "nodata_band": None,
            "nodata_value": [0, 0, 0],
            "min": None,
            "max": None,
        },
        "execution_planner": {
            "region_width": 4096,
            "region_height": 4096,
            "pixel_offset": [-1, -1],
        },
        "postprocess_limits": {
            "num_workers": [2, 2],
            "queue_tiles_capacity": 32,
            "max_tiles_inflight": 64,
        },
        # passes is a list of pass configurations
        "passes": [
            {
                "batch_size": batch_size,
                "tile_size": None,
                "tile_step": 0.5,
                "model_args": [
                    {
                        "name": model_variant,
                        "minimal_confidence": 0.005,
                        "use_half": True,
                    }
                ],
                "delineation_config": {
                    "pixel_area_threshold": 512,
                    "remaining_area_threshold": 0.8,
                    "compose_merge_iou": 0.8,
                    "merge_iou": 0.8,
                    "merge_relative_area_threshold": 0.5,
                    "merge_asymetric_pixel_area_threshold": 32,
                    "merge_asymetric_relative_area_threshold": 0.7,
                    "merging_edge_width": 4,
                    "merge_edge_iou": 0.6,
                    "merge_edge_pixels": 192,
                },
            }
        ],
        "polygonization_args": {
            "layer_name": "fields",
            "override_if_exists": True,
        },
        "filtering_args": {
            "minimum_area_m2": minimum_area_m2,
            "minimum_part_area_m2": 0,
            "minimum_hole_area_m2": minimum_hole_area_m2,
            "minimum_background_field_area_m2": 50000,
            "minimum_background_field_hole_area_m2": 25000,
            "middleground_offset": None,
            "minimum_middleground_field_area_m2": 10000,
            "minimum_middleground_field_hole_area_m2": 5000,
        },
        "simplification_args": {
            "simplify": True,
            "epsilon_scale": 1,
            "num_workers": -1,
            "raster_resolution": [8096, 8096],
        },
    }


def _generate_batch_yaml(
    conf_path: str,
    data_root: str,
    output_root: str,
    temp_root: str,
    mask_root: str,
    include_folders: list[str],
) -> dict[str, Any]:
    """Generate batch processing configuration.

    Args:
        conf_path: Path to conf.yaml
        data_root: Root directory containing input folders
        output_root: Output directory
        temp_root: Temporary files directory
        mask_root: Mask files directory (can be empty)
        include_folders: List of folder names to process

    Returns:
        Batch configuration dictionary
    """
    return {
        "base_config": conf_path,
        "data_root": data_root,
        "output_root": output_root,
        "mask_root": mask_root,
        "temp_root": temp_root,
        "keep_temp": False,
        "include": include_folders,
        "exclude": None,
        "override": None,
    }


def delineate_fields(
    raster_path: str,
    output_path: str | None = None,
    model: str = "small",
    bands: list[int] | None = None,
    batch_size: int | None = None,
    data_source: str | None = None,
) -> dict[str, Any]:
    """Run field boundary detection on a raster image.

    Args:
        raster_path: Path to input GeoTIFF
        output_path: Path for output GeoPackage (auto-generated if None)
        model: Model variant ("small" or "large")
        bands: RGB band indices (1-based), default [1, 2, 3]
        batch_size: Inference batch size (default: 4)
        data_source: Override auto-detection ("sentinel", "planet", "maxar")

    Returns:
        Dictionary with:
            - output_path: Path to output GeoPackage
            - num_fields: Number of detected fields
            - total_area_m2: Total field area
            - crs: Coordinate reference system
            - data_source: Detected/specified data source
            - model: Model variant used

    Raises:
        FileNotFoundError: If input file not found
        ValueError: If input validation fails
        RuntimeError: If delineation fails
    """
    # Validate input
    is_valid, error = _validate_input(raster_path)
    if not is_valid:
        if "not found" in error.lower():
            raise FileNotFoundError(error)
        raise ValueError(error)

    # Get model path
    from dta.dti.models import get_model_manager

    manager = get_model_manager()
    model_path = manager.get_model_path("delineate-anything-small")

    if model_path is None or not model_path.exists():
        raise RuntimeError("Delineate-Anything model not installed. Please download it first via the Models panel.")

    # Check delineate.py exists
    delineate_script = model_path / "delineate.py"
    if not delineate_script.exists():
        raise RuntimeError(f"delineate.py not found at {model_path}")

    # Auto-detect data source if not specified
    if data_source is None:
        data_source = detect_data_source(raster_path)

    # Get filtering thresholds
    thresholds = get_filtering_thresholds(data_source)

    # Set defaults
    bands = bands or DEFAULT_BANDS
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    # Create working directory
    run_id = f"delineate_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    work_dir = Path(tempfile.gettempdir()) / "dt4lc_delineate" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Setup paths
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    temp_dir = work_dir / "temp"
    mask_dir = work_dir / "masks"  # Empty masks dir (required by delineate.py)
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    # Create subfolder structure expected by delineate.py
    # delineate.py expects: data_root/folder_name/image.tif
    input_folder = input_dir / "data"
    input_folder.mkdir(exist_ok=True)

    # Copy/link input file
    input_file = input_folder / Path(raster_path).name
    shutil.copy2(raster_path, input_file)

    # Generate configuration files
    conf = _generate_conf_yaml(
        bands=bands,
        batch_size=batch_size,
        minimum_area_m2=thresholds["minimum_area_m2"],
        minimum_hole_area_m2=thresholds["minimum_hole_area_m2"],
        model_variant=model,
    )

    conf_path = work_dir / "conf.yaml"
    with open(conf_path, "w") as f:
        yaml.dump(conf, f, default_flow_style=False)

    batch = _generate_batch_yaml(
        conf_path=str(conf_path),
        data_root=str(input_dir),
        output_root=str(output_dir),
        temp_root=str(temp_dir),
        mask_root=str(mask_dir),
        include_folders=["data"],
    )

    batch_path = work_dir / "batch.yaml"
    with open(batch_path, "w") as f:
        yaml.dump(batch, f, default_flow_style=False)

    # Run delineate.py
    logger.info(f"Running Delineate-Anything on {raster_path}")
    logger.info(f"Data source: {data_source}, thresholds: {thresholds}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(model_path) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            ["python", str(delineate_script), "-b", str(batch_path)],
            cwd=str(model_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for CPU processing
        )

        if result.returncode != 0:
            logger.error(f"Delineate failed: {result.stderr}")
            raise RuntimeError(f"Delineation failed: {result.stderr[:500]}")

    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Delineation timed out after 30 minutes") from e

    # Find output GeoPackage
    output_gpkg = output_dir / "data.gpkg"
    if not output_gpkg.exists():
        # Try alternative naming
        gpkg_files = list(output_dir.glob("*.gpkg"))
        if gpkg_files:
            output_gpkg = gpkg_files[0]
        else:
            raise RuntimeError(f"No output GeoPackage found in {output_dir}")

    # Move to final destination
    if output_path is None:
        from dta.config import TEMP_PATH

        final_dir = TEMP_PATH / "delineate_outputs"
        final_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(final_dir / f"{Path(raster_path).stem}_fields.gpkg")

    shutil.copy2(output_gpkg, output_path)

    # Read results for statistics and generate visualization
    gdf = None
    try:
        import geopandas as gpd

        gdf = gpd.read_file(output_path)
        num_fields = len(gdf)
        total_area = float(gdf.geometry.area.sum()) if "geometry" in gdf.columns else 0
        crs = str(gdf.crs) if gdf.crs else None
    except Exception as e:
        logger.warning(f"Failed to read output statistics: {e}")
        num_fields = 0
        total_area = 0
        crs = None

    # Generate visualization
    visualizations = {}
    if gdf is not None and len(gdf) > 0:
        viz = _create_field_boundaries_visualization(
            raster_path,
            gdf,
            title=f"Field Boundaries - {Path(raster_path).stem}",
        )
        if viz:
            visualizations["field_boundaries"] = viz

    # Cleanup work directory
    try:
        shutil.rmtree(work_dir)
    except Exception as e:
        logger.warning(f"Failed to cleanup work directory: {e}")

    return {
        "output_path": output_path,
        "num_fields": num_fields,
        "total_area_m2": total_area,
        "crs": crs,
        "data_source": data_source,
        "model": model,
        "thresholds": thresholds,
        "visualizations": visualizations,
    }


def run(RasterPath: str) -> dict[str, Any]:  # noqa: N803
    """Registry-compatible entry point for field boundary detection.

    Args:
        RasterPath: Path to input GeoTIFF (registry type name)

    Returns:
        Detection results dictionary
    """
    # Get model variant from environment if set
    model = os.environ.get("DELINEATE_MODEL", "small")

    return delineate_fields(
        raster_path=RasterPath,
        model=model,
    )
