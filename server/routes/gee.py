"""Google Earth Engine data endpoints."""

from datetime import datetime
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/gee", tags=["gee"])


@router.post("/sentinel2")  # type: ignore[misc]
async def fetch_sentinel2_data(
    bbox: list[float],
    start_date: str,
    end_date: str,
    data_type: str = Query("rgb", description="Data type: rgb, ndvi, ndwi, or ndsi"),
    cloud_cover_max: float = Query(20.0, ge=0, le=100, description="Maximum cloud cover %"),
) -> JSONResponse:
    """Fetch Sentinel-2 data from Google Earth Engine.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_type: Type of data to fetch (rgb, ndvi, ndwi, ndsi)
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        JSON with tile URL and metadata
    """
    try:
        from dta.dti.data_sources.gee_sentinel2 import fetch_sentinel2_composite, fetch_sentinel2_indices

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        # Validate date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format") from None

        # Fetch data based on type
        if data_type == "rgb":
            result = fetch_sentinel2_composite(bbox, start_date, end_date, cloud_cover_max=cloud_cover_max)
        elif data_type in ["ndvi", "ndwi", "ndsi"]:
            result = fetch_sentinel2_indices(bbox, start_date, end_date, data_type, cloud_cover_max)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid data_type: {data_type}")

        if not result.get("ok"):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching Sentinel-2 data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}") from e


@router.post("/modis")  # type: ignore[misc]
async def fetch_modis_data(
    bbox: list[float],
    start_date: str,
    end_date: str,
    data_type: str = Query("rgb", description="Data type: rgb, ndvi, ndwi"),
    cloud_cover_max: float = Query(20.0, ge=0, le=100, description="Maximum cloud cover %"),
) -> JSONResponse:
    """Fetch MODIS Terra/Aqua data from Google Earth Engine.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_type: Type of data to fetch (rgb, ndvi, ndwi)
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        JSON with tile URL and metadata
    """
    try:
        from dta.dti.data_sources.gee_modis import fetch_modis_composite, fetch_modis_indices

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        # Validate date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format") from None

        # Fetch data based on type
        if data_type == "rgb":
            result = fetch_modis_composite(bbox, start_date, end_date, None, cloud_cover_max)
        elif data_type in ["ndvi", "ndwi"]:
            result = fetch_modis_indices(bbox, start_date, end_date, data_type, cloud_cover_max)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid data_type: {data_type}")

        if not result.get("ok"):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching MODIS data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}") from e


@router.post("/landsat")  # type: ignore[misc]
async def fetch_landsat_data(
    bbox: list[float],
    start_date: str,
    end_date: str,
    data_type: str = Query("rgb", description="Data type: rgb, ndvi, ndwi, ndsi"),
    cloud_cover_max: float = Query(20.0, ge=0, le=100, description="Maximum cloud cover %"),
) -> JSONResponse:
    """Fetch Landsat 8/9 data from Google Earth Engine.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_type: Type of data to fetch (rgb, ndvi, ndwi, ndsi)
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        JSON with tile URL and metadata
    """
    try:
        from dta.dti.data_sources.gee_landsat import fetch_landsat_composite, fetch_landsat_indices

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        # Validate date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format") from None

        # Fetch data based on type
        if data_type == "rgb":
            result = fetch_landsat_composite(bbox, start_date, end_date, None, cloud_cover_max)
        elif data_type in ["ndvi", "ndwi", "ndsi"]:
            result = fetch_landsat_indices(bbox, start_date, end_date, data_type, cloud_cover_max)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid data_type: {data_type}")

        if not result.get("ok"):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching Landsat data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}") from e


@router.post("/bulk-fetch")  # type: ignore[misc]
async def bulk_fetch_datasets(request: dict[str, Any]) -> JSONResponse:
    """Bulk fetch multiple bands and indices for pre/post periods.

    Request body should contain:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        dataset_id: Dataset ID ('sentinel-2', 'modis', 'landsat-8')
        bands: List of band IDs to fetch
        indices: List of indices to fetch ('ndvi', 'ndwi', 'ndsi')
        pre_start: Pre-period start date YYYY-MM-DD
        pre_end: Pre-period end date YYYY-MM-DD
        post_start: Post-period start date YYYY-MM-DD (optional if use_now=True)
        post_end: Post-period end date YYYY-MM-DD (optional if use_now=True)
        cloud_cover_max: Maximum cloud cover percentage (0-100)
        use_now: If True, calculate post period as last 7 days

    Returns:
        JSON with list of layer metadata for bulk import
    """
    try:
        from datetime import datetime, timedelta

        from dta.dti.data_sources.gee_bulk_fetch import bulk_fetch_data

        # Extract parameters from request body
        bbox = request.get("bbox", [])
        dataset_id = request.get("dataset_id", "")
        bands = request.get("bands", [])
        indices = request.get("indices", [])
        pre_start = request.get("pre_start", "")
        pre_end = request.get("pre_end", "")
        post_start = request.get("post_start")
        post_end = request.get("post_end")
        cloud_cover_max = request.get("cloud_cover_max", 20.0)
        use_now = request.get("use_now", False)

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        # Validate dataset_id
        if dataset_id not in ["sentinel-2", "modis", "landsat-8"]:
            raise HTTPException(status_code=400, detail=f"Invalid dataset_id: {dataset_id}")

        # Validate at least one band or index selected
        if not bands and not indices:
            raise HTTPException(status_code=400, detail="Must select at least one band or index")

        # Validate date formats
        try:
            datetime.strptime(pre_start, "%Y-%m-%d")
            datetime.strptime(pre_end, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Pre-period dates must be in YYYY-MM-DD format") from None

        # Calculate post period if use_now is True
        if use_now:
            today = datetime.now().date()
            seven_days_ago = today - timedelta(days=7)
            post_start = seven_days_ago.strftime("%Y-%m-%d")
            post_end = today.strftime("%Y-%m-%d")
        else:
            if not post_start or not post_end:
                raise HTTPException(status_code=400, detail="Post-period dates required when use_now=False")
            try:
                datetime.strptime(post_start, "%Y-%m-%d")
                datetime.strptime(post_end, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Post-period dates must be in YYYY-MM-DD format") from None

        # Build period dictionaries
        pre_period = {"start": pre_start, "end": pre_end}
        post_period = {"start": post_start, "end": post_end}

        # Execute bulk fetch
        result = bulk_fetch_data(
            dataset_id=dataset_id,
            bbox=bbox,
            bands=bands,
            indices=indices,
            pre_period=pre_period,
            post_period=post_period,
            cloud_cover_max=cloud_cover_max,
        )

        if not result.get("ok"):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk fetch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk fetch data: {str(e)}") from e


@router.get("/datasets")  # type: ignore[misc]
async def list_available_datasets() -> JSONResponse:
    """List all available GEE datasets with metadata.

    Returns:
        JSON with dataset configurations
    """
    datasets = {
        "sentinel-2": {
            "id": "sentinel-2",
            "name": "Sentinel-2",
            "description": "ESA Copernicus Sentinel-2 Surface Reflectance (10m resolution)",
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "supported_indices": ["ndvi", "ndwi", "ndsi"],
            "spatial_resolution": "10m",
            "temporal_resolution": "5 days",
        },
        "modis": {
            "id": "modis",
            "name": "MODIS Terra/Aqua",
            "description": "NASA MODIS Surface Reflectance 8-Day Composite (250-500m resolution)",
            "collection": "MODIS/006/MOD09A1",
            "bands": ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04"],
            "supported_indices": ["ndvi", "ndwi"],
            "spatial_resolution": "250-500m",
            "temporal_resolution": "8 days",
        },
        "landsat-8": {
            "id": "landsat-8",
            "name": "Landsat 8/9",
            "description": "USGS Landsat 8/9 Surface Reflectance (30m resolution)",
            "collection": "LANDSAT/LC08/C02/T1_L2",
            "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            "supported_indices": ["ndvi", "ndwi", "ndsi"],
            "spatial_resolution": "30m",
            "temporal_resolution": "16 days",
        },
    }

    return JSONResponse({"ok": True, "datasets": datasets})


@router.post("/layers/persist")  # type: ignore[misc]
async def persist_layer_metadata(request: dict[str, Any]) -> JSONResponse:
    """Persist layer metadata for future export and chat context.

    Request body should contain:
        layer_id: Unique layer identifier
        layer_name: Display name of the layer
        dataset_id: Dataset ID ('sentinel-2', 'modis', 'landsat-8')
        bands: List of band IDs
        indices: List of spectral indices
        period: Period label ('pre' or 'post')
        bbox: Bounding box [minX, minY, maxX, maxY]
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        tile_url: GEE tile URL for visualization
        cloud_cover_max: Maximum cloud cover percentage

    Returns:
        JSON with success status
    """
    try:
        from datetime import datetime

        from server.layer_metadata_store import save_layer_metadata

        # Extract fields from request
        layer_id = request.get("layer_id")
        if not layer_id:
            raise HTTPException(status_code=400, detail="layer_id is required")

        metadata = {
            "layer_id": layer_id,
            "layer_name": request.get("layer_name", ""),
            "dataset_id": request.get("dataset_id", ""),
            "bands": request.get("bands", []),
            "indices": request.get("indices", []),
            "period": request.get("period", ""),
            "bbox": request.get("bbox", []),
            "start_date": request.get("start_date", ""),
            "end_date": request.get("end_date", ""),
            "tile_url": request.get("tile_url", ""),
            "cloud_cover_max": request.get("cloud_cover_max", 20.0),
            "created_at": datetime.now().isoformat(),
        }

        logger.info(f"Persisting layer metadata: {layer_id}")
        save_layer_metadata(layer_id, metadata)

        return JSONResponse({"ok": True, "layer_id": layer_id})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to persist layer metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/layers/{layer_id}/export")  # type: ignore[misc]
async def export_layer_for_analysis(
    layer_id: str,
    scale: int = Query(10, description="Resolution in meters"),
    format: str = Query("geotiff", description="Export format"),
    source: str = Query("auto", description="Data source: gee, microsoft, or auto"),
    name: str = Query(None, description="Custom export name (optional)"),
) -> JSONResponse:
    """Export a layer to GeoTIFF for AI analysis.

    This retrieves layer metadata and exports from the chosen source:
    - gee: Google Earth Engine (fast, 32MB limit)
    - microsoft: Microsoft Planetary Computer (no limits)
    - auto: Automatically choose based on size

    Args:
        layer_id: Layer ID to export
        scale: Resolution in meters (default: 10m for Sentinel-2)
        format: Export format (currently only 'geotiff' supported)
        source: Data source (gee, microsoft, auto)

    Returns:
        JSON with export status and attachment object
    """
    try:
        from dta.dti.data_sources import gee_landsat, gee_modis, gee_sentinel2
        from dta.dti.data_sources.gee_export import (
            create_attachment_from_export,
            estimate_export_size,
            export_gee_image_to_geotiff,
        )
        from server.layer_metadata_store import get_layer_metadata

        # Initialize GEE using the proper initialization function
        if not gee_sentinel2.initialize_gee():
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize Google Earth Engine. Check GEE_PROJECT_ID environment variable.",
            )

        # Retrieve layer metadata
        metadata = get_layer_metadata(layer_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Layer {layer_id} not found")

        dataset_id = metadata["dataset_id"]
        bbox = metadata["bbox"]
        bands = metadata["bands"]
        indices = metadata["indices"]
        start_date = metadata["start_date"]
        end_date = metadata["end_date"]
        cloud_cover_max = metadata.get("cloud_cover_max", 20.0)
        period = metadata.get("period", "pre")

        logger.info(f"Exporting layer {layer_id} ({dataset_id}, {bands or indices})")

        # Estimate export size
        num_bands = len(bands) if bands else len(indices)
        size_estimate = estimate_export_size(bbox, scale, num_bands)

        # Determine data source based on user choice and size
        if source == "auto":
            # Auto-select: GEE for small, Microsoft for large
            chosen_source = "gee" if size_estimate["can_use_direct_download"] else "microsoft"
            logger.info(
                f"Auto-selecting source: {chosen_source} (size estimate: {size_estimate['estimated_size_mb']:.1f} MB)"
            )
        else:
            chosen_source = source
            logger.info(f"User selected source: {chosen_source}")

        # Generate filename: name_source_bands/indices_period_resolution
        if name:
            # Use custom name if provided
            base_name = name.lower().replace(" ", "_")
        else:
            # Generate from metadata
            base_name = f"{dataset_id.replace('-', '_')}"

        # Add source
        source_suffix = "mpc" if chosen_source == "microsoft" else "gee"

        # Add bands or indices
        if indices:
            bands_suffix = "_".join(indices)
        elif bands:
            bands_suffix = "_".join(bands)
        else:
            bands_suffix = "rgb"

        # Add period if available
        period_suffix = f"_{period}" if period else ""

        # Add resolution
        resolution_suffix = f"{scale}m"

        # Combine all parts
        export_filename = f"{base_name}_{source_suffix}_{bands_suffix}{period_suffix}_{resolution_suffix}"

        logger.info(f"Export filename: {export_filename}")

        # Reconstruct GEE image based on dataset
        if indices:
            # Export spectral index
            index_type = indices[0]  # Take first index if multiple

            if dataset_id == "sentinel-2":
                result = gee_sentinel2.fetch_sentinel2_indices(
                    bbox, start_date, end_date, index_type, cloud_cover_max, return_image=True
                )
            elif dataset_id == "modis":
                result = gee_modis.fetch_modis_indices(
                    bbox, start_date, end_date, index_type, cloud_cover_max, return_image=True
                )
            elif dataset_id == "landsat-8":
                result = gee_landsat.fetch_landsat_indices(
                    bbox, start_date, end_date, index_type, cloud_cover_max, return_image=True
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset_id}")

            image = result.get("image")
            export_bands = ["index"]  # The index band is renamed to 'index' in GEE functions

        elif bands:
            # Export band composite
            if dataset_id == "sentinel-2":
                result = gee_sentinel2.fetch_sentinel2_composite(
                    bbox, start_date, end_date, bands, cloud_cover_max, return_image=True
                )
            elif dataset_id == "modis":
                result = gee_modis.fetch_modis_composite(
                    bbox, start_date, end_date, bands, cloud_cover_max, return_image=True
                )
            elif dataset_id == "landsat-8":
                result = gee_landsat.fetch_landsat_composite(
                    bbox, start_date, end_date, bands, cloud_cover_max, return_image=True
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset_id}")

            image = result.get("image")
            export_bands = bands

        else:
            raise HTTPException(status_code=400, detail="Layer has no bands or indices to export")

        # Export from chosen source
        if chosen_source == "microsoft":
            # Use Microsoft Planetary Computer
            from dta.dti.data_sources.planetary_computer import export_sentinel2_from_mpc

            logger.info("Exporting from Microsoft Planetary Computer")

            export_result = export_sentinel2_from_mpc(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                bands=bands if bands else ["B04", "B03", "B02"],  # Default RGB
                scale=scale,
                cloud_cover_max=cloud_cover_max,
                filename=export_filename,
            )

        else:
            # Use Google Earth Engine
            if not image:
                raise HTTPException(status_code=500, detail="Failed to reconstruct GEE image")

            logger.info("Exporting from Google Earth Engine")

            # Check if region is too large for GEE direct download
            if not size_estimate["can_use_direct_download"]:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": f"Region too large for GEE ({size_estimate['estimated_size_mb']:.1f} MB). "
                        f"Please use 'Microsoft' as data source for large regions, "
                        "or reduce resolution (30m, 100m, 250m).",
                    }
                )

            # Direct download for small regions
            export_result = export_gee_image_to_geotiff(
                image=image, bbox=bbox, bands=export_bands, scale=scale, filename=export_filename
            )

        if export_result["status"] != "completed":
            raise HTTPException(status_code=500, detail=export_result.get("error", "Export failed"))

        # Create attachment object for chat
        attachment = create_attachment_from_export(
            file_path=export_result["file_path"], layer_name=metadata["layer_name"], metadata=metadata
        )

        logger.info(f"Successfully exported layer {layer_id} to {export_result['file_path']}")

        return JSONResponse(
            {
                "ok": True,
                "status": "completed",
                "attachment": attachment,
                "size_mb": export_result["size_bytes"] / (1024 * 1024),
                "source": chosen_source,  # Include source information for UI
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to export layer {layer_id}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/layers")  # type: ignore[misc]
async def list_persisted_layers() -> JSONResponse:
    """List all persisted GEE layers.

    Returns:
        JSON with list of layer metadata
    """
    try:
        from server.layer_metadata_store import list_all_layers

        layers = list_all_layers()
        return JSONResponse({"ok": True, "layers": layers, "count": len(layers)})

    except Exception as e:
        logger.error(f"Failed to list layers: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/layers/{layer_id}")  # type: ignore[misc]
async def delete_persisted_layer(layer_id: str) -> JSONResponse:
    """Delete persisted layer metadata.

    Args:
        layer_id: Layer ID to delete

    Returns:
        JSON with success status
    """
    try:
        from server.layer_metadata_store import delete_layer_metadata

        deleted = delete_layer_metadata(layer_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Layer {layer_id} not found")

        return JSONResponse({"ok": True, "layer_id": layer_id})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete layer {layer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/dates")  # type: ignore[misc]
async def get_sentinel2_dates(
    bbox: list[float] = Query(..., description="Bounding box [minX, minY, maxX, maxY]"),  # noqa: B008
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    cloud_cover_max: float = Query(20.0, ge=0, le=100, description="Maximum cloud cover %"),
) -> JSONResponse:
    """Get available Sentinel-2 acquisition dates for a bounding box.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        JSON with available dates
    """
    try:
        from dta.dti.data_sources.gee_sentinel2 import get_available_dates

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        result = get_available_dates(bbox, start_date, end_date, cloud_cover_max)

        if not result.get("ok"):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting available dates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dates: {str(e)}") from e
