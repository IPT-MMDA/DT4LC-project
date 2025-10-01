"""Visualization module for rendering maps and charts.

Provides functions to convert raw raster data into visual outputs:
- NDVI maps with colormaps
- Change detection maps
- Statistical charts
- GeoJSON for web mapping
"""

from __future__ import annotations

import base64
import io
from typing import Any

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Visualizer for geospatial data and analysis results."""

    # Color schemes
    NDVI_COLORMAP = LinearSegmentedColormap.from_list(
        "ndvi", ["#8B4513", "#F5DEB3", "#ADFF2F", "#006400"]
    )  # Brown -> Tan -> Green -> Dark Green

    CHANGE_COLORMAP = "RdBu_r"  # Red (decrease) -> White (no change) -> Blue (increase)

    def __init__(self, dpi: int = 100, figsize: tuple[int, int] = (10, 8)) -> None:
        """Initialize visualizer.

        Args:
            dpi: Resolution for rendered images
            figsize: Figure size in inches
        """
        self.dpi = dpi
        self.figsize = figsize

    def render_ndvi_map(
        self,
        ndvi_array: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Render NDVI array as a colormap PNG.

        Args:
            ndvi_array: NDVI values (-1 to 1)
            metadata: Optional metadata (bounds, crs, etc.)

        Returns:
            Dictionary with:
                - image: Base64-encoded PNG
                - colormap: Colormap name
                - statistics: Min/max/mean values
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Render NDVI
        im = ax.imshow(ndvi_array, cmap=self.NDVI_COLORMAP, vmin=-1, vmax=1, interpolation="bilinear")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label="NDVI")
        cbar.set_label("NDVI", rotation=270, labelpad=15)

        # Title
        ax.set_title("NDVI Map", fontsize=14, fontweight="bold")
        ax.axis("off")

        # Convert to base64
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        # Statistics
        stats = {
            "min": float(np.nanmin(ndvi_array)),
            "max": float(np.nanmax(ndvi_array)),
            "mean": float(np.nanmean(ndvi_array)),
            "std": float(np.nanstd(ndvi_array)),
        }

        return {
            "image": image_b64,
            "format": "png",
            "colormap": "ndvi",
            "statistics": stats,
            "metadata": metadata or {},
        }

    def render_change_map(
        self,
        change_array: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Render change detection array as a diverging colormap.

        Args:
            change_array: Change values (negative = decrease, positive = increase)
            metadata: Optional metadata

        Returns:
            Dictionary with image, colormap, statistics
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Determine symmetric range
        abs_max = np.nanmax(np.abs(change_array))
        vmin, vmax = -abs_max, abs_max

        # Render change
        im = ax.imshow(change_array, cmap=self.CHANGE_COLORMAP, vmin=vmin, vmax=vmax, interpolation="bilinear")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label="Change")
        cbar.set_label("NDVI Change", rotation=270, labelpad=15)

        # Title
        ax.set_title("NDVI Change Detection", fontsize=14, fontweight="bold")
        ax.axis("off")

        # Convert to base64
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        # Statistics
        stats = {
            "min_change": float(np.nanmin(change_array)),
            "max_change": float(np.nanmax(change_array)),
            "mean_change": float(np.nanmean(change_array)),
            "total_decrease": float(np.sum(change_array[change_array < 0])),
            "total_increase": float(np.sum(change_array[change_array > 0])),
        }

        return {
            "image": image_b64,
            "format": "png",
            "colormap": "RdBu_r",
            "statistics": stats,
            "metadata": metadata or {},
        }

    def render_statistics_chart(self, stats: dict[str, Any], chart_type: str = "bar") -> dict[str, Any]:
        """Render statistics as a chart.

        Args:
            stats: Statistics dictionary
            chart_type: Type of chart ("bar", "histogram")

        Returns:
            Dictionary with base64-encoded chart image
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if chart_type == "bar":
            # Bar chart of statistics
            keys = list(stats.keys())
            values = [stats[k] for k in keys]

            ax.bar(keys, values, color="steelblue", alpha=0.7)
            ax.set_xlabel("Statistic")
            ax.set_ylabel("Value")
            ax.set_title("Raster Statistics", fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "histogram":
            # Histogram (assumes stats contains 'values' array)
            values = stats.get("values", [])
            ax.hist(values, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Value Distribution", fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

        # Convert to base64
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "image": image_b64,
            "format": "png",
            "chart_type": chart_type,
        }

    def to_geojson(
        self,
        array: np.ndarray,
        transform: Any,
        threshold: float | None = None,
        crs: str = "EPSG:4326",
    ) -> dict[str, Any]:
        """Convert raster to GeoJSON (simplified version).

        Args:
            array: Raster array
            transform: Affine transform
            threshold: Optional threshold for filtering
            crs: Coordinate reference system

        Returns:
            GeoJSON FeatureCollection (simplified)
        """
        # For now, return a placeholder
        # Full implementation would use rasterio.features.shapes()

        features = []

        # Example: Create a bounding box feature
        height, width = array.shape
        bounds = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [0, 0],
                        [width, 0],
                        [width, height],
                        [0, height],
                        [0, 0],
                    ]
                ],
            },
            "properties": {
                "type": "bounds",
                "width": width,
                "height": height,
            },
        }
        features.append(bounds)

        return {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs}},
            "features": features,
        }


# Convenience functions
def render_ndvi(ndvi_array: np.ndarray, metadata: dict | None = None) -> dict:
    """Quick NDVI rendering."""
    viz = Visualizer()
    return viz.render_ndvi_map(ndvi_array, metadata)


def render_change(change_array: np.ndarray, metadata: dict | None = None) -> dict:
    """Quick change map rendering."""
    viz = Visualizer()
    return viz.render_change_map(change_array, metadata)


def render_chart(stats: dict, chart_type: str = "bar") -> dict:
    """Quick chart rendering."""
    viz = Visualizer()
    return viz.render_statistics_chart(stats, chart_type)
