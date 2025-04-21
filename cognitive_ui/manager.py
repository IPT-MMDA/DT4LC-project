#!/usr/bin/env python3
"""
Twin management for Cognitive Digital Twin Interactive UI

This module contains functions for initializing and managing the Cognitive Digital Twin.
"""

import os
import traceback
from pathlib import Path

import numpy as np
import rasterio
import streamlit as st

from cognitive_ui.config import CACHE_DIR, DEFAULT_HISTORICAL_TIMESTAMP
from cognitive_ui.core.visualization import save_array_as_geotiff
from cognitive_ui.utils import debug_info

from .cognitive_functions import CognitiveDigitalTwin
from .core.visualization import enhance_raster_for_visualization


def initialize_twin(debug_mode: bool = False) -> CognitiveDigitalTwin | None:
    """Initialize the Cognitive Digital Twin.

    Args:
        debug_mode: If True, forces creation of a basic twin without relying on session state
    """
    print("Starting Cognitive Digital Twin initialization...")

    # When in debug mode, bypass session state and create a fresh twin
    if debug_mode:
        print("Debug mode: Creating fresh twin")
        try:
            # Create default twin and immediately ensure it has current imagery
            fresh_twin = CognitiveDigitalTwin()

            # Initialize basic visualization if not present
            vis = fresh_twin.get_visualization()
            if vis is not None and "visualization" not in st.session_state:
                st.session_state.visualization = vis
                st.session_state.visualization_rgb = vis

            return fresh_twin
        except Exception as e:
            print(f"Error creating debug twin: {e}")
            return None

    # Check if we already have a twin in session state
    if "twin" in st.session_state and st.session_state.twin is not None:
        twin_instance = st.session_state.twin
        if isinstance(twin_instance, CognitiveDigitalTwin):
            print(f"Using existing twin from session state: {type(twin_instance)}")
            return twin_instance
        else:
            print(f"Twin in session state is not a CognitiveDigitalTwin: {type(twin_instance)}")
            return None

    # Initialize default data source selection if not already set
    if "data_source" not in st.session_state:
        st.session_state.data_source = "example_dataset"  # Changed default from kahovka to example
        print(f"Setting default data source: {st.session_state.data_source}")

    try:
        # Create the twin with the selected data source
        if st.session_state.data_source == "kahovka_data":
            print("Attempting to load Kahovka data...")
            # Use Kahovka data
            kahovka_path = Path(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "resources",
                    "kahovka_data",
                    "hlsl_20230601.tif",
                )
            )

            # Debug the path
            debug_info("Kahovka path", str(kahovka_path))
            debug_info("Path exists", str(kahovka_path.exists()))

            # Try alternate paths if the file doesn't exist
            if not kahovka_path.exists():
                # Try relative path from current directory
                alt_path = Path("resources/kahovka_data/hlsl_20230601.tif")
                debug_info("Trying alternate path", str(alt_path))
                if alt_path.exists():
                    kahovka_path = alt_path
                    debug_info("Using alternate path", str(kahovka_path))
                else:
                    debug_info("Alternate path not found", str(alt_path))

                    # Try yet another path
                    alt_path2 = Path("src/interactive_ui/resources/kahovka_data/hlsl_20230601.tif")
                    debug_info("Trying second alternate path", str(alt_path2))
                    if alt_path2.exists():
                        kahovka_path = alt_path2
                        debug_info("Using second alternate path", str(kahovka_path))
                    else:
                        debug_info("Second alternate path not found", str(alt_path2))

            # Check if file exists
            if not kahovka_path.exists():
                print(f"Kahovka data file not found at {kahovka_path}. Falling back to default data.")
                print("Creating default CognitiveDigitalTwin instance...")
                st.session_state.twin = CognitiveDigitalTwin()
                st.session_state.data_source = "example_dataset"
                return st.session_state.twin

            # Custom initialization for Kahovka data to handle band differences
            print("Creating CognitiveDigitalTwin with Kahovka data...")
            st.session_state.twin = CognitiveDigitalTwin()

            try:
                # Load Kahovka data manually
                with rasterio.open(kahovka_path) as src:
                    img = src.read()

                    # Debug info
                    print(f"Kahovka image shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

                # Fix all NaN values in the original image right at the start
                img = np.nan_to_num(img, nan=0.0)
                print(f"After NaN fixing - min: {img.min()}, max: {img.max()}")

                # Create a better visualization for Kahovka data
                try:
                    # If image has 5 bands, we need to handle visualization differently
                    if img.shape[0] == 5:
                        # For visualization, use standard RGB channels (assuming bands 0,1,2 are RGB)
                        rgb_img = img[:3].copy()  # Use only first 3 bands for RGB visualization

                        # Manually normalize each band for better visualization
                        normalized_img = np.zeros_like(rgb_img, dtype=np.float32)
                        for i in range(3):
                            # Get 2nd and 98th percentile for robust normalization using nanpercentile to handle NaN values
                            p2, p98 = np.nanpercentile(rgb_img[i], (2, 98))
                            normalized_img[i] = np.clip((rgb_img[i] - p2) / (p98 - p2), 0, 1)

                        # Convert to channels-last format for matplotlib
                        rgb_display = np.transpose(normalized_img, (1, 2, 0))

                        # Store the visualization
                        st.session_state.kahovka_visualization = rgb_display
                        # Also store as visualization_rgb to match with the historical data naming convention
                        st.session_state.visualization_rgb = rgb_display
                        print(f"Kahovka visualization created with shape: {rgb_display.shape}")

                        # Create a padded version for Prithvi processing
                        padded_img = np.zeros((6, img.shape[1], img.shape[2]), dtype=img.dtype)
                        padded_img[:5, :, :] = img  # Copy original 5 bands
                        padded_img[5, :, :] = img[4, :, :]  # Duplicate the last band
                    else:
                        # For unexpected channel counts, still try to create a visualization
                        print(f"Unexpected channel count in Kahovka data: {img.shape[0]}")

                        if img.shape[0] >= 3:
                            # Just use first 3 bands for RGB
                            rgb_img = img[:3].copy()

                            # Normalize
                            normalized_img = np.zeros_like(rgb_img, dtype=np.float32)
                            for i in range(3):
                                p2, p98 = np.nanpercentile(rgb_img[i], (2, 98))
                                normalized_img[i] = np.clip((rgb_img[i] - p2) / (p98 - p2), 0, 1)

                            # Convert to channels-last
                            rgb_display = np.transpose(normalized_img, (1, 2, 0))
                            st.session_state.kahovka_visualization = rgb_display
                            st.session_state.visualization_rgb = rgb_display
                        else:
                            # Fall back to grayscale if fewer than 3 channels
                            gray_img = img[0].copy()
                            p2, p98 = np.nanpercentile(gray_img, (2, 98))
                            normalized = np.clip((gray_img - p2) / (p98 - p2), 0, 1)
                            # Create RGB by duplicating the single channel
                            rgb_display = np.stack([normalized, normalized, normalized], axis=-1)
                            st.session_state.kahovka_visualization = rgb_display
                            st.session_state.visualization_rgb = rgb_display

                        # Pad for Prithvi
                        padded_img = np.zeros((6, img.shape[1], img.shape[2]), dtype=img.dtype)
                        for i in range(img.shape[0]):
                            padded_img[i] = img[i]
                        # Fill any remaining channels with copies of the last available channel
                        for i in range(img.shape[0], 6):
                            padded_img[i] = img[min(img.shape[0] - 1, i)]

                except Exception as e:
                    print(f"Error in Kahovka visualization: {e}")
                    # Create a simple colored placeholder
                    h, w = img.shape[1], img.shape[2]
                    placeholder = np.ones((h, w, 3), dtype=np.float32)
                    placeholder[:, :, 0] = 0.2  # R
                    placeholder[:, :, 1] = 0.5  # G
                    placeholder[:, :, 2] = 0.8  # B
                    st.session_state.kahovka_visualization = placeholder
                    st.sidebar.error(f"Error creating visualization: {e}")

                    # Still create a padded version for other processing
                    # First fix NaN values in original image
                    img = np.nan_to_num(img, nan=0.0)
                    padded_img = np.zeros((6, img.shape[1], img.shape[2]), dtype=img.dtype)
                    padded_img[: min(5, img.shape[0]), :, :] = img[: min(5, img.shape[0])]  # Copy available bands
                    # Fill remaining bands with zeros

                # Set the current imagery in the twin
                st.session_state.twin.physical_state["current_imagery"] = padded_img

                # Extract environmental parameters directly since we're bypassing _load_physical_representation
                st.session_state.twin._extract_environmental_parameters(padded_img)

                # Add a custom wrapper to call _analyze_with_prithvi safely
                try:
                    st.session_state.twin._analyze_with_prithvi(padded_img)
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Prithvi model analysis limited for Kahovka data: {str(e)}")
                    # Set empty features as fallback
                    st.session_state.twin.physical_state["prithvi_features"] = np.array([])

                # Display warning about band adaptation
                st.sidebar.warning(
                    "⚠️ Kahovka data has 5 bands while Prithvi expects 6. A synthetic band has been added for compatibility."
                )

                # Try alternative band combinations for better visualization
                try:
                    # Store additional visualizations with different band combinations
                    if img.shape[0] >= 5:
                        # Store the RGB visualization (bands 0,1,2)
                        rgb_img = img[:3].copy()
                        normalized_rgb = np.zeros_like(rgb_img, dtype=np.float32)
                        for i in range(3):
                            p2, p98 = np.nanpercentile(rgb_img[i], (2, 98))
                            normalized_rgb[i] = np.clip((rgb_img[i] - p2) / (p98 - p2), 0, 1)
                        st.session_state.kahovka_visualization_rgb = np.transpose(normalized_rgb, (1, 2, 0))

                        # False color - NIR-Red-Green (4,2,1)
                        false_color_bands = [min(4, img.shape[0] - 1), 2, 1]
                        false_color_img = np.stack([img[i] for i in false_color_bands])
                        normalized_false = np.zeros_like(false_color_img, dtype=np.float32)
                        for i in range(3):
                            p2, p98 = np.nanpercentile(false_color_img[i], (2, 98))
                            normalized_false[i] = np.clip((false_color_img[i] - p2) / (p98 - p2), 0, 1)
                        st.session_state.kahovka_visualization_false = np.transpose(normalized_false, (1, 2, 0))

                        # SWIR Composite (if available) - typically bands 5,4,3 or similar
                        if img.shape[0] >= 5:  # Need at least 5 bands for SWIR
                            swir_bands = [min(4, img.shape[0] - 1), 3, 2]
                            swir_img = np.stack([img[i] for i in swir_bands])
                            normalized_swir = np.zeros_like(swir_img, dtype=np.float32)
                            for i in range(3):
                                p2, p98 = np.nanpercentile(swir_img[i], (2, 98))
                                normalized_swir[i] = np.clip((swir_img[i] - p2) / (p98 - p2), 0, 1)
                            st.session_state.kahovka_visualization_swir = np.transpose(normalized_swir, (1, 2, 0))

                        # Set the current visualization based on user selection
                        if hasattr(st.session_state, "viz_option"):
                            if st.session_state.viz_option == "Natural Color (RGB)":
                                st.session_state.kahovka_visualization = st.session_state.kahovka_visualization_rgb
                            elif st.session_state.viz_option == "False Color (NIR-R-G)":
                                st.session_state.kahovka_visualization = st.session_state.kahovka_visualization_false
                            elif st.session_state.viz_option == "SWIR Composite" and hasattr(
                                st.session_state, "kahovka_visualization_swir"
                            ):
                                st.session_state.kahovka_visualization = st.session_state.kahovka_visualization_swir
                            else:
                                # Default to RGB
                                st.session_state.kahovka_visualization = st.session_state.kahovka_visualization_rgb
                        else:
                            # Default if no selection yet
                            st.session_state.kahovka_visualization = st.session_state.kahovka_visualization_rgb

                        # Store with consistent variable names for use with both current and historical views
                        st.session_state.visualization_rgb = st.session_state.kahovka_visualization_rgb
                        st.session_state.visualization_false = st.session_state.kahovka_visualization_false
                        if hasattr(st.session_state, "kahovka_visualization_swir"):
                            st.session_state.visualization_swir = st.session_state.kahovka_visualization_swir

                        # Set the main visualization
                        st.session_state.visualization = st.session_state.kahovka_visualization
                        debug_info("Kahovka visualization set", "Success")
                except Exception as e:
                    print(f"Error creating alternative visualizations: {e}")
                    # This is non-critical, so continue without interrupting the flow
            except Exception as e:
                print(f"Error loading Kahovka data: {e}")
                # Fallback to default Prithvi data
                print("Falling back to default Prithvi data due to error")
                st.session_state.twin = CognitiveDigitalTwin()
                st.session_state.data_source = "example_dataset"
        elif st.session_state.data_source == "upload_data" and "uploaded_data_path" in st.session_state:
            # Use user uploaded data
            try:
                print(f"Loading uploaded data from {st.session_state.uploaded_data_path}")
                # Create the CognitiveDigitalTwin with the uploaded data
                st.session_state.twin = CognitiveDigitalTwin()

                # Load the data directly
                uploaded_path = st.session_state.uploaded_data_path
                with rasterio.open(uploaded_path) as src:
                    img = src.read()
                    print(f"Uploaded image shape: {img.shape}")

                # Fix NaN values
                img = np.nan_to_num(img, nan=0.0)

                # Create a padded version for Prithvi (if needed)
                if img.shape[0] < 6:
                    padded_img = np.zeros((6, img.shape[1], img.shape[2]), dtype=img.dtype)
                    for i in range(min(img.shape[0], 6)):
                        padded_img[i] = img[i]
                    # Fill remaining bands with copies of the last available channel
                    for i in range(img.shape[0], 6):
                        padded_img[i] = img[min(img.shape[0] - 1, i)]
                else:
                    padded_img = img[:6]  # Use first 6 bands if more are available

                # Create visualizations for the uploaded data
                try:
                    # Try to create RGB visualization
                    if img.shape[0] >= 3:
                        # Use the first 3 bands for RGB
                        rgb_img = img[:3].copy()

                        # Normalize for better visualization
                        normalized_img = np.zeros_like(rgb_img, dtype=np.float32)
                        for i in range(3):
                            p2, p98 = np.nanpercentile(rgb_img[i], (2, 98))
                            normalized_img[i] = np.clip((rgb_img[i] - p2) / (p98 - p2), 0, 1)

                        # Convert to channels-last format for matplotlib
                        rgb_display = np.transpose(normalized_img, (1, 2, 0))
                        st.session_state.visualization = rgb_display
                        st.session_state.visualization_rgb = rgb_display
                    else:
                        # For grayscale images
                        gray_img = img[0].copy()
                        p2, p98 = np.nanpercentile(gray_img, (2, 98))
                        normalized = np.clip((gray_img - p2) / (p98 - p2), 0, 1)
                        rgb_display = np.stack([normalized, normalized, normalized], axis=-1)
                        st.session_state.visualization = rgb_display
                        st.session_state.visualization_rgb = rgb_display

                    # Try to create false color visualization (NIR, Red, Green)
                    if img.shape[0] >= 4:
                        # Use near-infrared, red, and green bands (assumed to be 3,2,1)
                        nir_band_idx = min(3, img.shape[0] - 1)  # Near-infrared (usually band 4, index 3)
                        red_band_idx = min(2, img.shape[0] - 1)  # Red (usually band 3, index 2)
                        green_band_idx = min(1, img.shape[0] - 1)  # Green (usually band 2, index 1)

                        false_color_img = np.stack([img[nir_band_idx], img[red_band_idx], img[green_band_idx]])

                        # Normalize for better visualization
                        normalized_false = np.zeros_like(false_color_img, dtype=np.float32)
                        for i in range(3):
                            p2, p98 = np.nanpercentile(false_color_img[i], (2, 98))
                            normalized_false[i] = np.clip((false_color_img[i] - p2) / (p98 - p2), 0, 1)

                        st.session_state.visualization_false = np.transpose(normalized_false, (1, 2, 0))

                    # Try to create SWIR visualization if enough bands are available
                    if img.shape[0] >= 5:
                        # Use SWIR, NIR, and Red bands (assumed positions)
                        swir_bands = [min(4, img.shape[0] - 1), min(3, img.shape[0] - 1), min(2, img.shape[0] - 1)]
                        swir_img = np.stack([img[i] for i in swir_bands])

                        # Normalize for better visualization
                        normalized_swir = np.zeros_like(swir_img, dtype=np.float32)
                        for i in range(3):
                            p2, p98 = np.nanpercentile(swir_img[i], (2, 98))
                            normalized_swir[i] = np.clip((swir_img[i] - p2) / (p98 - p2), 0, 1)

                        st.session_state.visualization_swir = np.transpose(normalized_swir, (1, 2, 0))
                except Exception as e:
                    print(f"Error in visualization: {e}")
                    # Create a simple colored placeholder
                    h, w = img.shape[1], img.shape[2]
                    placeholder = np.ones((h, w, 3), dtype=np.float32)
                    placeholder[:, :, 0] = 0.2  # R
                    placeholder[:, :, 1] = 0.5  # G
                    placeholder[:, :, 2] = 0.8  # B
                    st.session_state.visualization_rgb = placeholder

                # Set current visualization based on selection
                if (
                    hasattr(st.session_state, "viz_option")
                    and st.session_state.viz_option == "False Color (NIR-R-G)"
                    and hasattr(st.session_state, "visualization_false")
                ):
                    st.session_state.visualization = st.session_state.visualization_false
                elif (
                    hasattr(st.session_state, "viz_option")
                    and st.session_state.viz_option == "SWIR Composite"
                    and hasattr(st.session_state, "visualization_swir")
                ):
                    st.session_state.visualization = st.session_state.visualization_swir
                else:
                    st.session_state.visualization = st.session_state.visualization_rgb

                # Set the current imagery in the twin
                st.session_state.twin.physical_state["current_imagery"] = padded_img

                # Extract environmental parameters
                try:
                    st.session_state.twin._extract_environmental_parameters(padded_img)
                    st.session_state.twin._analyze_with_prithvi(padded_img)
                except Exception as e:
                    st.warning(f"Limited analysis capabilities for uploaded data: {str(e)}")

            except Exception as e:
                st.error(f"Error processing uploaded data: {e}")
                # Fallback to default
                st.session_state.twin = CognitiveDigitalTwin()
        else:
            # Use default Prithvi data for example_dataset
            print("Initializing with default Prithvi data")
            try:
                st.session_state.twin = CognitiveDigitalTwin()
                # For debugging
                print(f"Initialized twin with default Prithvi data: {type(st.session_state.twin)}")
                # Ensure we have necessary session state variables initialized
                if "visualization" not in st.session_state:
                    # Get the default visualization from twin
                    vis = st.session_state.twin.get_visualization()
                    if vis is not None:
                        st.session_state.visualization = vis
                        st.session_state.visualization_rgb = vis
            except Exception as e:
                print(f"Error creating default twin: {e}")
                return None

        # Always initialize these to prevent errors
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
        if "historical_data_added" not in st.session_state:
            st.session_state.historical_data_added = False
        if "current_query" not in st.session_state:
            st.session_state.current_query = "What are the main environmental challenges in this area?"

        # Main layer visibility
        st.session_state.show_physical_layer = True
        st.session_state.show_cognitive_layer = True
        st.session_state.show_interface_layer = True

        # Sub-layer visibility for Cognitive Layer
        st.session_state.show_scientific_interpretation = True
        st.session_state.show_causal_reasoning = True
        st.session_state.show_intervention_suggestions = True

        # Sub-layer visibility for Interface Layer
        st.session_state.show_query_processing = True
        st.session_state.show_knowledge_synthesis = True
        st.session_state.show_uncertainty_communication = True

        # Store generated content (initialize to None)
        st.session_state.interpretation = None
        st.session_state.causal_hypotheses = None
        st.session_state.interventions = None
        st.session_state.query_response = None
        st.session_state.synthesis_response = None
        st.session_state.uncertainty_response = None
        st.session_state.historical_visualization = None

    except Exception as e:
        print(f"Error during twin initialization: {e}")
        print(traceback.format_exc())
        return None

    if "twin" in st.session_state and st.session_state.twin is not None:
        result_twin = st.session_state.twin
        if isinstance(result_twin, CognitiveDigitalTwin):
            print(f"Returning twin of type: {type(result_twin)}")
            return result_twin
        else:
            print(f"Twin in session state is not a CognitiveDigitalTwin: {type(result_twin)}")
            return None
    else:
        print("No twin in session state")
        return None


def fix_kahovka_visualization() -> None:
    """Fix the visualization for Kahovka data.
    This is called after the twin is initialized to ensure proper visualization in the UI.
    """
    if st.session_state.data_source == "kahovka_data":
        debug_info("Fixing Kahovka visualization", "Starting")

        # First check if visualization_rgb exists for Kahovka
        if hasattr(st.session_state, "kahovka_visualization_rgb"):
            debug_info("Found kahovka_visualization_rgb", "Using it")
            # Set the main visualization from kahovka_visualization_rgb
            st.session_state.visualization = st.session_state.kahovka_visualization_rgb
            st.session_state.visualization_rgb = st.session_state.kahovka_visualization_rgb

            # Copy other visualizations if they exist
            if hasattr(st.session_state, "kahovka_visualization_false"):
                st.session_state.visualization_false = st.session_state.kahovka_visualization_false
            if hasattr(st.session_state, "kahovka_visualization_swir"):
                st.session_state.visualization_swir = st.session_state.kahovka_visualization_swir

            # Apply the current visualization option if set
            if hasattr(st.session_state, "viz_option"):
                if st.session_state.viz_option == "False Color (NIR-R-G)" and hasattr(
                    st.session_state, "visualization_false"
                ):
                    st.session_state.visualization = st.session_state.visualization_false
                elif st.session_state.viz_option == "SWIR Composite" and hasattr(
                    st.session_state, "visualization_swir"
                ):
                    st.session_state.visualization = st.session_state.visualization_swir

            debug_info("Fixed Kahovka visualization", "Success")
        else:
            debug_info("No kahovka_visualization_rgb found", "Warning")


def generate_synthetic_historical_data(twin: CognitiveDigitalTwin | None = None) -> bool:
    """Generate synthetic historical data for demonstration purposes.

    Args:
        twin: The CognitiveDigitalTwin instance

    Returns:
        True if historical data was successfully generated, False otherwise
    """
    if twin is None:
        st.warning("Cannot generate historical data: Twin not initialized")
        return False

    try:
        debug_info("Generating synthetic historical data", "Triggered")
        # Get current imagery
        if "physical_state" in twin.__dict__:
            current_imagery = twin.physical_state.get("current_imagery")
            if current_imagery is not None:
                # Create a synthetic historical state
                debug_info("Creating synthetic historical data", f"Shape: {current_imagery.shape}")
                historical_imagery = current_imagery.copy()

                # Modify NIR band to simulate changes
                if historical_imagery.shape[0] >= 4:
                    historical_imagery[3] = historical_imagery[3] * 0.85

                # Use default timestamp - ensure it's a string
                timestamp_str = DEFAULT_HISTORICAL_TIMESTAMP
                debug_info("Using timestamp", timestamp_str)

                # Save to temporary file
                temp_file_path = str(CACHE_DIR / f"historical_{timestamp_str.replace('-', '')}.tif")
                save_array_as_geotiff(historical_imagery, temp_file_path)

                # Add to twin
                debug_info("Adding historical state to twin", f"Path: {temp_file_path}")
                twin.add_historical_state(imagery_path=Path(temp_file_path), timestamp=timestamp_str)

                # Create visualizations
                historical_vis = enhance_raster_for_visualization(historical_imagery)
                st.session_state.historical_visualization = historical_vis
                st.session_state.historical_visualization_rgb = historical_vis

                # Create additional visualizations if possible
                if historical_imagery.shape[0] >= 4:
                    # False color
                    false_color_bands = [3, 2, 1]  # NIR, Red, Green
                    false_color_img = np.stack([historical_imagery[i] for i in false_color_bands])
                    normalized_false = np.zeros_like(false_color_img, dtype=np.float32)
                    for i in range(3):
                        p2, p98 = np.nanpercentile(false_color_img[i], (2, 98))
                        normalized_false[i] = np.clip((false_color_img[i] - p2) / (p98 - p2), 0, 1)
                    st.session_state.historical_visualization_false = np.transpose(normalized_false, (1, 2, 0))

                # Add SWIR visualization if possible
                if historical_imagery.shape[0] >= 5:
                    # SWIR composite
                    swir_bands = [4, 3, 2]  # SWIR, NIR, Red
                    swir_img = np.stack([historical_imagery[i] for i in swir_bands])
                    normalized_swir = np.zeros_like(swir_img, dtype=np.float32)
                    for i in range(3):
                        p2, p98 = np.nanpercentile(swir_img[i], (2, 98))
                        normalized_swir[i] = np.clip((swir_img[i] - p2) / (p98 - p2), 0, 1)
                    st.session_state.historical_visualization_swir = np.transpose(normalized_swir, (1, 2, 0))

                # Ensure we have a copy in the base visualization key
                if (
                    st.session_state.historical_visualization_rgb is not None
                    and "historical_visualization" not in st.session_state
                ):
                    st.session_state.historical_visualization = st.session_state.historical_visualization_rgb.copy()

                # Debug info for visualizations
                debug_info(
                    "Historical visualization keys",
                    f"{[k for k in list(st.session_state.keys()) if isinstance(k, str) and 'historical_visualization' in k]}",
                )

                # Set flag
                st.session_state.historical_data_added = True
                return True
            else:
                st.warning("Cannot generate historical data: No current imagery found")
                return False
        else:
            st.warning("Cannot generate historical data: twin physical state not found")
            return False
    except Exception as e:
        debug_info("Error generating historical data", str(e))
        st.error(f"Error generating synthetic historical data: {str(e)}")
        return False
