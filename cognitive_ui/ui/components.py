#!/usr/bin/env python3
"""
UI Components for Cognitive Digital Twin Interactive UI

This module contains UI component functions for the Cognitive Digital Twin application.
"""

import tempfile
from pathlib import Path
import streamlit as st
import rasterio
import numpy as np

from cognitive_ui.cognitive_functions import CognitiveDigitalTwin
from cognitive_ui.core.visualization import enhance_raster_for_visualization, load_raster
from cognitive_ui.utils import debug_info


def display_sidebar(twin: CognitiveDigitalTwin | None = None) -> None:
    """Display and manage the sidebar UI elements.

    Args:
        twin: Optional CognitiveDigitalTwin instance. If None, limited functionality is available.
    """
    with st.sidebar:
        # Dataset selection
        with st.expander("🗃️ Dataset", expanded=True):
            # Store the previous data source to detect changes
            previous_data_source = st.session_state.data_source if "data_source" in st.session_state else None

            # Let user select data source
            current_data_source = st.radio(
                "Select data source",
                ["example_dataset", "kahovka_data", "upload_data"],
                format_func=lambda x: {
                    "kahovka_data": "Kahovka Dam (2023)",
                    "example_dataset": "Sample Prithvi Imagery (2017)",
                    "upload_data": "Upload Your Own Data",
                }[x],
                index=0,
            )

            # If data source has changed, reset twin and update session state
            if previous_data_source != current_data_source:
                debug_info("Data source changed", f"From {previous_data_source} to {current_data_source}")
                st.session_state.data_source = current_data_source

                # Handle Kahovka data source change immediately for better UX
                if current_data_source == "kahovka_data" and "kahovka_visualization_rgb" in st.session_state:
                    # Update visualization without full twin reset for faster UI update
                    st.session_state.visualization = st.session_state.kahovka_visualization_rgb
                    debug_info("Updated visualization immediately for Kahovka", "Success")

                # Only reset if we're not in emergency mode
                if not st.session_state.get("in_emergency_mode", False):
                    if "twin" in st.session_state:
                        st.session_state.twin = None
                        st.session_state.historical_data_added = False
                        st.rerun()
                else:
                    st.warning("Currently in emergency mode. Please click Reset Twin to change data source.")

            if st.session_state.data_source == "upload_data":
                uploaded_file = st.file_uploader(
                    "Upload GeoTIFF or similar raster data", type=["tif", "tiff", "geotiff", "img"]
                )

                if uploaded_file is not None:
                    # Save the uploaded file to a temporary location
                    temp_path = Path(tempfile.gettempdir()) / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Store the path in session state
                    st.session_state.uploaded_data_path = str(temp_path)

                    # Allow user to specify the date for the uploaded data
                    st.text_input(
                        "Data Date (YYYY-MM-DD)",
                        value="2023-01-01",
                        key="uploaded_data_date",
                        help="Specify the date when this data was collected",
                    )

                    # Check if this is a new upload (different from what we have)
                    if "previous_upload_path" not in st.session_state or st.session_state.previous_upload_path != str(
                        temp_path
                    ):
                        st.session_state.previous_upload_path = str(temp_path)
                        # Only reset if we're not in emergency mode
                        if not st.session_state.get("in_emergency_mode", False):
                            if "twin" in st.session_state:
                                st.session_state.twin = None
                                st.session_state.historical_data_added = False
                                st.rerun()
                        else:
                            st.warning(
                                "Currently in emergency mode. Please click Reset Twin to use new uploaded data."
                            )

                    # Allow user to specify the coordinate reference system if necessary
                    st.text_input("CRS (Optional)", placeholder="e.g., EPSG:4326", key="user_crs")

                    # Get metadata about the uploaded file
                    try:
                        with rasterio.open(temp_path) as src:
                            st.write(f"Loaded data with {src.count} bands.")
                            st.write(f"CRS: {src.crs}")
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")

                    # Add a button to process and use the uploaded data
                    use_uploaded_button = st.button(
                        "📥 Use Uploaded Data", help="Process and use the data file you uploaded"
                    )
                    if use_uploaded_button:
                        with st.spinner("Processing uploaded data..."):
                            try:
                                # Get path from session state
                                data_path = Path(st.session_state.uploaded_data_path)
                                timestamp = st.session_state.uploaded_data_date

                                # Reset twin if not in emergency mode
                                if not st.session_state.get("in_emergency_mode", False):
                                    st.session_state.twin = None
                                    # Initialize a new twin with the uploaded data
                                    twin = CognitiveDigitalTwin(study_area_path=data_path)
                                    st.session_state.twin = twin
                                else:
                                    # In emergency mode, just update the existing twin
                                    if twin is not None:
                                        # Reload the physical representation with new data
                                        twin.study_area_path = data_path
                                        twin._load_physical_representation()

                                # Load and visualize the uploaded data
                                uploaded_raster = load_raster(data_path)
                                visualization = enhance_raster_for_visualization(uploaded_raster)
                                st.session_state.visualization = visualization

                                # Create RGB visualization if enough bands
                                if uploaded_raster.shape[0] >= 3:
                                    # Create RGB visualization
                                    rgb_img = uploaded_raster[:3].copy()
                                    normalized_rgb = np.zeros_like(rgb_img, dtype=np.float32)
                                    for i in range(3):
                                        p2, p98 = np.nanpercentile(rgb_img[i], (2, 98))
                                        normalized_rgb[i] = np.clip((rgb_img[i] - p2) / (p98 - p2), 0, 1)
                                    st.session_state.visualization_rgb = np.transpose(normalized_rgb, (1, 2, 0))

                                    # Set default visualization option to RGB
                                    st.session_state.viz_option = "Natural Color (RGB)"

                                    # Create false color visualization if enough bands
                                    if uploaded_raster.shape[0] >= 4:
                                        false_color_bands = [3, 2, 1]  # NIR, Red, Green
                                        false_color_img = np.stack([uploaded_raster[i] for i in false_color_bands])
                                        normalized_false = np.zeros_like(false_color_img, dtype=np.float32)
                                        for i in range(3):
                                            p2, p98 = np.nanpercentile(false_color_img[i], (2, 98))
                                            normalized_false[i] = np.clip((false_color_img[i] - p2) / (p98 - p2), 0, 1)
                                        st.session_state.visualization_false = np.transpose(
                                            normalized_false, (1, 2, 0)
                                        )

                                    # Create SWIR visualization if enough bands
                                    if uploaded_raster.shape[0] >= 5:
                                        swir_bands = [4, 3, 2]  # SWIR, NIR, Red
                                        swir_img = np.stack([uploaded_raster[i] for i in swir_bands])
                                        normalized_swir = np.zeros_like(swir_img, dtype=np.float32)
                                        for i in range(3):
                                            p2, p98 = np.nanpercentile(swir_img[i], (2, 98))
                                            normalized_swir[i] = np.clip((swir_img[i] - p2) / (p98 - p2), 0, 1)
                                        st.session_state.visualization_swir = np.transpose(normalized_swir, (1, 2, 0))

                                # Update the main visualization based on the selected visualization option
                                if st.session_state.viz_option == "Natural Color (RGB)":
                                    st.session_state.visualization = st.session_state.visualization_rgb
                                elif (
                                    st.session_state.viz_option == "False Color (NIR-R-G)"
                                    and "visualization_false" in st.session_state
                                ):
                                    st.session_state.visualization = st.session_state.visualization_false
                                elif (
                                    st.session_state.viz_option == "SWIR Composite"
                                    and "visualization_swir" in st.session_state
                                ):
                                    st.session_state.visualization = st.session_state.visualization_swir

                                # Display success message
                                st.success(f"Uploaded data from {timestamp} processed successfully!")
                                st.rerun()  # Refresh the UI to show the processed data
                            except Exception as e:
                                st.error(f"Error processing uploaded data: {str(e)}")

        # Add Historical Data uploader section in its own expander
        with st.expander("🕒 Historical Data", expanded=True):
            st.markdown("*Upload historical satellite imagery for change analysis:*")

            hist_file = st.file_uploader(
                "Upload historical satellite imagery",
                type=["tif", "tiff", "geotiff", "img"],
                key="hist_upload",
            )

            if hist_file is not None:
                try:
                    # Create temp directory if it doesn't exist
                    temp_dir = Path(tempfile.gettempdir())
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    debug_info("Temp directory", str(temp_dir))

                    # Create a unique filename to avoid conflicts
                    historical_temp_path = temp_dir / f"historical_{hist_file.name}"
                    debug_info("Historical temp path", str(historical_temp_path))

                    # Save the uploaded file
                    with open(historical_temp_path, "wb") as f:
                        file_content = hist_file.getvalue()
                        debug_info("File content size", f"{len(file_content)} bytes")
                        f.write(file_content)

                    # Store the path in session state
                    st.session_state.uploaded_historical_path = str(historical_temp_path)
                    debug_info("Path stored in session state", st.session_state.uploaded_historical_path)

                    # Show success message
                    st.success(f"File saved to temporary location: {historical_temp_path}")

                    # Allow user to specify the historical date
                    st.text_input("Historical Date (YYYY-MM-DD)", value="2017-01-01", key="historical_date")

                    # Get metadata about the historical file
                    try:
                        with rasterio.open(historical_temp_path) as src:
                            debug_info("Rasterio file info", f"Bands: {src.count}, Size: {src.width}x{src.height}")
                            st.write(f"Historical data loaded with {src.count} bands.")
                            st.write(f"Shape: {src.height} x {src.width}, {src.count} bands")
                            st.write(f"CRS: {src.crs}")
                    except Exception as e:
                        debug_info("Rasterio error", str(e))
                        st.error(f"Error reading historical file: {str(e)}")
                        st.error("Please make sure the file is a valid geospatial raster format.")
                except Exception as e:
                    debug_info("File upload error", str(e))
                    st.error(f"Error saving uploaded file: {str(e)}")
                    st.error("Please try again or use a different file.")

            # Add direct button to generate synthetic data in the sidebar
            if st.button("Generate Sample Historical Data", key="gen_hist_sidebar"):
                st.session_state.trigger_hist_gen = True
                st.rerun()

            # Replace the old code with a clean implementation
            # Button to use uploaded historical data if a file is uploaded
            if hist_file is not None:
                if st.button("📥 Use Uploaded Historical Data", key="use_historical_btn"):
                    try:
                        if twin is None:
                            st.error("Twin not initialized. Please select a dataset first.")
                        else:
                            # Use the path from session state
                            historical_path = Path(st.session_state.uploaded_historical_path)
                            if not historical_path.exists():
                                st.error(f"File not found: {historical_path}")
                            else:
                                # Get timestamp from input field
                                timestamp = st.session_state.historical_date
                                # Add historical state
                                twin.add_historical_state(imagery_path=historical_path, timestamp=timestamp)

                                # Verify historical state was added successfully
                                historical_added = False
                                if (
                                    hasattr(twin, "physical_state")
                                    and "historical_states" in twin.physical_state
                                    and len(twin.physical_state["historical_states"]) > 0
                                ):
                                    historical_added = True
                                    debug_info(
                                        "Historical state added to twin",
                                        f"Total states: {len(twin.physical_state['historical_states'])}",
                                    )
                                else:
                                    debug_info("Failed to add historical state to twin", "No historical states found")

                                # Add visualizations for the historical data
                                try:
                                    # Load the historical data to create visualizations
                                    debug_info("Loading historical raster", f"Path: {historical_path}")
                                    historical_raster = load_raster(historical_path)
                                    debug_info("Historical raster loaded", f"Shape: {historical_raster.shape}")

                                    # Create base visualization
                                    historical_vis = enhance_raster_for_visualization(historical_raster)
                                    st.session_state.historical_visualization = historical_vis
                                    st.session_state.historical_visualization_rgb = historical_vis
                                    debug_info("Created base historical visualizations", "RGB and base")

                                    # Create additional visualization types if data has enough bands
                                    if historical_raster.shape[0] >= 4:
                                        # False color (NIR, Red, Green)
                                        false_color_bands = [3, 2, 1]
                                        false_color_img = np.stack([historical_raster[i] for i in false_color_bands])
                                        normalized_false = np.zeros_like(false_color_img, dtype=np.float32)
                                        for i in range(3):
                                            p2, p98 = np.nanpercentile(false_color_img[i], (2, 98))
                                            normalized_false[i] = np.clip((false_color_img[i] - p2) / (p98 - p2), 0, 1)
                                        st.session_state.historical_visualization_false = np.transpose(
                                            normalized_false, (1, 2, 0)
                                        )
                                        debug_info("Created false color historical visualization", "Success")

                                    # Create SWIR visualization if enough bands
                                    if historical_raster.shape[0] >= 5:
                                        # SWIR composite (SWIR, NIR, Red)
                                        swir_bands = [4, 3, 2]
                                        swir_img = np.stack([historical_raster[i] for i in swir_bands])
                                        normalized_swir = np.zeros_like(swir_img, dtype=np.float32)
                                        for i in range(3):
                                            p2, p98 = np.nanpercentile(swir_img[i], (2, 98))
                                            normalized_swir[i] = np.clip((swir_img[i] - p2) / (p98 - p2), 0, 1)
                                        st.session_state.historical_visualization_swir = np.transpose(
                                            normalized_swir, (1, 2, 0)
                                        )
                                        debug_info("Created SWIR historical visualization", "Success")

                                    # Verify visualizations were created successfully
                                    vis_keys = [
                                        k
                                        for k in list(st.session_state.keys())
                                        if isinstance(k, str) and "historical_visualization" in k
                                    ]
                                    debug_info("Historical visualizations created", f"Keys: {vis_keys}")

                                    # Add store current visualization option
                                    if "viz_option" not in st.session_state:
                                        st.session_state.viz_option = "Natural Color (RGB)"
                                        debug_info("Set default visualization option", "Natural Color (RGB)")

                                except Exception as e:
                                    st.warning(
                                        f"Created historical state but couldn't create visualizations: {str(e)}"
                                    )
                                    debug_info("Visualization error", str(e))

                                # Mark historical data as added if either metadata or visualizations were successful
                                if historical_added or "historical_visualization" in st.session_state:
                                    st.session_state.historical_data_added = True
                                    debug_info("Historical data marked as added", "Success")
                                    # Show success message
                                    st.success(f"Historical data from {timestamp} added successfully!")
                                    # Rerun to update UI
                                    st.rerun()
                                else:
                                    debug_info(
                                        "Historical data addition incomplete",
                                        "Failed to set historical_data_added flag",
                                    )
                    except Exception as e:
                        st.error(f"Error processing historical data: {str(e)}")
                        import traceback

                        print(f"Historical data error: {traceback.format_exc()}")
