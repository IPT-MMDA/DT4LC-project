#!/usr/bin/env python3
"""
UI Components for Cognitive Digital Twin Interactive UI

This module contains UI component functions for the Cognitive Digital Twin application.
"""

from pathlib import Path

import numpy as np
import rasterio
import streamlit as st

from cognitive_ui.cognitive_functions import CognitiveDigitalTwin
from cognitive_ui.config import TMP_DIR
from cognitive_ui.core.visualization import enhance_raster_for_visualization, load_raster
from cognitive_ui.utils import debug_info


def display_sidebar(twin: CognitiveDigitalTwin | None = None) -> None:
    """Display and manage the sidebar UI elements.

    Args:
        twin: Optional CognitiveDigitalTwin instance. If None, limited functionality is available.
    """
    with st.sidebar:
        # Advanced mode toggle
        if "batch_mode" not in st.session_state:
            st.session_state.batch_mode = False

        st.session_state.batch_mode = st.checkbox(
            "🔧 Advanced Mode: Batch changes",
            value=st.session_state.batch_mode,
            help="Enable to make multiple changes before applying them",
        )

        # Dataset selection
        with st.expander("🗃️ Dataset", expanded=True):
            # Store the previous data source to detect changes
            previous_data_source = st.session_state.data_source if "data_source" in st.session_state else None

            # Initialize session state variables
            if "data_source_widget_value" not in st.session_state:
                st.session_state.data_source_widget_value = previous_data_source or "example_dataset"
            if "pending_data_source" not in st.session_state:
                st.session_state.pending_data_source = previous_data_source or "example_dataset"

            # Let user select data source
            selected_data_source = st.radio(
                "Select data source",
                ["example_dataset", "kahovka_data", "upload_data"],
                format_func=lambda x: {
                    "kahovka_data": "Kahovka Dam (2023)",
                    "example_dataset": "Sample Prithvi Imagery (2017)",
                    "upload_data": "Upload Your Own Data",
                }[x],
                index=["example_dataset", "kahovka_data", "upload_data"].index(
                    st.session_state.data_source_widget_value
                ),
                key="data_source_selector",
            )

            # Check if user made a new selection
            if selected_data_source != st.session_state.data_source_widget_value:
                st.session_state.pending_data_source = selected_data_source
                # In batch mode, update widget state immediately to show the selection
                if st.session_state.batch_mode:
                    st.session_state.data_source_widget_value = selected_data_source

            # Show status and confirmation if change is pending
            pending_data_source = st.session_state.pending_data_source
            if previous_data_source != pending_data_source:
                if st.session_state.batch_mode:
                    st.info(
                        f"📋 Pending: Switch to {['Sample Prithvi Imagery', 'Kahovka Dam', 'Upload Data'][['example_dataset', 'kahovka_data', 'upload_data'].index(pending_data_source)]}"
                    )
                else:
                    # Show immediate confirmation dialog for non-batch mode
                    st.warning("⚠️ Switching dataset will reload the twin (10-30 seconds)")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Apply", type="primary", key="confirm_data_change"):
                            st.session_state.data_source = pending_data_source
                            st.session_state.data_source_widget_value = pending_data_source
                            debug_info("Data source changed", f"From {previous_data_source} to {pending_data_source}")

                            # Handle Kahovka data source change immediately for better UX
                            if (
                                pending_data_source == "kahovka_data"
                                and "kahovka_visualization_rgb" in st.session_state
                            ):
                                st.session_state.visualization = st.session_state.kahovka_visualization_rgb
                                debug_info("Updated visualization immediately for Kahovka", "Success")

                            # Reset twin
                            if "twin" in st.session_state:
                                st.session_state.twin = None
                                st.session_state.historical_data_added = False
                                st.rerun()
                    with col2:
                        if st.button("❌ Cancel", key="cancel_data_change"):
                            st.session_state.pending_data_source = previous_data_source
                            st.session_state.data_source_widget_value = previous_data_source
                            st.rerun()

            if st.session_state.pending_data_source == "upload_data" or st.session_state.data_source == "upload_data":
                uploaded_file = st.file_uploader(
                    "📁 Upload GeoTIFF or similar raster data",
                    type=["tif", "tiff", "geotiff", "img"],
                    help="Supported formats: GeoTIFF (.tif, .tiff), IMG files",
                )

                if uploaded_file is not None:
                    # File info display
                    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                    st.info(f"📄 **{uploaded_file.name}** ({file_size_mb:.1f} MB)")

                    # Allow user to specify the date for the uploaded data
                    upload_date = st.text_input(
                        "Data Date (YYYY-MM-DD)",
                        value="2023-01-01",
                        key="uploaded_data_date",
                        help="Specify the date when this data was collected",
                    )

                    # Initialize file processing state
                    if "file_processing_state" not in st.session_state:
                        st.session_state.file_processing_state = "ready"

                    # Process file button or auto-process - use project temp directory
                    temp_path = TMP_DIR / uploaded_file.name
                    file_needs_processing = (
                        "uploaded_data_path" not in st.session_state
                        or st.session_state.uploaded_data_path != str(temp_path)
                    )

                    if file_needs_processing:
                        if st.session_state.batch_mode:
                            st.info("📋 File ready for batch processing")
                            if st.button("🔄 Process File", key="process_upload_file"):
                                st.session_state.file_processing_state = "processing"
                                st.rerun()
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🚀 Process File", type="primary", key="process_upload_immediate"):
                                    st.session_state.file_processing_state = "processing"
                                    st.rerun()
                            with col2:
                                if st.button("❌ Cancel", key="cancel_upload"):
                                    st.session_state.file_processing_state = "ready"
                                    st.rerun()

                    # File processing with progress
                    if st.session_state.file_processing_state == "processing":
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            # Step 1: Save file
                            status_text.text("💾 Saving file...")
                            progress_bar.progress(20)

                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getvalue())

                            # Step 2: Validate file
                            status_text.text("🔍 Validating raster data...")
                            progress_bar.progress(40)

                            # Quick validation using rasterio
                            with rasterio.open(temp_path) as src:
                                bands = src.count
                                height, width = src.height, src.width

                            # Step 3: Store metadata
                            status_text.text("📊 Processing metadata...")
                            progress_bar.progress(70)

                            st.session_state.uploaded_data_path = str(temp_path)
                            st.session_state.upload_metadata = {
                                "filename": uploaded_file.name,
                                "size_mb": file_size_mb,
                                "bands": bands,
                                "dimensions": f"{width}x{height}",
                                "date": upload_date,
                            }

                            # Step 4: Complete
                            status_text.text("✅ File processed successfully!")
                            progress_bar.progress(100)

                            st.session_state.file_processing_state = "completed"

                            # Auto-apply if not in batch mode
                            if not st.session_state.batch_mode:
                                st.session_state.previous_upload_path = str(temp_path)
                                if "twin" in st.session_state:
                                    st.session_state.twin = None
                                    st.session_state.historical_data_added = False
                                    st.rerun()

                            # Clear processing state after a short delay
                            import time

                            time.sleep(1)
                            status_text.text("")
                            progress_bar.empty()

                        except Exception as e:
                            status_text.text(f"❌ Error: {str(e)}")
                            st.error(f"File processing failed: {str(e)}")
                            st.session_state.file_processing_state = "error"

                    # Show file metadata if file is processed
                    if st.session_state.file_processing_state == "completed" and "upload_metadata" in st.session_state:
                        meta = st.session_state.upload_metadata
                        st.success(f"✅ **{meta['filename']}** ready ({meta['bands']} bands, {meta['dimensions']})")

                        # Allow user to specify the coordinate reference system if necessary
                        st.text_input("CRS (Optional)", placeholder="e.g., EPSG:4326", key="user_crs")

        # Visualization settings
        with st.expander("🎨 Visualization Settings", expanded=False):
            # Brightness adjustment
            brightness = st.slider(
                "Brightness",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.get("brightness_factor", 1.0),
                step=0.1,
                help="Adjust image brightness (1.0 = normal, >1.0 = brighter, <1.0 = darker)",
                key="brightness_slider",
            )

            if brightness != st.session_state.get("brightness_factor", 1.0):
                st.session_state.brightness_factor = brightness
                # Force regeneration of visualizations with new settings
                _regenerate_visualizations()
                st.rerun()

            # Contrast settings
            contrast_low = st.slider(
                "Contrast Lower Percentile",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.get("contrast_low", 2.0),
                step=0.5,
                help="Lower percentile for contrast enhancement (smaller = more contrast)",
            )

            contrast_high = st.slider(
                "Contrast Upper Percentile",
                min_value=90.0,
                max_value=100.0,
                value=st.session_state.get("contrast_high", 98.0),
                step=0.5,
                help="Upper percentile for contrast enhancement (larger = more contrast)",
            )

            # Store contrast settings in session state
            if contrast_low != st.session_state.get("contrast_low", 2.0) or contrast_high != st.session_state.get(
                "contrast_high", 98.0
            ):
                st.session_state.contrast_low = contrast_low
                st.session_state.contrast_high = contrast_high
                # Regenerate visualizations with new contrast settings
                _regenerate_visualizations()
                st.rerun()

            # Reset button
            if st.button("🔄 Reset to Defaults", key="reset_viz_settings"):
                st.session_state.brightness_factor = 1.0
                st.session_state.contrast_low = 2.0
                st.session_state.contrast_high = 98.0
                _regenerate_visualizations()
                st.rerun()

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
                    # Use project temp directory
                    debug_info("Project temp directory", str(TMP_DIR))

                    # Create a unique filename to avoid conflicts
                    historical_temp_path = TMP_DIR / f"historical_{hist_file.name}"
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

        # Apply all button for batch mode (only show if there are pending changes)
        if st.session_state.batch_mode:
            current_data_source = (
                st.session_state.data_source if "data_source" in st.session_state else "example_dataset"
            )
            pending_data_source = st.session_state.get("pending_data_source", current_data_source)

            if current_data_source != pending_data_source:
                st.markdown("---")
                st.markdown("### 🚀 Pending Changes")
                st.info(f"Dataset: {current_data_source} → {pending_data_source}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Apply All Changes", type="primary", key="apply_all_changes"):
                        # Apply data source change
                        st.session_state.data_source = pending_data_source
                        st.session_state.data_source_widget_value = pending_data_source
                        debug_info("Batch data source change", f"From {current_data_source} to {pending_data_source}")

                        # Handle visualization updates
                        if pending_data_source == "kahovka_data" and "kahovka_visualization_rgb" in st.session_state:
                            st.session_state.visualization = st.session_state.kahovka_visualization_rgb

                        # Reset twin
                        if "twin" in st.session_state:
                            st.session_state.twin = None
                            st.session_state.historical_data_added = False
                            st.success("✅ Changes applied! Reloading twin...")
                            st.rerun()

                with col2:
                    if st.button("🔄 Reset", key="reset_pending_changes"):
                        st.session_state.pending_data_source = current_data_source
                        st.session_state.data_source_widget_value = current_data_source
                        st.rerun()


def _regenerate_visualizations() -> None:
    """Regenerate all visualizations with current settings."""
    try:
        # Clear brightness applied flags so new settings can be applied
        _clear_brightness_flags()

        # Get current twin
        if "twin" not in st.session_state or st.session_state.twin is None:
            return

        twin = st.session_state.twin
        current_imagery = twin.physical_state.get("current_imagery")

        if current_imagery is not None:
            from cognitive_ui.core.visualization import enhance_raster_with_current_settings

            # Handle different data sources with appropriate band mappings
            if st.session_state.get("data_source") == "kahovka_data":
                # Kahovka data: regenerate with proper band mappings
                if current_imagery.shape[0] >= 3:
                    # RGB: Use bands 0,1,2 for Kahovka
                    new_rgb_viz = enhance_raster_with_current_settings(current_imagery, rgb_bands=(0, 1, 2))
                    st.session_state.visualization_rgb = new_rgb_viz
                    st.session_state.kahovka_visualization_rgb = new_rgb_viz

                    # False color: NIR(4), Red(2), Green(1) for Kahovka
                    if current_imagery.shape[0] >= 5:
                        false_color_bands = (min(4, current_imagery.shape[0] - 1), 2, 1)
                        false_viz = enhance_raster_with_current_settings(current_imagery, rgb_bands=false_color_bands)
                        st.session_state.visualization_false = false_viz
                        st.session_state.kahovka_visualization_false = false_viz

                        # SWIR: for Kahovka 5-band data
                        swir_bands = (min(4, current_imagery.shape[0] - 1), 3, 2)
                        swir_viz = enhance_raster_with_current_settings(current_imagery, rgb_bands=swir_bands)
                        st.session_state.visualization_swir = swir_viz
                        st.session_state.kahovka_visualization_swir = swir_viz
            else:
                # Standard data sources (Prithvi, uploaded): use standard band mappings
                # RGB visualization
                new_viz = twin.get_visualization()  # This uses enhance_raster_with_current_settings now
                if new_viz is not None:
                    st.session_state.visualization_rgb = new_viz

                # False color visualization (NIR, Red, Green)
                if current_imagery.shape[0] >= 4:
                    false_color_bands = (3, 2, 1)  # Standard NIR, Red, Green mapping
                    false_viz = enhance_raster_with_current_settings(current_imagery, rgb_bands=false_color_bands)
                    st.session_state.visualization_false = false_viz

                # SWIR visualization
                if current_imagery.shape[0] >= 5:
                    swir_bands = (4, 3, 2)  # Standard SWIR, NIR, Red mapping
                    swir_viz = enhance_raster_with_current_settings(current_imagery, rgb_bands=swir_bands)
                    st.session_state.visualization_swir = swir_viz

            # Update current visualization based on selected option
            viz_option = st.session_state.get("viz_option", "Natural Color (RGB)")
            if viz_option == "False Color (NIR-R-G)" and "visualization_false" in st.session_state:
                st.session_state.visualization = st.session_state.visualization_false
            elif viz_option == "SWIR Composite" and "visualization_swir" in st.session_state:
                st.session_state.visualization = st.session_state.visualization_swir
            else:
                st.session_state.visualization = st.session_state.visualization_rgb

            debug_info(
                "Regenerated visualizations",
                f"Active viz: {viz_option}, main viz shape: {st.session_state.visualization.shape if 'visualization' in st.session_state and st.session_state.visualization is not None else 'None'}",
            )

        # Regenerate historical visualizations if they exist
        if st.session_state.get("historical_data_added", False):
            _regenerate_historical_visualizations()

    except Exception as e:
        debug_info("Error regenerating visualizations", str(e))


def _regenerate_historical_visualizations() -> None:
    """Regenerate historical visualizations with current settings."""
    try:
        # This would need to access the historical data and regenerate
        # For now, we'll just update existing ones with brightness if they exist
        brightness_factor = st.session_state.get("brightness_factor", 1.0)

        for key in [
            "historical_visualization",
            "historical_visualization_rgb",
            "historical_visualization_false",
            "historical_visualization_swir",
        ]:
            if key in st.session_state and st.session_state[key] is not None:
                # Apply brightness adjustment to existing historical visualizations
                original = st.session_state[key]
                st.session_state[key] = np.clip(original * brightness_factor, 0, 1)

    except Exception as e:
        debug_info("Error regenerating historical visualizations", str(e))


def _clear_brightness_flags() -> None:
    """Clear brightness applied flags so visualizations can be regenerated."""
    viz_keys = [
        "visualization",
        "visualization_rgb",
        "visualization_false",
        "visualization_swir",
        "kahovka_visualization",
        "kahovka_visualization_rgb",
        "kahovka_visualization_false",
        "kahovka_visualization_swir",
        "historical_visualization",
        "historical_visualization_rgb",
        "historical_visualization_false",
        "historical_visualization_swir",
    ]

    for key in viz_keys:
        flag_key = f"{key}_brightness_applied"
        if flag_key in st.session_state:
            del st.session_state[flag_key]
