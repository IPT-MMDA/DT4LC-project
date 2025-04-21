import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .core.llm import query_gemini
from .core.visualization import enhance_raster_for_visualization, load_raster
from .sample_model import PRITHVI_MODEL_PATH
from .sample_model import model as prithvi_model


class CognitiveDigitalTwin:
    """
    Cognitive Digital Twin Framework for Environmental Monitoring

    A framework that integrates a physical representation layer (Prithvi-1 satellite imagery analysis)
    with a cognitive layer (LLM-based scientific interpretation) and an interface layer
    for natural language interaction with the digital twin.
    """

    def __init__(self, study_area_path: Path | None = None):
        """
        Initialize the Cognitive Digital Twin

        Args:
            study_area_path: Path to satellite imagery of the study area
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Default study area if none provided
        if study_area_path is None:
            self.study_area_path = (
                PRITHVI_MODEL_PATH
                / "examples"
                / "HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif"
            )
        else:
            self.study_area_path = study_area_path

        # Physical layer state
        self.physical_state: dict[str, Any] = {
            "current_imagery": None,
            "historical_states": [],
            "detected_changes": [],
            "parameters": {"vegetation_indices": {}, "land_cover": {}, "soil_moisture": {}},
        }

        # Cognitive layer state
        self.cognitive_state: dict[str, Any] = {
            "interpretations": [],
            "causal_hypotheses": [],
            "scientific_context": {},
            "intervention_suggestions": [],
        }

        # Load and initialize physical representation
        self._load_physical_representation()

    # -----------------------------------------------------------------
    # Physical Layer Methods
    # -----------------------------------------------------------------

    def _load_physical_representation(self) -> None:
        """Load the physical representation from satellite imagery"""
        try:
            raster = load_raster(self.study_area_path)
            self.physical_state["current_imagery"] = raster

            # Process with Prithvi to extract features
            self._analyze_with_prithvi(raster)

            # Extract environmental parameters
            self._extract_environmental_parameters(raster)

        except Exception as e:
            print(f"Error loading physical representation: {e}")

    def _analyze_with_prithvi(self, raster: NDArray[np.float32]) -> None:
        """
        Use Prithvi-1 model to analyze satellite imagery

        Args:
            raster: The satellite imagery to analyze
        """
        try:
            # Get original dimensions
            _, h, w = raster.shape

            # Calculate padding needed to make dimensions divisible by 16
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            # Pad the raster if needed
            if pad_h > 0 or pad_w > 0:
                padded_raster = np.pad(raster, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            else:
                padded_raster = raster

            # Convert to tensor for Prithvi model
            raster_tensor = torch.from_numpy(padded_raster).float().unsqueeze(0).to(self.device)

            # Run through Prithvi model for feature extraction
            with torch.no_grad():
                model_output = prithvi_model(raster_tensor)

                # Handle case where model returns a tuple
                if isinstance(model_output, tuple):
                    # Use the first element of the tuple as features
                    features = model_output[0]
                else:
                    features = model_output

            # Store the extracted features
            self.physical_state["prithvi_features"] = features.detach().cpu().numpy()

        except Exception as e:
            print(f"Error in Prithvi processing: {e}")
            # Set empty features as fallback
            self.physical_state["prithvi_features"] = np.array([])

    def _extract_environmental_parameters(self, raster: NDArray[np.float32]) -> None:
        """
        Extract environmental parameters from satellite imagery

        Args:
            raster: The satellite imagery to analyze
        """
        # Extract and store parameters in the physical state
        if raster.shape[0] >= 4:
            self._calculate_ndvi_and_land_cover(raster, self.physical_state["parameters"])

    def _calculate_ndvi_and_land_cover(self, raster: NDArray[np.float32], target_params: dict[str, Any]) -> None:
        """
        Calculate NDVI and land cover classifications from raster data

        Args:
            raster: Satellite imagery raster data
            target_params: Dictionary to store the resulting parameters
        """
        if raster.shape[0] < 4:
            return

        # NDVI calculation (band 4 is NIR and band 3 is Red)
        nir, red = raster[3], raster[2]
        ndvi = np.zeros_like(nir)
        np.divide(nir - red, nir + red, out=ndvi, where=(nir + red) > 0)

        # Store NDVI value
        if "vegetation_indices" not in target_params:
            target_params["vegetation_indices"] = {}
        target_params["vegetation_indices"]["ndvi"] = float(ndvi.mean())

        # Land cover classification based on NDVI
        ndvi_thresholds = {
            "water": (-1, 0),
            "barren": (0, 0.2),
            "grassland": (0.2, 0.4),
            "shrubland": (0.4, 0.6),
            "forest": (0.6, 1),
        }

        # Calculate land cover percentages
        if "land_cover" not in target_params:
            target_params["land_cover"] = {}

        for land_type, (lower, upper) in ndvi_thresholds.items():
            cover_percent = float(np.sum((ndvi >= lower) & (ndvi < upper)) / ndvi.size * 100)
            target_params["land_cover"][land_type] = cover_percent

    # -----------------------------------------------------------------
    # Change Analysis Methods
    # -----------------------------------------------------------------

    def add_historical_state(self, imagery_path: Path, timestamp: str) -> None:
        """
        Add a historical state to enable change analysis

        Args:
            imagery_path: Path to historical satellite imagery
            timestamp: Timestamp for the historical imagery
        """
        try:
            historical_raster = load_raster(imagery_path)

            # Extract parameters from historical imagery
            historical_state: dict[str, Any] = {"timestamp": timestamp, "imagery": historical_raster, "parameters": {}}

            # Calculate NDVI and land cover for historical imagery
            if historical_raster.shape[0] >= 4:
                self._calculate_ndvi_and_land_cover(historical_raster, historical_state["parameters"])

                # For backward compatibility
                if (
                    "vegetation_indices" in historical_state["parameters"]
                    and "ndvi" in historical_state["parameters"]["vegetation_indices"]
                ):
                    historical_state["parameters"]["ndvi"] = historical_state["parameters"]["vegetation_indices"][
                        "ndvi"
                    ]

            # Add to historical states
            self.physical_state["historical_states"].append(historical_state)

            # Detect changes if there are multiple historical states
            if len(self.physical_state["historical_states"]) > 1:
                self._detect_changes()

        except Exception as e:
            print(f"Error adding historical state: {e}")
            import traceback

            print(traceback.format_exc())

    def _detect_changes(self) -> None:
        """
        Detect changes between current and historical states

        Compares current state with most recent historical state and
        records detected changes in the physical_state.
        """
        if not self.physical_state["historical_states"]:
            return

        # Compare current state with most recent historical state
        current_ndvi = self.physical_state["parameters"]["vegetation_indices"].get("ndvi")
        if current_ndvi is None:
            return

        historical_state = self.physical_state["historical_states"][-1]
        historical_ndvi = None

        # Try to extract historical NDVI from different possible structures
        if "parameters" in historical_state:
            params = historical_state["parameters"]
            if "vegetation_indices" in params and "ndvi" in params["vegetation_indices"]:
                historical_ndvi = params["vegetation_indices"]["ndvi"]
            elif "ndvi" in params:
                historical_ndvi = params["ndvi"]

        if historical_ndvi is None:
            return

        # Calculate change
        ndvi_change = current_ndvi - historical_ndvi

        # Record detected change
        change_record = {
            "parameter": "ndvi",
            "previous_value": historical_ndvi,
            "current_value": current_ndvi,
            "change": ndvi_change,
            "timestamp": historical_state.get("timestamp", "unknown"),
        }

        self.physical_state["detected_changes"].append(change_record)

    # -----------------------------------------------------------------
    # Cognitive Layer Methods
    # -----------------------------------------------------------------

    def _construct_prompt(self, template: str, context_data: Any) -> str:
        """
        Construct a prompt for the LLM with context data

        Args:
            template: The prompt template string
            context_data: Data to include in the prompt context

        Returns:
            Formatted prompt string
        """
        context_text = json.dumps(context_data, indent=2) if context_data else ""
        return template.format(context=context_text)

    def generate_scientific_interpretation(self) -> str:
        """
        Generate scientific interpretation of observed changes using the cognitive layer (LLM)

        Returns:
            Scientific interpretation of the observed changes
        """
        # Prepare context from physical layer
        context = {
            "land_cover": self.physical_state["parameters"]["land_cover"],
            "vegetation_indices": self.physical_state["parameters"]["vegetation_indices"],
            "detected_changes": self.physical_state["detected_changes"][-5:]
            if self.physical_state["detected_changes"]
            else [],
        }

        # Construct prompt for scientific interpretation
        prompt_template = """
        As an environmental scientist specializing in Earth observation, analyze the following 
        satellite-derived data from our environmental digital twin:
        
        {context}
        
        Please provide:
        1. A detailed scientific interpretation of the observed land cover and vegetation patterns
        2. Potential causal factors explaining the observed patterns based on ecological science
        3. Relevant environmental research context that could explain these patterns
        4. Scientific hypotheses about future changes if these trends continue
        
        Focus on scientific depth and interdisciplinary connections between ecology, hydrology, 
        climate science, and human impact. Include references to relevant research concepts.
        """

        prompt = self._construct_prompt(prompt_template, context)

        # Query LLM for interpretation
        interpretation = query_gemini(prompt, max_tokens=500)

        # Store interpretation in cognitive state
        if interpretation:
            self.cognitive_state["interpretations"].append({"timestamp": "current", "interpretation": interpretation})

        return interpretation if interpretation else "Unable to generate interpretation"

    def generate_causal_hypothesis(self) -> str:
        """
        Generate causal hypotheses for observed changes

        Returns:
            Causal hypotheses for observed changes
        """
        # Prepare context about detected changes
        context = self.physical_state["detected_changes"][-5:] if self.physical_state["detected_changes"] else []

        # Construct prompt for causal reasoning
        prompt_template = """
        As an Earth system scientist, analyze the following observed changes in our environmental digital twin:
        
        {context}
        
        Please provide:
        1. Multiple potential causal mechanisms that could explain these changes
        2. A ranking of these causal hypotheses by likelihood, based on scientific literature
        3. The key evidence we should look for to confirm each hypothesis
        4. Potential confounding factors that could be misleading our analysis
        
        Focus on scientific causal reasoning, distinguishing correlation from causation, and 
        explaining complex environmental interactions across Earth system components.
        """

        prompt = self._construct_prompt(prompt_template, context)

        # Query LLM for causal hypotheses
        causal_hypotheses = query_gemini(prompt, max_tokens=400)

        # Store in cognitive state
        if causal_hypotheses:
            self.cognitive_state["causal_hypotheses"].append({"timestamp": "current", "hypotheses": causal_hypotheses})

        return causal_hypotheses if causal_hypotheses else "Unable to generate causal hypotheses"

    def suggest_interventions(self) -> str:
        """
        Suggest targeted interventions based on scientific analysis

        Returns:
            Suggested interventions
        """
        # Combine physical and cognitive state for context
        context = {
            "physical_parameters": self.physical_state["parameters"],
            "interpretation": self.cognitive_state["interpretations"][-1]["interpretation"]
            if self.cognitive_state["interpretations"]
            else "No interpretation available",
        }

        # Construct prompt for intervention suggestions
        prompt_template = """
        As an applied environmental scientist, review the following digital twin data and interpretation:
        
        {context}
        
        Based on this information, please provide:
        1. Three evidence-based interventions that could improve environmental outcomes in this area
        2. The scientific rationale behind each intervention
        3. Expected outcomes based on peer-reviewed research
        4. Monitoring approaches to verify intervention effectiveness
        
        Focus on scientifically-grounded, practical interventions that address root causes identified in the analysis.
        """

        prompt = self._construct_prompt(prompt_template, context)

        # Query LLM for intervention suggestions
        interventions = query_gemini(prompt, max_tokens=400)

        # Store in cognitive state
        if interventions:
            self.cognitive_state["intervention_suggestions"].append(
                {"timestamp": "current", "suggestions": interventions}
            )

        return interventions if interventions else "Unable to generate intervention suggestions"

    # -----------------------------------------------------------------
    # Interface Layer Methods
    # -----------------------------------------------------------------

    def process_query(self, query: str) -> str:
        """
        Process a natural language query about the digital twin

        Args:
            query: Natural language query

        Returns:
            Response to the query
        """
        # Prepare context about the digital twin state
        context = {
            "physical_parameters": self.physical_state["parameters"],
            "detected_changes": self.physical_state["detected_changes"][-3:]
            if self.physical_state["detected_changes"]
            else [],
            "latest_interpretation": self.cognitive_state["interpretations"][-1]["interpretation"]
            if self.cognitive_state["interpretations"]
            else "",
            "latest_causal_hypothesis": self.cognitive_state["causal_hypotheses"][-1]["hypotheses"]
            if self.cognitive_state["causal_hypotheses"]
            else "",
        }

        # Construct prompt for query processing
        prompt_template = f"""
        As an environmental digital twin with scientific reasoning capabilities, answer the following query:
        
        Query: "{query}"
        
        Digital Twin State:
        {{context}}
        
        Please provide a scientifically accurate, comprehensive response, drawing on the provided context
        and your knowledge of environmental science. Include uncertainty levels where appropriate.
        """

        prompt = self._construct_prompt(prompt_template, context)

        # Query LLM for response
        response = query_gemini(prompt, max_tokens=350)

        return response if response else "Unable to process query"

    def get_visualization(self) -> NDArray[np.float32] | None:
        """
        Get visualization of the current physical state

        Returns:
            Visualization of current state or None if not available
        """
        if self.physical_state["current_imagery"] is not None:
            return enhance_raster_for_visualization(self.physical_state["current_imagery"])
        return None
