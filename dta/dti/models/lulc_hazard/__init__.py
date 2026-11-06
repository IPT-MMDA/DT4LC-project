"""LULC change-hazard model (``models/lulc-change-hazard``).

:mod:`.features` builds the model's per-cell features from Earth Engine for any bounding box;
:mod:`.model` loads and checks a trained bundle and gives its probabilities and ranks;
:mod:`.inference` holds the registry runners that write the hazard ranks and the forecast map;
:mod:`.history` the past change, with no model; :mod:`.stats` the class-change numbers and sentences.
"""
