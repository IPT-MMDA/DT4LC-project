# HACK until we have a proper model loading mechanism and finalized models

import torch
import yaml

from cognitive_ui.config import RESOURCES_PATH, ROOT_DIR
from digital_twin.models.prithvi_v1.prithvi_mae import PrithviMAE

PRITHVI_MODEL_PATH = RESOURCES_PATH / "prithvi_eo_v1_100m"

# Load weights
weights_path = PRITHVI_MODEL_PATH / "Prithvi_EO_V1_100M.pt"
checkpoint = torch.load(weights_path, map_location="cpu")

# Read model config
model_cfg_path = ROOT_DIR / "digital_twin" / "models" / "prithvi_v1" / "config.yaml"  # HACK
with open(model_cfg_path) as f:
    model_config = yaml.safe_load(f)

model_args, train_args = model_config["model_args"], model_config["train_params"]

# Will use only 1 frame for now (the model was trained on 3 frames)
model_args["num_frames"] = 1

model = PrithviMAE(**model_args)
model.eval()

# strict=false since we are loading with only 1 frame, but the warning is expected
del checkpoint["encoder.pos_embed"]
del checkpoint["decoder.decoder_pos_embed"]
_ = model.load_state_dict(checkpoint, strict=False)
