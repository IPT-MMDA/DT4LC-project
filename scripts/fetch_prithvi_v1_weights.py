import logging
import os

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

# Configure logging for better feedback
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
MODEL_FILENAME = "Prithvi_EO_V1_100M.pt"
EXAMPLES_DIR_PATTERN = "examples/*"  # Pattern to match files/folders within examples

BASE_TARGET_DIR = "resources/prithvi_eo_v1_100m"
MODEL_TARGET_DIR = BASE_TARGET_DIR  # Model file goes into the base directory
EXAMPLES_TARGET_DIR = os.path.join(BASE_TARGET_DIR, "examples")  # Examples go into a subdirectory

# --- Ensure target directories exist ---
try:
    logging.info(f"Ensuring base directory exists: {MODEL_TARGET_DIR}")
    os.makedirs(MODEL_TARGET_DIR, exist_ok=True)
    # snapshot_download will create the examples target dir if needed,
    # but creating it explicitly doesn't hurt.
    logging.info(f"Ensuring examples directory exists: {EXAMPLES_TARGET_DIR}")
    os.makedirs(EXAMPLES_TARGET_DIR, exist_ok=True)
    logging.info("Target directories ensured.")
except OSError as e:
    logging.error(f"Error creating directories: {e}")
    # Exit if directories can't be created
    exit(1)

# --- Download the single model file ---
logging.info(f"Attempting to download model file: {MODEL_FILENAME} from {REPO_ID}")
try:
    # hf_hub_download downloads a single file.
    # local_dir specifies the *directory* where snapshots related to this repo are stored.
    # The function returns the *full path* to the cached file, which might be in a subdirectory.
    # If you need the file *directly* in MODEL_TARGET_DIR without cache structure,
    # you might need to move it after download, but using the returned path is standard.
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        cache_dir=MODEL_TARGET_DIR,  # Use target dir as cache to place file inside
        local_dir=MODEL_TARGET_DIR,  # Explicitly request download *to* this dir structure
        local_dir_use_symlinks=False,  # Download actual file, not symlink to cache
        resume_download=True,  # Resume if download was interrupted
    )
    # Note: Due to caching mechanisms, the file might end up in a subdirectory like
    # 'resources/prithvi_eo_v1_300m/models--ibm-nasa-geospatial--Prithvi-EO-1.0-100M/snapshots/<hash>/'.
    # The returned `model_path` variable holds the correct location.
    logging.info(f"Model file downloaded successfully. Path: {model_path}")
    # If you absolutely need it directly in BASE_TARGET_DIR, uncomment and adapt the move below:
    # import shutil
    # final_model_dest = os.path.join(MODEL_TARGET_DIR, MODEL_FILENAME)
    # if os.path.abspath(model_path) != os.path.abspath(final_model_dest):
    #     logging.info(f"Moving model file to {final_model_dest}")
    #     shutil.move(model_path, final_model_dest)
    #     model_path = final_model_dest # Update path variable

except LocalEntryNotFoundError:
    logging.error(f"Model file '{MODEL_FILENAME}' not found in repo '{REPO_ID}'. Check filename and repo ID.")
except HfHubHTTPError as e:
    logging.error(f"HTTP error downloading model file: {e}")
except Exception as e:
    logging.error(f"An unexpected error occurred during model download: {e}", exc_info=True)


# --- Download the examples directory contents ---
logging.info(f"Attempting to download examples directory contents from {REPO_ID}")
try:
    # snapshot_download downloads multiple files matching patterns.
    # Use allow_patterns to get only the examples directory content.
    # Set local_dir to the desired final location for the examples contents.
    examples_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[EXAMPLES_DIR_PATTERN],  # Crucial: gets only things matching this pattern
        local_dir=EXAMPLES_TARGET_DIR,  # Target directory for examples content
        local_dir_use_symlinks=False,  # Download actual files
        ignore_patterns=["*.git*", "*/.gitattributes"],  # Avoid downloading git files
        resume_download=True,  # Resume if download was interrupted
        # Using cache_dir=EXAMPLES_TARGET_DIR might help place files more directly,
        # but local_dir usually handles the final placement structure.
        cache_dir=EXAMPLES_TARGET_DIR,
    )
    # snapshot_download returns the path to the directory holding the downloaded files.
    # With local_dir set, this should be EXAMPLES_TARGET_DIR itself.
    logging.info(f"Examples directory contents downloaded successfully to: {examples_path}")

except HfHubHTTPError as e:
    logging.error(f"HTTP error downloading examples directory: {e}")
except Exception as e:
    logging.error(f"An unexpected error occurred during examples download: {e}", exc_info=True)

logging.info("Download script finished.")
