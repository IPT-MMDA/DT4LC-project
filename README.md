# Cognitive Digital Twin for Land Cover Change Detection (DT4LC Project)

Scalable Digital Twin Models for Land Cover Change Detection Using Machine Learning

## Overview

This project implements a cognitive digital twin framework that enables interactive analysis of land cover changes using satellite imagery. At current stage it provides an intuitive Streamlit-based UI for exploring datasets, analyzing temporal changes, and querying the model for insights about environmental patterns.

## Features

- Interactive visualization of satellite imagery
- Temporal comparison of land cover changes
- AI-powered analysis and interpretation of environmental patterns
- Query interface for exploring specific aspects of detected changes
- Synthetic data generation for historical comparisons

## Installation

### Prerequisites

- Python 3.10 or higher
- UV package manager (recommended) or pip

### Installing UV

UV is a fast, reliable Python package installer and resolver. To install UV:

```bash
# macOS/Linux
curl -sSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

For more installation options, see the [UV documentation](https://github.com/astral-sh/uv).

### Setting up the environment

```bash
# Clone the repository
git clone https://github.com/your-org/dt4lc-project.git
cd dt4lc-project

# Create and activate a virtual environment with UV
uv venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install the package with development dependencies
uv pip install -e ".[dev,ui,models]"
```

## Data Management

### Downloading Model Weights and Sample Data

To run the application with pre-configured examples, you need to download the Prithvi model weights and sample data. Use the provided script:

```bash
# Download Prithvi model weights and sample data
python scripts/fetch_prithvi_v1_weight.py
```

This script will:

1. Download the pre-trained Prithvi model weights
2. Place them in the correct directory structure
3. Download sample satellite imagery datasets for testing

After running this script, the application will be ready to use with working examples.

## Running the Application

Once installed, you can run the application in several ways:

### Using the CLI Command

```bash
# Run with default settings
cdt

# Check version
cdt --version
```

### Using Streamlit Directly

```bash
# Run the Streamlit app directly
streamlit run cognitive_ui/app.py
```

### Using UV

```bash
# Run with UV
uv run cdt
```

The application will be available at [localhost](http://localhost:8501) by default.

## Configuration

### Streamlit Configuration

Streamlit settings are configured in `.streamlit/config.toml`. You can modify this file to change server behavior, themes, and other Streamlit-specific settings.

### Application Configuration

Application-specific settings are in `cognitive_ui/config.py`. This includes:

- Path definitions
- Visualization parameters
- UI settings
- Default query templates

## Project Structure

```text
dt4lc-project/
├── .streamlit/              # Streamlit configuration
├── cognitive_ui/            # Main application package
│   ├── app.py               # Streamlit application entry point
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Application configuration
│   ├── manager.py           # Digital twin management
│   ├── utils.py             # Utility functions
│   └── ui/                  # UI components
├── digital_twin/            # Core digital twin models
│   └── models/
│       └── prithvi_v1/      # Prithvi model implementation
├── resources/               # Data resources
├── scripts/                 # Utility scripts
└── pyproject.toml           # Project configuration
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{cognitive_digital_twin,
  author = {Anton Chernyatevich},
  title = {Cognitive Digital Twin for Land Cover Change Detection (DT4LC Project)},
  year = {2025},
  url = {https://github.com/IPT-MMDA/DT4LC-project}
}
```

## License

This project is licensed under the Research Use License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
