# Assignment 3: Object Detection and Recognition

This project implements object detection using **Faster R-CNN** on the **Penn-Fudan Database for Pedestrian Detection and Segmentation**. It also includes scripts for preparing the dataset for **YOLO** models.

## Features

- **Faster R-CNN**: Implementation using a pre-trained ResNet-50 backbone.
- **Data Preparation**: Automatic download and extraction of the Penn-Fudan Pedestrian dataset.
- **YOLO Conversion**: Scripts to convert the dataset into YOLO format (`images/labels`).
- **Dependency Management**: Uses `uv` for modern, fast Python package management.

## Prerequisites

- **Python**: >= 3.14
- **uv**: An extremely fast Python package installer and resolver.

## Installation

This project uses `uv` for dependency management. To set up the environment:

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the repository** (if applicable) and navigate to the project directory:
    ```bash
    cd assignment3
    ```

3.  **Sync dependencies**:
    ```bash
    uv sync
    ```
    This will create a virtual environment (`.venv`) and install all required packages specified in `pyproject.toml`.

## Usage

### 1. Data Preparation

To download the dataset and prepare it for YOLO training (if needed), run the preparation script:

```bash
uv run python prepare_data.py
```
This script will:
- Download `PennFudanPed.zip`.
- Extract it to the `data/` directory.
- Convert annotations to YOLO format in `data/yolo_data/`.

### 2. Running the Analysis (Jupyter Notebook)

The main analysis and Faster R-CNN training logic are contained in the Jupyter Notebook.

To launch the notebook server within the environment:

```bash
uv run jupyter notebook
```

Then open `Assignment_3_new.ipynb` in your browser to run the cells.

## Project Structure

- `Assignment_3_new.ipynb`: Main notebook for Faster R-CNN implementation.
- `prepare_data.py`: Script to download data and convert to YOLO format.
- `data/`: Directory where the dataset will be stored.
- `pyproject.toml`: Project dependencies and configuration.
- `uv.lock`: Locked dependencies for reproducible builds.

## Models

- **Faster R-CNN**: Retrained from `torchvision` weights (`FasterRCNN_ResNet50_FPN_Weights`).
- **YOLO**: Data prepared for Ultralytics YOLO models.

---
**Note**: Ensure you have a GPU environment set up if you plan to train large models, though the provided scripts handles device selection automatically (`cuda` or `cpu`).
