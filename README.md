# 3D Reconstruction with NeRF

This repository provides a framework for 3D reconstruction using Neural Radiance Fields (NeRF) and related models. It includes training, inference, and visualization tools, as well as dataset utilities for STL files and human models.

## Features
- NeRF-based 3D reconstruction
- Support for STL datasets
- Training and inference scripts
- Streamlit-based visualization tools
- Data loaders for various formats (Blender, COLMAP, LLFF, NSVF, etc.)
- Model implementations (MLP, SH, TensorBase, TensoRF)

## Directory Structure
```
main.py                  # Entry point for main operations
stl_dataset.py           # STL dataset utilities
utils.py                 # General utilities
few_nerf/                # Core NeRF implementation
	├── train.py           # Training script
	├── renderer.py        # Rendering utilities
	├── models/            # Model definitions
	├── dataLoader/        # Data loading modules
	├── configs/           # Configuration files
	├── extra/             # Additional scripts (metrics, auto-run)
streamlit_page/          # Streamlit visualization and inference
	├── inference_model/   # Inference scripts for STL models
	├── test_model/        # Testing and comparison scripts
	├── train_model/       # Training and data visualization scripts
```

## Installation
1. Clone the repository:
	 ```bash
	 git clone https://github.com/hautran7201/3D-reconstruction.git
	 cd 3D-reconstruction
	 ```
2. Install Python dependencies:
	 ```bash
	 pip install -r few_nerf/requirements.txt
	 ```

## Usage
- **Training:**
	```bash
	python few_nerf/train.py --config few_nerf/configs/lego.txt
	```
- **Inference & Visualization:**
	Use Streamlit scripts in `streamlit_page/` for interactive visualization:
	```bash
	streamlit run streamlit_page/inference_model/stl_inference.py
	```
- **Custom Dataset:**
	Modify or add dataset scripts in `stl_dataset.py` or `few_nerf/dataLoader/`.

## Requirements
- Python 3.8+
- See `few_nerf/requirements.txt` for full package list

## License
See `few_nerf/LICENSE` for license details.

## Citation
If you use this codebase for research, please cite the original NeRF paper and this repository.

---
For more details, see the documentation in each subfolder and script comments.

