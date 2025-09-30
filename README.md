# CSRNet Crowd Counting - Streamlit App

This project refactors the mentor-provided notebook logic into a modular Streamlit app for easy, efficient inference and visualization of crowd density using CSRNet.

## Structure
- `model.py` — CSRNet model definition and checkpoint loader helper.
- `utils.py` — Preprocessing, tensor/NumPy conversions, and heatmap overlay utilities.
- `streamlit_app.py` — Streamlit UI, cached model loading, and image inference.
- `requirements.txt` — Python dependencies.
- `deepvision.ipynb` — Original notebook (training/data prep references).

## Setup (Windows)
1. Create a virtual environment (recommended):
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   Note: Installing PyTorch may take time. If you want CUDA support, follow the official install matrix: https://pytorch.org/get-started/locally/
   For example (CUDA 12.x, adjust as needed):
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

## Run the app
```powershell
streamlit run streamlit_app.py
```
Streamlit will print a local URL (e.g., http://localhost:8501). Open it in your browser.

## Usage
- Optionally upload a trained CSRNet checkpoint (`.pth` or `.pt`) in the sidebar to use your trained weights. Without a checkpoint, the app uses ImageNet-initialized frontend weights.
- Upload an image (`.jpg`, `.jpeg`, `.png`).
- Click "Run Inference" to estimate crowd count and view the density heatmap overlay.

## Tips for Efficiency
- Model is loaded once and cached via `st.cache_resource`.
- GPU is used automatically if you enable the option and CUDA is available.
- Large images are resized to 512x512 for fast inference.

## Training (Optional)
Training utilities and data generation exist in `deepvision.ipynb`. You can train your own model there and export a checkpoint to use in the app:
- Place your checkpoint file (e.g., `best_model.pth`) locally, then upload it from the app sidebar.

## Troubleshooting
- If OpenCV DLL errors occur on Windows, reinstall `opencv-python`:
  ```powershell
  pip install --force-reinstall --no-cache-dir opencv-python
  ```
- If PyTorch install fails or CPU-only is desired, install CPU wheels from the default index:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- If the app cannot load your checkpoint, ensure it contains a `model_state_dict` or a compatible state dict for `CSRNet`.
