<div align="center">

# 🔍 Deepfake Detection Project v5

[![Streamlit](https://img.shields.io/badge/Streamlit-%F0%9F%93%88-FF4B4B)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%F0%9F%94%A5-ee4c2c)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-%F0%9F%93%9A-3776ab)](https://github.com/huggingface/pytorch-image-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with Love](https://img.shields.io/badge/made%20with-%E2%9D%A4%EF%B8%8F-red.svg)](#)

</div>

Modern Streamlit app for deepfake detection on images, videos, and live camera. It ensembles two models (EfficientNet-B3 and Swin-Base) and presents predictions with confidence and visualizations. Includes training scripts for both backbones and a reproducible setup for pulling pre-trained weights from Kaggle.

### 🚀 Live demo

[Open the Streamlit app](https://ddp-v5.streamlit.app/)

If the app idles, it may take a few minutes to spin up and download models from Kaggle.

---

## 🧭 Table of Contents

- [Features](#-features)
- [Project structure](#-project-structure-high-level)
- [Quickstart (local)](#-quickstart-local)
- [Usage](#-usage)
  - [Image](#image)
  - [Video](#video)
  - [Live Camera (local only)](#live-camera-local-only)
- [How it works (inference)](#-how-it-works-inference)
- [Training](#-training)
- [Troubleshooting](#-troubleshooting)
- [Tech stack](#-tech-stack)
- [License](#-license)

---

## ✨ Features

- **🖼️ Image analysis**: face detection, model predictions, overall verdict, and side-by-side visuals
- **🎥 Video analysis**: frame sampling, per-frame face detection, model aggregation, sample faces grid
- **📷 Live camera (local only)**: real-time inference via webcam
- **🧠 Two complementary backbones**:
  - EfficientNet-B3 with custom classifier head
  - Swin Transformer (Base, patch4, window7, 224) with custom classifier head
- **⚡ Cached model loading** for fast, repeated inference
- **📈 Training pipelines** with MLflow logging, learning curves, confusion matrices, and ROC curves

---

## 🗂️ Project structure (high level)

- `app.py`: Streamlit UI and page flow
- `utils_*.py`: image/video processing, models, session/state management, formatting
- `training/`: reproducible training scripts and dataset pipeline
- `setup.sh`: downloads pre-trained weights from Kaggle into `runs/`
- `requirements.txt` and `packages.txt`: Python and system dependencies

---

## 🏁 Quickstart (local)

### 1) 📦 Python and system dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

On Linux (or Streamlit Cloud), you may also need these system packages:

```bash
cat packages.txt | xargs -I{} sudo apt-get install -y {}
```

Notes:

- On Windows, `opencv-python` (not headless) may be preferable if you need GUI/video codecs.
- GPU is optional. The app runs on CPU by default.

### 2) 🤖 Models

At runtime, if the `runs/` directory does not exist, `app.py` attempts to execute:

```bash
bash setup.sh
```

`setup.sh` does the following:

1. Downloads the Kaggle dataset `ameencaslam/ddp-v5-runs`
2. Unzips into `runs/`

This provides model weights at:

- `runs/models/efficientnet/best_model_cpu.pth`
- `runs/models/swin/best_model_cpu.pth`

If you prefer manual setup, create the above files at the same paths.

Kaggle CLI setup (only if using `setup.sh`):

```bash
pip install kaggle
# Place kaggle.json (with your API token) at ~/.kaggle/kaggle.json and chmod 600
```

### 3) 🧪 Run the app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. The home page lets you pick Image, Video, or Live Camera.

---

## 🧰 Usage

### Image

1. Select "Image"
2. Upload a JPG/PNG with a clear face
3. The app will detect a face, run EfficientNet and Swin models, and show:
   - Each model’s prediction and confidence
   - Final verdict banner (REAL/FAKE)
   - Visualization with detected face and cropped face

### Video

1. Select "Video"
2. Upload an MP4/AVI/MOV
3. Choose number of frames to analyze (slider)
4. The app extracts frames, detects faces, runs both models, and aggregates results per model
5. View final verdict and a grid of sample detected faces

### Live Camera (local only)

- Select "Live Camera" when running locally (webcam required)
- Pick a model and start/stop the camera stream
- The frame is annotated with prediction and confidence

---

## 🧩 How it works (inference)

- Face detection via MediaPipe (`utils_image_processor.extract_face`)
- Model-specific transforms (`utils_image_processor.get_transforms`):
  - EfficientNet: resize/center-crop to 300
  - Swin: resize/center-crop to 224
- Models are defined in `utils_eff.py` and `utils_swin.py`, and loaded with `utils_model.get_cached_model`
- Decision threshold: sigmoid(output) > 0.5 → FAKE, else REAL
- Overall verdict: if either model predicts FAKE, final verdict is FAKE

Expected weight paths:

- EfficientNet: `runs/models/efficientnet/best_model_cpu.pth`
- Swin: `runs/models/swin/best_model_cpu.pth`

---

## 🏋️‍♂️ Training

Training scripts live under `training/` and use MLflow for metrics/artifacts.

### Dataset expectations

`training/data_handler.py` expects a directory with the following:

- Split files at root: `train_split.txt`, `val_split.txt`, `test_split.txt`
- For each split, images under subfolders with relative paths listed in the split files, e.g.:
  - `train/fake/...`, `train/real/...`
  - `val/fake/...`, `val/real/...`
  - `test/fake/...`, `test/real/...`

Each split file contains relative paths like `fake/img_0001.jpg` or `real/img_0002.jpg`. Labels are inferred from the path prefix.

### EfficientNet training

```bash
python training/train_efficientnet.py
```

Key settings (edit in file as needed):

- `DATA_DIR = '/kaggle/input/2body-images-10k-split'`
- `IMAGE_SIZE = 300`, `BATCH_SIZE = 32`, `NUM_EPOCHS = 20`
- Optimizer: AdamW with different LRs for backbone/classifier
- Logs: MLflow (`mlruns/`), plus confusion matrix, ROC, and learning curves
- Saves best checkpoints to `models/efficientnet/`

### Swin training

```bash
python training/train_swin.py
```

Key settings:

- `DATA_DIR = '/kaggle/input/2body-images-10k-split'`
- `IMAGE_SIZE = 224`, `BATCH_SIZE = 32`, `NUM_EPOCHS = 20`
- Optimizer: AdamW with different LRs
- Logs: MLflow (`mlruns/`), plus confusion matrix, ROC, and learning curves
- Saves best checkpoints to `models/swin/`

To use your own trained weights in the app, copy your best CPU-state-dict files to:

- `runs/models/efficientnet/best_model_cpu.pth`
- `runs/models/swin/best_model_cpu.pth`

---

## 🛠️ Troubleshooting

- Streamlit Cloud cold start: first request triggers model download from Kaggle; wait for completion
- Kaggle CLI errors: ensure `~/.kaggle/kaggle.json` exists with correct permissions and that the dataset is accessible
- Missing `libGL`/X11 errors on Linux: install packages from `packages.txt`
- Webcam not working: Live Camera is disabled on deployed instances; run locally
- CUDA out-of-memory: the app runs on CPU by default; training scripts auto-detect GPU

---

## 🧱 Tech stack

- Streamlit UI
- PyTorch for modeling
- MediaPipe for face detection
- OpenCV/Pillow for image/video IO
- MLflow for experiment tracking
- `timm` and `efficientnet-pytorch` for backbones

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
