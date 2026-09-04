# Finger Symbol Recognition (Air Drawing)

An interactive air-drawing and real-time alphanumeric character recognition system powered by MediaPipe Hands and an optimized convolutional neural network (EMNIST CNN / ONNX Runtime).

---

## Features

- **Touchless Air Drawing**: Tracks the index fingertip to draw smooth lines in 2D camera space.
- **EMNIST 47-Class Recognition**: Classifies digits (0-9), uppercase letters (A-Z), and distinct lowercase letters (a, b, d, e, f, g, h, n, q, r, t) using an ONNX-exported CNN.
- **Live Preview Box**: Real-time bottom-right preview displaying the centered 28x28 bounding box (scaled to 80x80 px) alongside Top-3 predictions and confidence scores.
- **Hands-Free Gesture Control**: Save character, per-character backspace, canvas reset, and full-text horizontal palm swipe erase.
- **Resizable HD Window**: Default 720p/1080p camera capture with free window resizing and fullscreen support.
- **Configurable Settings**: Camera, colors, brush thickness, and gesture sensitivity thresholds managed via `config.yaml`.

---

## Gesture Reference

| Gesture | Hand State | Action |
| :--- | :--- | :--- |
| **Draw** | Index finger extended, others folded | Draw lines on the air canvas |
| **Save Character** | Open palm (static) | Recognize symbol and append to text |
| **Delete Character (Backspace)** | Pinky finger raised, others folded | Delete last character (`text[:-1]`) |
| **Clear Canvas** | Full closed fist | Clear current drawing on canvas |
| **Erase All Text** | Open palm moving horizontally (Swipe) | Clear entire accumulated text |

---

## Installation and Quick Start

### Using `uv` (Recommended)

1. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   uv run main.py
   ```

---

### Using standard `pip`

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

---

## Configuration (`config.yaml`)

```yaml
camera:
  index: 0
  width: 1280
  height: 720
  window_name: "Air Drawing"

drawing:
  main_color: [255, 0, 0]
  brush_thickness: 8
  text_block_color: [255, 255, 0]
  banner_alpha: 0.4
  confidence_threshold: 0.7

gestures:
  swipe_distance_threshold: 0.12
  swipe_ratio_threshold: 1.3
  swipe_time_window: 0.4
  swipe_cooldown: 0.8
  backspace_cooldown: 0.6
```

---

## Model Training

To retrain or fine-tune the CNN on the EMNIST Balanced dataset:

```bash
uv run python src/finger_symbol_recognition/model.py
```

The script downloads the dataset to `data/`, runs deterministic training with fixed seeds, and exports the model to `models/emnist_balanced.onnx`.

---

## Project Structure

```text
├── config.yaml               # Application configuration
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # EMNIST raw dataset files
├── models/                   # MediaPipe task and ONNX models
└── src/
    ├── main.ipynb            # Interactive development notebook
    └── finger_symbol_recognition/
        ├── __init__.py       # Package initialization
        ├── app.py            # AirDrawingApp application class
        ├── classifier.py     # EMNISTClassifier (ONNX Runtime)
        └── model.py          # EMNISTNet architecture and PyTorch training pipeline
```

---

## Keyboard Shortcuts

- `q` - Quit the application.
- `c` - Clear canvas and text.
