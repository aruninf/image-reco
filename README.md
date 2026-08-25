# Face Recognition System

A small OpenCV-based pipeline that captures faces from your webcam, trains an
LBPH (Local Binary Patterns Histograms) recognizer, and runs real-time
recognition with name + metadata overlay.

## How it works (data flow)

```
capture_face_data()  ->  faces/<person>/<n>.jpg  +  metadata.json
train_model()        ->  trained_model.yml  +  metadata.json (label_map)
recognize_faces()    ->  webcam window with name / role / dept overlay
```

1. **Capture** – opens the webcam, detects a face with the Haar cascade,
   preprocesses it (crop → resize to 200x200 → CLAHE contrast normalization →
   slight blur), and saves one image per frame into `faces/<name>/`.
2. **Train** – reads every `faces/<name>/*.jpg`, builds an LBPH model, and
   saves it as `trained_model.yml`. Also writes a `label_map` into
   `metadata.json` so each numeric label maps back to a name + info.
3. **Recognize** – opens the webcam, detects faces, predicts a label + distance
   (lower distance = better match), looks the label up in `metadata.json`, and
   draws the name/info on screen. Press `q` to quit.

## File-by-file

| File | What it is | Why it exists | Safe to delete? |
|------|-----------|---------------|-----------------|
| `face_recognizer.py` | Core module: the `FaceRecognizerSystem` class (capture / train / recognize / metadata). Also has a CLI menu when run directly (`python face_recognizer.py`). | The actual logic. | **No** – it's the program. |
| `image.ipynb` | Jupyter notebook UI wrapping the module in cells (capture / train / recognize). | Convenient interactive interface. | Yes, if you only use the CLI. |
| `haarcascade_frontalface_default.xml` | Pre-trained OpenCV Haar cascade for face **detection** (bounding boxes). | Required by `cv2.CascadeClassifier` to find faces. | Yes – auto-redownloaded (needs internet). Keep it offline. |
| `trained_model.yml` | Trained LBPH recognizer (histograms per label). | Loaded by `recognize_faces()`. | Yes – regenerate via `train_model()`. |
| `metadata.json` | Maps labels → `{name, folder, info}`. After training it holds only `{"label_map": {...}}`. | Lets recognition show names/role/dept instead of raw IDs. | Yes – regenerated on next train. |
| `faces/` | Captured training images, one subfolder per person (`faces/<name>/*.jpg`). | The dataset the model trains on. | Yes – deletes all training data (then retrain). |
| `venv/` | Python virtualenv with `cv2`, `numpy`, `pillow`, `jupyter`, `ipykernel`. ~2.4 GB. | Isolated deps so `cv2` import works. | Yes – but recreate it to run (see below). |
| `.ipynb_checkpoints/` | Auto-save backups of the notebook. | Jupyter safety net. | **Yes** – auto-regenerated. |
| `__pycache__/` | Compiled `.pyc` bytecode cache. | Python speedup. | **Yes** – auto-regenerated. |

## How to run (important: use the venv)

The notebook/kernel must run under the venv Python (which has `cv2`), NOT the
system/JupyterLab-bundled Python. Launch Jupyter from inside the venv:

```bash
cd ~/Downloads/image-reco
source venv/bin/activate
jupyter lab
```

Then open `image.ipynb` and run the cells top to bottom.
(You can also skip the notebook and run `python face_recognizer.py` for the CLI.)

## Cleanup / reset

Delete any of the marked-safe files freely. Common resets:

- **Soft reset (keep captured faces):** delete `trained_model.yml` + `metadata.json`, then re-run Train.
- **Full reset (start over):** delete `faces/`, `trained_model.yml`, `metadata.json`, `haarcascade_frontalface_default.xml`, `.ipynb_checkpoints/`, `__pycache__/`.
- **Recreate the venv if deleted:**
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install opencv-contrib-python numpy pillow jupyterlab ipykernel
  python -m ipykernel install --user --name image-reco-venv --display-name "image-reco (venv)"
  ```
