# Face Recognition System (v2)

A local, dependency-light face-recognition app: enroll known people (with a
blink to prove they're real), recognize them live, get alerted on unknown
faces, and open a face-locked encrypted notes "locker". No cloud, no raw
images stored — only 128-d face embeddings + metadata.

## Features

- **Enrollment** — capture a person via webcam; requires a **blink** (liveness)
  so a held photo can't be enrolled. Stores a 128-d embedding, not photos.
- **Live recognition** — identifies known people and draws name/confidence.
- **Unknown-person alert** — logs `Unknown person detected @ HH:MM:SS` when an
  unrecognized face appears (and a blink is seen, to avoid false triggers).
- **Face-locker** — per-person text notes, encrypted at rest (Fernet), opened
  only for a recognized + blink-verified person.

## How it works

```
Add Person  ->  known_faces.json (name + info + 128-d encoding)
Recognize   ->  webcam -> encoding -> compare to known_faces.json
Locker      ->  lockers/<safe_name>.enc (Fernet-encrypted notes)
```

Detection/encoding use `face_recognition` (dlib). Blink is detected via Eye
Aspect Ratio from facial landmarks. The GUI is a tiny web UI served by Python's
built-in `http.server` (no Jupyter needed).

## Setup

```bash
cd image-reco
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8000.

> Note: `face_recognition` pulls in `dlib`, which compiles on first install
> (can take a few minutes on macOS). `requirements.txt` pins `setuptools<81`
> because `face_recognition_models` needs `pkg_resources`.

## Using the GUI

1. Fill **Name / Role / Dept / Notes**, click **Add Person (blink)**, and blink
   at the camera until it captures ~8 frames.
2. Click **Recognize (live)** — your name appears in green; unknown faces log a
   red "Unknown" alert. Press `q` in the webcam window to quit.
3. With yourself recognized, click **Open My Locker**, edit notes, **Save
   Locker**. Notes are encrypted in `lockers/`.

## File reference

| File | Purpose |
|------|---------|
| `app.py` | Web GUI + camera loops (enroll / recognize / locker). |
| `face_system.py` | Encodings, matching, blink (EAR) — headless, unit-tested. |
| `locker.py` | Encrypted per-person note store — unit-tested. |
| `known_faces.json` | Known people (embeddings + info). **Local only, gitignored.** |
| `lockers/` | Encrypted notes. **Local only, gitignored.** |
| `requirements.txt` | Dependencies. |
| `tests/` | Assert-based self-checks for `face_system` and `locker`. |

## Tests

```bash
python tests/test_face_system.py   # EAR, blink detector, match, frame encoding
python tests/test_locker.py        # encrypt/decrypt round-trip, non-plaintext
```

## Cleanup / reset

- Soft reset (keep enrolled people): delete `known_faces.json` + `lockers/`,
  then re-enroll.
- Full reset: delete `known_faces.json` and `lockers/`.
- `venv/` can be deleted and recreated from `requirements.txt`.
