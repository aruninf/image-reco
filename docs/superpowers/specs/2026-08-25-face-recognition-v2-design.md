# Face Recognition System — v2 Design

**Date:** 2026-08-25
**Status:** Approved (design)
**Goal:** Extend the existing local face-recognition app with: known-person
recognition, unknown-person alerts, embedding-based storage (no raw images),
blink liveness (anti-photo-spoofing), and a per-person face-locked text-note
"locker". Keep it a dependency-light, local, single-machine desktop app.

---

## 1. Architecture & files

Keep the existing structure; replace the LBPH/Haar core with a dlib-based
embedding core and add a locker module.

| File | Role |
|------|------|
| `face_system.py` | New. `FaceSystem` class: detection, 128-d encodings, enrollment, recognition, blink liveness. Replaces `face_recognizer.py`. |
| `locker.py` | New. Encrypted per-person text-note store (Fernet). |
| `app.py` | Extended GUI (stdlib `http.server`): Add Person, Recognize, Open Locker. |
| `known_faces.json` | New. Persistent known-people database (name + info + encoding). |
| `lockers/` | New dir. One encrypted note file per person (`<safe_name>.enc`). |
| `requirements.txt` | Add `face_recognition`, `cryptography`. Keep `opencv-python`, `numpy`, `pillow`. |
| `README.md` | Update usage (new buttons, new deps, removed LBPH/haar). |
| `face_recognizer.py` | Removed (replaced by `face_system.py`). |
| `haarcascade_frontalface_default.xml`, `trained_model.yml`, `faces/` | Removed from repo (LBPH/Haar artifacts). `faces/` already gitignored. |

`face_system.py` is the only component that touches the camera and the
recognition model. `locker.py` has no camera dependency (pure crypto + file
IO). `app.py` wires the two together over HTTP.

---

## 2. Recognition engine (replaces LBPH + Haar)

- Library: **`face_recognition`** (dlib backend).
- Detection: `face_recognition.face_locations(frame, model="hog")` (CPU-friendly).
- Encoding: `face_recognition.face_encodings(frame, locations)[0]` → 128-d float
  vector.
- Identification: `face_recognition.face_distance(known_encodings, unknown_encoding)`
  → smallest distance wins. Threshold `MATCH_THRESHOLD = 0.6` (configurable).
  Distance < threshold → known (that person); else → "Unknown".
- No raw images stored. Only the 128-d encoding + metadata are persisted.

**Data format — `known_faces.json`:**
```json
[
  {
    "name": "Arun WT",
    "safe_name": "arun_wt",
    "info": { "role": "Admin", "dept": "IT", "notes": "Authorized" },
    "encoding": [0.12, -0.34, ... 128 floats ...]
  }
]
```

---

## 3. Enrollment ("Add Person")

1. GUI sends `name` + `info` (role/dept/notes fields).
2. Camera opens; for each frame: detect face, run **blink check** (see §4).
3. Once a blink is detected, capture encodings from the next few frames,
   average them into one encoding (robustness to jitter).
4. Save entry to `known_faces.json` (append; dedupe by `safe_name`).
5. Log "Enrolled <name>".
- Fail/abort: no face seen, or no blink within a timeout (e.g., 10s) → log error,
  do not save.

---

## 4. Liveness — blink challenge (Eye Aspect Ratio)

- Use `face_recognition.face_landmarks(frame)` → 6 points per eye.
- EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||).
- Blink detected when EAR drops below `BLINK_EAR = 0.21` then rises back above
  `BLINK_OPEN = 0.28` (hysteresis to avoid flicker). Both enrollment and
  recognition require one confirmed blink before proceeding, defeating a held
  photo.
- Thresholds tunable constants; documented in `face_system.py`.

---

## 5. Recognition + unknown alert

- Live loop: detect face → blink check → encode → `face_distance` to all known.
- If best distance < `MATCH_THRESHOLD`: overlay `{name} ({role})` (green box).
- Else: overlay "Unknown" (red box) and write a **GUI log line**:
  `"Unknown person detected @ <HH:MM:SS>"`. (No OS notification, per decision.)
- Multiple faces: process each independently (each gets its own label/box); the
  locker panel targets the largest/centered recognized face.
- Press `q` to quit (OpenCV window).

---

## 6. Face-locker

- Each known person has an encrypted note file `lockers/<safe_name>.enc`.
- Encryption: `cryptography.fernet.Fernet` with a key stored in a local
  key file `lockers/.key` (machine-local). Notes are never plaintext on disk.
  - *Limitation (documented):* unlock is gated by face + blink, not a secret
    password. Same-machine attackers with file access could read `.key`. For a
    personal local app this is acceptable; flag in README.
- Access flow: during Recognize, when a known person is identified AND blink
  passed, the GUI shows an "Open Locker" textarea pre-loaded with their decrypted
  notes. Edits → "Save" re-encrypts to `<safe_name>.enc`. Unknown → no panel.
- Notes are plain text (no rich formatting) in v1.

---

## 7. GUI (`app.py`)

- Keep stdlib `http.server`, single-threaded (OpenCV GUI must run on main
  thread — macOS constraint already validated).
- Inputs: Name, Role, Dept, Notes.
- Buttons:
  - **Add Person** → POST `/enroll` (name + info in body/query).
  - **Recognize (live)** → POST `/recognize` (opens webcam window).
  - **Open Locker** → POST `/locker/open` returns the recognized person's notes;
    **Save** → POST `/locker/save`.
- Log panel shows all events including unknown alerts.
- The recognized person's identity is held server-side (last recognized
  `safe_name`) so the locker endpoints know whose notes to open/save.

---

## 8. Dependencies

`requirements.txt`:
```
opencv-python
numpy
pillow
face_recognition
cryptography
```
- **Risk:** `dlib` (pulled by `face_recognition`) may require a local build on
  macOS. Mitigation: verify `pip install face_recognition` works in the venv
  early; if no wheel, install `cmake` first or use a prebuilt dlib wheel.
- Keep `opencv-contrib-python`? Not needed once Haar/LBPH removed; switch to
  plain `opencv-python`. (Will confirm `face_recognition` doesn't need contrib.)

---

## 9. Error handling & edge cases

- Camera open failure → log error, return (no crash).
- `known_faces.json` missing/empty → Recognize logs "No known people enrolled".
- No face in frame → skip, keep looping.
- Multiple faces → label each; locker targets the largest.
- Low light / detection fails → warn in log; rely on encoding averaging.
- Blink not detected within timeout → abort with log message (enrollment) or
  treat recognition as failed (no unlock).
- Corrupt/locked locker file → log error, don't crash.

---

## 10. Testing

- One `assert`-based self-check (`python face_system.py --selftest` or a
  `test_smoke.py`):
  - Two synthetic encodings: near-identical → distance < threshold (match);
    random → distance > threshold (unknown).
  - EAR function: closed-eye landmark set → blink True; open-eye → False.
- Otherwise manual run via GUI. No test framework.

---

## 11. Out of scope (v1)

- File attachments in locker, credential storage, external/webhook alerts,
  multi-machine sync, GPU acceleration, passive (model-based) anti-spoofing.
- Can be added later without re-architecting.
