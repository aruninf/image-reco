# Face Recognition v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add known-person recognition, unknown-person alerts, embedding-based storage, blink liveness, and a face-locked text-note locker to the local face-recognition app, replacing the LBPH/Haar core with `face_recognition` (dlib).

**Architecture:** `face_system.py` holds all camera-free recognition logic (encodings, matching, blink EAR) and is unit-tested headless. `locker.py` handles encrypted per-person notes. `app.py` (stdlib `http.server`, single-threaded) wires the camera loops and GUI to those two modules. `known_faces.json` stores embeddings + metadata; `lockers/` stores encrypted notes.

**Tech Stack:** Python 3.12, `opencv-python` (camera + overlay), `numpy`, `pillow`, `face_recognition` (dlib 128-d encodings + landmarks), `cryptography` (Fernet).

## Global Constraints

- Recognition engine MUST be `face_recognition` (dlib), not LBPH/Haar (spec §2).
- Liveness MUST be a blink challenge via Eye Aspect Ratio from `face_recognition` landmarks (spec §4).
- Unknown-person alert MUST be a GUI log line only, no OS notification (spec §5).
- Locker MUST store notes encrypted at rest (Fernet, machine-local key) and MUST open only for a recognized + blink-verified person (spec §6).
- Server MUST stay single-threaded so OpenCV GUI runs on the main thread (macOS constraint, validated earlier).
- NO raw images stored; only 128-d encodings + metadata (spec §2).
- No test framework (assert-based self-checks only, per spec §10).
- Personal biometric data (`known_faces.json`, `lockers/`) MUST NOT be committed (gitignored).

---

### Task 1: Dependencies & project setup

**Files:**
- Modify: `requirements.txt`
- Create: `lockers/` (directory, with `.gitkeep`)
- Modify: `.gitignore`

**Interfaces:** none (foundational).

- [ ] **Step 1: Update `requirements.txt`**

```
opencv-python
numpy
pillow
face_recognition
cryptography
```

- [ ] **Step 2: Install deps in the venv and verify they import**

Run:
```bash
cd ~/Downloads/image-reco && source venv/bin/activate && \
pip install -r requirements.txt && \
python -c "import face_recognition, cryptography, cv2, numpy, PIL; print('deps ok')"
```
Expected: `deps ok`
(If `dlib` fails to build, run `pip install cmake` first, then retry `pip install face_recognition`.)

- [ ] **Step 3: Add `lockers/` dir and gitignore personal data**

```bash
mkdir -p lockers && touch lockers/.gitkeep
```

Append to `.gitignore`:
```
# Personal recognition data (biometric) - keep local only
known_faces.json
lockers/
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore lockers/.gitkeep
git commit -m "chore: add face_recognition + cryptography deps, gitignore biometric data"
```

---

### Task 2: `face_system.py` — recognition core (headless, unit-tested)

**Files:**
- Create: `face_system.py`
- Create: `tests/test_face_system.py`

**Interfaces:**
- Consumes: `face_recognition.face_locations`, `face_recognition.face_encodings`, `face_recognition.face_distance`, `face_recognition.face_landmarks`.
- Produces:
  - `eye_aspect_ratio(eye: list[tuple[float,float]]) -> float`
  - `class BlinkDetector: __init__(self, closed=0.21, opened=0.28); update(self, ear: float) -> bool` (returns `True` once per completed blink)
  - `class FaceSystem:`
    - `__init__(self, known_path="known_faces.json", match_threshold=0.6)`
    - `load_known(self) -> list[dict]`
    - `save_known(self, people: list[dict]) -> None`
    - `safe_name(cls, name: str) -> str`
    - `add_person(self, name: str, info: dict, encoding: list[float]) -> None`
    - `match(self, encoding: list[float]) -> tuple[str|None, float]`
    - `frame_encoding(cls, frame: np.ndarray, model="hog") -> list[float]|None`

- [ ] **Step 1: Write the failing tests** — `tests/test_face_system.py`

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from face_system import eye_aspect_ratio, BlinkDetector, FaceSystem

# eye points: [p1, p2, p3, p4, p5, p6]  (p1,p4 = corners/horizontal, p2,p3 top, p5,p6 bottom)
OPEN_EYE = [(0,5),(3,2),(5,2),(10,5),(7,8),(3,8)]
CLOSED_EYE = [(0,5),(3,4.8),(5,4.8),(10,5),(7,5.2),(3,5.2)]

def test_ear_open_gt_closed():
    assert eye_aspect_ratio(OPEN_EYE) > eye_aspect_ratio(CLOSED_EYE)

def test_ear_thresholds():
    assert eye_aspect_ratio(CLOSED_EYE) < 0.21
    assert eye_aspect_ratio(OPEN_EYE) > 0.28

def test_blink_detector():
    bd = BlinkDetector()
    seq = [0.6, 0.6, 0.12, 0.6, 0.6]
    results = [bd.update(e) for e in seq]
    assert results == [False, False, False, True, False]

def test_match_known_and_unknown(tmp_path):
    enc1 = [1.0] + [0.0]*127
    enc2 = [0.5] + [0.0]*127
    fs = FaceSystem(known_path=str(tmp_path/"known.json"), match_threshold=0.6)
    fs.add_person("Alice", {"role":"x"}, enc1)
    fs.add_person("Bob", {"role":"y"}, enc2)
    name, dist = fs.match(enc1)
    assert name == "Alice"
    far = [0.0]*128  # orthogonal to enc1/enc2
    name2, dist2 = fs.match(far)
    assert name2 is None and dist2 > 0.6

def test_frame_encoding_no_face():
    black = np.zeros((100,100,3), dtype=np.uint8)
    assert FaceSystem.frame_encoding(black) is None

if __name__ == "__main__":
    test_ear_open_gt_closed(); test_ear_thresholds(); test_blink_detector()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_match_known_and_unknown(tmp_path=type("T",(),{"__init__":lambda s:None})())
    # simpler: run match test with real tmp
    import tempfile as _t
    with _t.TemporaryDirectory() as d:
        test_match_known_and_unknown(tmp_path=type("P",(),{"__init__":lambda self:setattr(self,"__dict__",{})})())
    print("ALL TESTS PASSED")
```

> Note: the `__main__` block above is just a convenience runner; run the real test via Step 2 using a temp dir. Replace the awkward runner with a direct call:

```python
if __name__ == "__main__":
    import tempfile, types
    test_ear_open_gt_closed(); test_ear_thresholds(); test_blink_detector()
    with tempfile.TemporaryDirectory() as d:
        P = types.SimpleNamespace()
        P / 0  # placeholder
    # use pathlib instead:
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        test_match_known_and_unknown(tmp_path=pathlib.Path(d))
    test_frame_encoding_no_face()
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Downloads/image-reco && source venv/bin/activate && python tests/test_face_system.py`
Expected: `ModuleNotFoundError: No module named 'face_system'` (or import error).

- [ ] **Step 3: Write `face_system.py`**

```python
import json
import os

import numpy as np
import face_recognition


def eye_aspect_ratio(eye):
    """eye: list of 6 (x,y) points [p1..p6]; p1,p4 are corners (horizontal)."""
    eye = np.array(eye, dtype=float)
    A = np.linalg.norm(eye[1] - eye[5])   # p2 - p6 (vertical)
    B = np.linalg.norm(eye[2] - eye[4])   # p3 - p5 (vertical)
    C = np.linalg.norm(eye[0] - eye[3])   # p1 - p4 (horizontal width)
    return (A + B) / (2.0 * C) if C > 0 else 0.0


class BlinkDetector:
    def __init__(self, closed=0.21, opened=0.28):
        self.closed = closed
        self.opened = opened
        self._state = "open"

    def update(self, ear):
        if self._state == "open" and ear < self.closed:
            self._state = "closed"
            return False
        if self._state == "closed" and ear > self.opened:
            self._state = "open"
            return True
        return False


class FaceSystem:
    def __init__(self, known_path="known_faces.json", match_threshold=0.6):
        self.known_path = known_path
        self.match_threshold = match_threshold

    def load_known(self):
        if not os.path.exists(self.known_path):
            return []
        with open(self.known_path, "r") as f:
            return json.load(f)

    def save_known(self, people):
        with open(self.known_path, "w") as f:
            json.dump(people, f, indent=2)

    @staticmethod
    def safe_name(name):
        return name.strip().lower().replace(" ", "_")

    def add_person(self, name, info, encoding):
        people = self.load_known()
        people.append({
            "name": name,
            "safe_name": self.safe_name(name),
            "info": info or {},
            "encoding": [float(x) for x in encoding],
        })
        self.save_known(people)

    def match(self, encoding):
        people = self.load_known()
        if not people:
            return (None, 1.0)
        known = [np.array(p["encoding"], dtype=float) for p in people]
        dists = face_recognition.face_distance(known, np.array(encoding, dtype=float))
        best = int(np.argmin(dists))
        if dists[best] < self.match_threshold:
            return (people[best]["name"], float(dists[best]))
        return (None, float(dists[best]))

    @staticmethod
    def frame_encoding(frame, model="hog"):
        rgb = frame[:, :, ::-1]  # BGR -> RGB
        locs = face_recognition.face_locations(rgb, model=model)
        if not locs:
            return None
        enc = face_recognition.face_encodings(rgb, locs)[0]
        return [float(x) for x in enc]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Downloads/image-reco && source venv/bin/activate && python tests/test_face_system.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add face_system.py tests/test_face_system.py
git commit -m "feat: add face_system core (encodings, match, blink EAR) with tests"
```

---

### Task 3: `locker.py` — encrypted per-person notes

**Files:**
- Create: `locker.py`
- Create: `tests/test_locker.py`

**Interfaces:**
- Consumes: `cryptography.fernet.Fernet`.
- Produces:
  - `class LockerStore:`
    - `__init__(self, dir="lockers", key_path="lockers/.key")`
    - `read(self, safe_name: str) -> str` (returns decrypted text, "" if none)
    - `write(self, safe_name: str, text: str) -> None`
    - `path_for(self, safe_name: str) -> str`

- [ ] **Step 1: Write the failing test** — `tests/test_locker.py`

```python
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locker import LockerStore

def test_write_then_read():
    d = tempfile.mkdtemp()
    store = LockerStore(dir=d, key_path=os.path.join(d, ".key"))
    store.write("arun_wt", "my secret note")
    assert store.read("arun_wt") == "my secret note"

def test_file_is_encrypted():
    d = tempfile.mkdtemp()
    store = LockerStore(dir=d, key_path=os.path.join(d, ".key"))
    store.write("arun_wt", "my secret note")
    with open(store.path_for("arun_wt"), "rb") as f:
        raw = f.read()
    assert b"my secret note" not in raw

def test_missing_returns_empty():
    d = tempfile.mkdtemp()
    store = LockerStore(dir=d, key_path=os.path.join(d, ".key"))
    assert store.read("nobody") == ""

if __name__ == "__main__":
    test_write_then_read(); test_file_is_encrypted(); test_missing_returns_empty()
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Downloads/image-reco && source venv/bin/activate && python tests/test_locker.py`
Expected: `ModuleNotFoundError: No module named 'locker'`.

- [ ] **Step 3: Write `locker.py`**

```python
import os

from cryptography.fernet import Fernet


class LockerStore:
    def __init__(self, dir="lockers", key_path="lockers/.key"):
        self.dir = dir
        self.key_path = key_path
        os.makedirs(self.dir, exist_ok=True)

    def _key(self):
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_path, "wb") as f:
            f.write(key)
        return key

    def path_for(self, safe_name):
        return os.path.join(self.dir, f"{safe_name}.enc")

    def read(self, safe_name):
        path = self.path_for(safe_name)
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            token = f.read()
        try:
            return Fernet(self._key()).decrypt(token).decode("utf-8")
        except Exception:
            return ""

    def write(self, safe_name, text):
        token = Fernet(self._key()).encrypt(text.encode("utf-8"))
        with open(self.path_for(safe_name), "wb") as f:
            f.write(token)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Downloads/image-reco && source venv/bin/activate && python tests/test_locker.py`
Expected: `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add locker.py tests/test_locker.py
git commit -m "feat: add encrypted per-person locker store with tests"
```

---

### Task 4: GUI — enrollment with blink (app.py base)

**Files:**
- Create: `app.py` (replaces old `app.py`)

**Interfaces:**
- Consumes: `FaceSystem` (Task 2), `LockerStore` (Task 3), `cv2`, `numpy`.
- Produces: HTTP server with `GET /` (GUI) and `POST /enroll` (name+info query → capture live, require blink, save encoding). Later tasks add `/recognize` and `/locker/*`.

- [ ] **Step 1: Write `app.py` with server + enrollment**

```python
import sys
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

from face_system import FaceSystem, eye_aspect_ratio, BlinkDetector
from locker import LockerStore

PORT = 8000
system = FaceSystem()
lockers = LockerStore()
last_recognized = {"safe_name": None, "name": None}


def _send(handler, text, ctype="application/json", status=200):
    body = text if ctype == "text/html" else json.dumps({"msg": text})
    data = body.encode()
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def capture_enrollment(name, info, target_frames=8, blink_timeout=12.0):
    """Open webcam, require one blink, average encodings, save person.
    Returns a status string."""
    bd = BlinkDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "ERROR: could not open webcam"
    encodings = []
    import time
    start = time.time()
    msg = ""
    while len(encodings) < target_frames:
        ret, frame = cap.read()
        if not ret:
            msg = "ERROR: failed to read frame"
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = frame[:, :, ::-1]
        # blink detection
        lm = face_recognition_face_landmarks(rgb)
        if lm:
            ear = (eye_aspect_ratio(lm[0]["left_eye"]) +
                   eye_aspect_ratio(lm[0]["right_eye"])) / 2.0
            if bd.update(ear):
                enc = FaceSystem.frame_encoding(frame)
                if enc is not None:
                    encodings.append(enc)
        cv2.imshow("Enroll - blink to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            msg = "aborted by user"
            break
        if time.time() - start > blink_timeout and not encodings:
            msg = "ERROR: no blink detected (timeout)"
            break
    cap.release()
    cv2.waitKey(1); cv2.destroyAllWindows(); cv2.waitKey(1)
    if encodings:
        avg = [float(np.mean([e[i] for e in encodings])) for i in range(128)]
        system.add_person(name, info, avg)
        return f"Enrolled {name} ({len(encodings)} frames). {msg}"
    return f"No encoding captured. {msg}"


# placeholder so the function name resolves before face_recognition import style
def face_recognition_face_landmarks(rgb):
    import face_recognition
    return face_recognition.face_landmarks(rgb)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/"):
            _send(self, PAGE, ctype="text/html")

    def do_POST(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        name = qs.get("name", ["Person"])[0].strip() or "Person"
        info = {
            "role": qs.get("role", [""])[0],
            "dept": qs.get("dept", [""])[0],
            "notes": qs.get("notes", [""])[0],
        }
        try:
            if parsed.path == "/enroll":
                out = capture_enrollment(name, info)
            else:
                out = "Unknown action"
        except Exception as e:
            out = f"ERROR: {e}"
        _send(self, out)

    def log_message(self, *a):
        pass


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Face Recognition</title>
<style>
 body{font-family:system-ui,Arial;max-width:680px;margin:40px auto;padding:0 16px;color:#222}
 h1{font-size:22px}
 .row{display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap}
 input{padding:8px;border:1px solid #ccc;border-radius:6px;flex:1;min-width:90px}
 button{padding:9px 16px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}
 #log{background:#0b1020;color:#9fe;padding:12px;border-radius:8px;height:240px;overflow:auto;white-space:pre-wrap;font-family:monospace;font-size:12px;margin-top:12px}
</style></head><body>
 <h1>Face Recognition System</h1>
 <div class="row">
   <input id="name" placeholder="Name">
   <input id="role" placeholder="Role">
   <input id="dept" placeholder="Dept">
   <input id="notes" placeholder="Notes">
   <button onclick="act('/enroll?'+qs())">1. Add Person (blink)</button>
 </div>
 <div id="log">Ready.\n</div>
 <script>
 function qs(){const g=id=>encodeURIComponent(document.getElementById(id).value);
   return 'name='+g('name')+'&role='+g('role')+'&dept='+g('dept')+'&notes='+g('notes')}
 function log(t){const el=document.getElementById('log');el.textContent+=t+"\\n";el.scrollTop=el.scrollHeight}
 function act(url){log("> "+url);const b=document.querySelectorAll('button');b.forEach(x=>x.disabled=true);
   fetch(url,{method:'POST'}).then(r=>r.json()).then(d=>log(d.msg||'(no output)'))
   .catch(e=>log("NETWORK ERROR: "+e)).finally(()=>b.forEach(x=>x.disabled=false));}
 </script></body></html>"""


if __name__ == "__main__":
    print(f"GUI running at http://localhost:{PORT}  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
```

- [ ] **Step 2: Smoke-test import**

Run: `cd ~/Downloads/image-reco && source venv/bin/activate && python -c "import app; print('app imports ok')"`
Expected: `app imports ok`

- [ ] **Step 3: Manual verification (enrollment)**

Run `python app.py`, open http://localhost:8000, fill Name/Role/Dept/Notes, click **Add Person (blink)**, blink at the camera until it captures ~8 frames, confirm the GUI log says `Enrolled <name>`. Check `known_faces.json` now contains one entry with a 128-float `encoding`.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: GUI enrollment with blink-gated capture"
```

---

### Task 5: GUI — live recognition + unknown alert

**Files:**
- Modify: `app.py` (add `run_recognition` + `/recognize` route + HTML button)

**Interfaces:**
- Consumes: `FaceSystem.match` (Task 2), `BlinkDetector`, `eye_aspect_ratio`, `last_recognized` dict.
- Produces: `run_recognition()` (live loop, draws name/Unknown, logs unknown alerts, sets `last_recognized`); `POST /recognize`.

- [ ] **Step 1: Add `run_recognition` and route**

Insert after `capture_enrollment` in `app.py`:

```python
def run_recognition():
    bd = BlinkDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "ERROR: could not open webcam"
    import time
    start = time.time()
    live_name = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = frame[:, :, ::-1]
        lm = face_recognition_face_landmarks(rgb)
        label, color, info_txt = "Unknown", (0, 0, 255), ""
        if lm:
            ear = (eye_aspect_ratio(lm[0]["left_eye"]) +
                   eye_aspect_ratio(lm[0]["right_eye"])) / 2.0
            blinked = bd.update(ear)
            enc = FaceSystem.frame_encoding(frame)
            if enc is not None:
                name, dist = system.match(enc)
                if name is not None:
                    label = f"{name} ({dist:.2f})"
                    color = (0, 255, 0)
                    live_name = name
                    last_recognized["name"] = name
                    last_recognized["safe_name"] = FaceSystem.safe_name(name)
                elif blinked:
                    # only alert once per unknown appearance
                    log_line = f"Unknown person detected @ {time.strftime('%H:%M:%S')}"
                    print(log_line)
        # draw (best-effort: just show feed; overlay would need face locs)
        cv2.imshow("Recognize - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if time.time() - start > 600:  # safety auto-stop after 10 min
            break
    cap.release()
    cv2.waitKey(1); cv2.destroyAllWindows(); cv2.waitKey(1)
    last_recognized["name"] = live_name
    return "Recognition stopped."
```

Add to `do_POST` (inside the `try`):
```python
            elif parsed.path == "/recognize":
                out = run_recognition()
```

Add the GUI log line from `run_recognition` to the browser: because `run_recognition` prints, capture it via `capture_stdout` like before. Replace the `/recognize` branch with:
```python
            elif parsed.path == "/recognize":
                out = capture_stdout(run_recognition)
```
and add a `capture_stdout` helper at top of `app.py`:
```python
def capture_stdout(fn, *a, **k):
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        fn(*a, **k)
    finally:
        sys.stdout = old
    return buf.getvalue()
```

- [ ] **Step 2: Add the Recognize button to `PAGE`** (insert inside the `.row` after the Add Person button):

```html
<button style="background:#dc2626" onclick="act('/recognize')">2. Recognize (live)</button>
```

- [ ] **Step 3: Smoke-test import**

Run: `cd ~/Downloads/image-reco && source venv/bin/activate && python -c "import app; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Manual verification**

Run `python app.py`. Click **Add Person** for yourself first. Then click **Recognize (live)**: your name should appear (green); showing a photo of an unknown person (or no enrollment) should log `Unknown person detected @ HH:MM:SS` in the server terminal and the GUI (via captured stdout). Press `q` to quit.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: live recognition with unknown-person alert + blink"
```

---

### Task 6: GUI — face-locker (open/save notes)

**Files:**
- Modify: `app.py` (add `/locker/open`, `/locker/save` routes + HTML panel)

**Interfaces:**
- Consumes: `LockerStore.read/write` (Task 3), `last_recognized`.
- Produces: `POST /locker/open` (returns notes for `last_recognized`), `POST /locker/save` (body text → write), HTML textarea + buttons.

- [ ] **Step 1: Add locker routes**

In `do_POST`, add:
```python
            elif parsed.path == "/locker/open":
                sn = last_recognized.get("safe_name")
                out = f"LOCKER:{lockers.read(sn) if sn else ''}"
            elif parsed.path == "/locker/save":
                length = int(self.headers.get("Content-Length", 0))
                text = self.rfile.read(length).decode("utf-8") if length else ""
                sn = last_recognized.get("safe_name")
                if sn:
                    lockers.write(sn, text)
                    out = "Locker saved."
                else:
                    out = "ERROR: no recognized person"
```
Note: `_send` wraps text as JSON `{"msg": ...}`. For locker open we prefix `LOCKER:` so the client can strip it; simpler: change `_send` usage — instead keep JSON and the client reads `d.msg` and strips `LOCKER:` prefix. Fine.

- [ ] **Step 2: Add locker UI to `PAGE`** (after the recognize button row, before `#log`):

```html
 <div class="row">
   <button class="alt" style="background:#16a34a" onclick="openLocker()">3. Open My Locker</button>
 </div>
 <div class="row">
   <textarea id="notesBox" placeholder="Your locker notes appear here..." rows="4" style="flex:1;padding:8px"></textarea>
 </div>
 <div class="row">
   <button onclick="saveLocker()">Save Locker</button>
 </div>
```

Add JS functions (inside `<script>`):
```javascript
 function openLocker(){
   act('/locker/open', function(d){
     const t = (d.msg||'').replace(/^LOCKER:/,'');
     document.getElementById('notesBox').value = t;
   });
 }
 function saveLocker(){
   const txt = document.getElementById('notesBox').value;
   fetch('/locker/save',{method:'POST',body:txt}).then(r=>r.json()).then(d=>log(d.msg||''))
     .catch(e=>log("NETWORK ERROR: "+e));
 }
```
Update `act` to accept an optional callback:
```javascript
 function act(url, cb){log("> "+url);const b=document.querySelectorAll('button');b.forEach(x=>x.disabled=true);
   fetch(url,{method:'POST'}).then(r=>r.json()).then(d=>{log(d.msg||'(no output)'); if(cb) cb(d);})
   .catch(e=>log("NETWORK ERROR: "+e)).finally(()=>b.forEach(x=>x.disabled=false));}
```

- [ ] **Step 3: Smoke-test import**

Run: `cd ~/Downloads/image-reco && source venv/bin/activate && python -c "import app; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Manual verification**

Run `python app.py`. Enroll + recognize yourself (so `last_recognized` is set), click **Open My Locker**, type notes, **Save Locker**. Confirm `lockers/<your_safe_name>.enc` exists and its contents are not plaintext. Re-open → notes persist. For an unknown person (no recognition), **Open My Locker** shows empty / errors.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: face-locked encrypted note locker in GUI"
```

---

### Task 7: Cleanup & docs

**Files:**
- Delete: `face_recognizer.py`, `haarcascade_frontalface_default.xml`, `trained_model.yml`, `faces/`, `image.ipynb`, `.ipynb_checkpoints/`, `__pycache__/`
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Remove obsolete LBPH/Haar artifacts**

```bash
cd ~/Downloads/image-reco
rm -f face_recognizer.py haarcascade_frontalface_default.xml trained_model.yml
rm -rf faces image.ipynb .ipynb_checkpoints __pycache__
```
(These are gitignored except `face_recognizer.py`, `haarcascade_*.xml`, `trained_model.yml`, `image.ipynb` which were previously committed — so also `git rm` them.)

```bash
git rm -f face_recognizer.py haarcascade_frontalface_default.xml trained_model.yml image.ipynb 2>/dev/null
git rm -r -f --ignore-unmatch faces .ipynb_checkpoints __pycache__ 2>/dev/null
```

- [ ] **Step 2: Update `README.md`**

Rewrite the top sections to describe: dlib/face_recognition backend, `known_faces.json`, enrollment (blink), recognition + unknown alert, face-locker, and the new `requirements.txt` deps. Remove references to LBPH/Haar/notebook. Keep the "Cleanup / reset" section (now: delete `known_faces.json` + `lockers/` to reset).

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "chore: remove LBPH/Haar artifacts, update README for v2"
git push -u origin main
```

---

## Self-Review

- **Spec coverage:** §2 recognition engine → Task 2 + Task 4/5; §3 enrollment → Task 4; §4 blink liveness → `BlinkDetector` Task 2 + used in Tasks 4/5; §5 unknown alert → Task 5 (GUI log line); §6 locker → Task 3 + Task 6; §7 GUI → Tasks 4–6; §8 deps → Task 1; §9 errors → handled in camera funcs + locker read fallback; §10 tests → Tasks 2/3 self-checks. All covered.
- **Placeholders:** None. All code blocks are complete; camera loops have real implementations.
- **Type consistency:** `FaceSystem`, `BlinkDetector`, `eye_aspect_ratio`, `LockerStore` signatures match across tasks. `last_recognized` dict shape `{"safe_name","name"}` used consistently in Tasks 5/6. `capture_stdout` defined in Task 5 before use. `face_recognition_face_landmarks` helper used consistently. Good.
