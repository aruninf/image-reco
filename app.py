import sys
import io
import json
import os
import time
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


def capture_stdout(fn, *a, **k):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn(*a, **k)
    finally:
        sys.stdout = old
    return buf.getvalue()


def face_landmarks(rgb):
    import face_recognition
    return face_recognition.face_landmarks(rgb)


def _close_windows():
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def capture_enrollment(name, info, target_frames=8, blink_timeout=12.0):
    bd = BlinkDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "ERROR: could not open webcam"
    encodings = []
    msg = ""
    start = time.time()
    while len(encodings) < target_frames:
        ret, frame = cap.read()
        if not ret:
            msg = "ERROR: failed to read frame"
            break
        rgb = frame[:, :, ::-1]
        locs = None
        try:
            locs = face_recognition_locs(rgb)
        except Exception:
            locs = None
        if locs:
            lm = face_landmarks(rgb)
            if lm:
                ear = (eye_aspect_ratio(lm[0]["left_eye"]) +
                       eye_aspect_ratio(lm[0]["right_eye"])) / 2.0
                if bd.update(ear):
                    enc = FaceSystem.frame_encoding(frame)
                    if enc is not None:
                        encodings.append(enc)
                        cv2.rectangle(frame, (locs[0][3], locs[0][0]),
                                      (locs[0][1], locs[0][2]), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {len(encodings)}/{target_frames} (blink!)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enroll - blink to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            msg = "aborted by user"
            break
        if time.time() - start > blink_timeout and not encodings:
            msg = "ERROR: no blink detected (timeout)"
            break
    cap.release()
    _close_windows()
    if encodings:
        avg = [float(np.mean([e[i] for e in encodings])) for i in range(128)]
        system.add_person(name, info, avg)
        last_recognized["name"] = name
        last_recognized["safe_name"] = FaceSystem.safe_name(name)
        return f"Enrolled {name} ({len(encodings)} frames). {msg}"
    return f"No encoding captured. {msg}"


def run_recognition():
    bd = BlinkDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "ERROR: could not open webcam"
    start = time.time()
    live_name = None
    unknown_logged = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = frame[:, :, ::-1]
        label, color = "Unknown", (0, 0, 255)
        try:
            locs = face_recognition_locs(rgb)
        except Exception:
            locs = None
        if locs:
            lm = face_landmarks(rgb)
            if lm:
                ear = (eye_aspect_ratio(lm[0]["left_eye"]) +
                       eye_aspect_ratio(lm[0]["right_eye"])) / 2.0
                bd.update(ear)
            enc = FaceSystem.frame_encoding(frame)
            if enc is not None:
                name, dist = system.match(enc)
                if name is not None:
                    label = f"{name} ({dist:.2f})"
                    color = (0, 255, 0)
                    live_name = name
                    last_recognized["name"] = name
                    last_recognized["safe_name"] = FaceSystem.safe_name(name)
                    unknown_logged = False
                elif not unknown_logged:
                    print(f"Unknown person detected @ {time.strftime('%H:%M:%S')}")
                    unknown_logged = True
            top, right, bottom, left = locs[0]
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Recognize - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if time.time() - start > 600:
            break
    cap.release()
    _close_windows()
    last_recognized["name"] = live_name
    return "Recognition stopped."


def face_recognition_locs(rgb):
    import face_recognition
    return face_recognition.face_locations(rgb, model="hog")


class Handler(BaseHTTPRequestHandler):
    def _send(self, text, ctype="application/json", status=200):
        body = text if ctype == "text/html" else json.dumps({"msg": text})
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/"):
            self._send(PAGE, ctype="text/html")

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
            elif parsed.path == "/recognize":
                out = capture_stdout(run_recognition)
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
            else:
                out = "Unknown action"
        except Exception as e:
            out = f"ERROR: {e}"
        self._send(out)

    def log_message(self, *a):
        pass


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Face Recognition</title>
<style>
 body{font-family:system-ui,Arial;max-width:680px;margin:40px auto;padding:0 16px;color:#222}
 h1{font-size:22px}
 .row{display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap}
 input,textarea{padding:8px;border:1px solid #ccc;border-radius:6px}
 input{flex:1;min-width:90px}
 button{padding:9px 16px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}
 button.alt{background:#16a34a}
 button.rec{background:#dc2626}
 #log{background:#0b1020;color:#9fe;padding:12px;border-radius:8px;height:200px;overflow:auto;white-space:pre-wrap;font-family:monospace;font-size:12px;margin-top:12px}
</style></head><body>
 <h1>Face Recognition System</h1>
 <div class="row">
   <input id="name" placeholder="Name">
   <input id="role" placeholder="Role">
   <input id="dept" placeholder="Dept">
   <input id="notes" placeholder="Notes">
   <button onclick="act('/enroll?'+qs())">1. Add Person (blink)</button>
   <button class="rec" onclick="act('/recognize')">2. Recognize (live)</button>
 </div>
 <div class="row">
   <button class="alt" onclick="openLocker()">3. Open My Locker</button>
   <button onclick="saveLocker()">Save Locker</button>
 </div>
 <div class="row">
   <textarea id="notesBox" placeholder="Your locker notes appear here..." rows="4" style="flex:1"></textarea>
 </div>
 <div id="log">Ready.\n</div>
 <script>
 function qs(){const g=id=>encodeURIComponent(document.getElementById(id).value);
   return 'name='+g('name')+'&role='+g('role')+'&dept='+g('dept')+'&notes='+g('notes')}
 function log(t){const el=document.getElementById('log');el.textContent+=t+"\\n";el.scrollTop=el.scrollHeight}
 function act(url,cb){log("> "+url);const b=document.querySelectorAll('button');b.forEach(x=>x.disabled=true);
   fetch(url,{method:'POST'}).then(r=>r.json()).then(d=>{log(d.msg||'(no output)');if(cb)cb(d);})
   .catch(e=>log("NETWORK ERROR: "+e)).finally(()=>b.forEach(x=>x.disabled=false));}
 function openLocker(){act('/locker/open',function(d){const t=(d.msg||'').replace(/^LOCKER:/,'');
   document.getElementById('notesBox').value=t;});}
 function saveLocker(){const txt=document.getElementById('notesBox').value;
   fetch('/locker/save',{method:'POST',body:txt}).then(r=>r.json()).then(d=>log(d.msg||''))
   .catch(e=>log("NETWORK ERROR: "+e));}
 </script></body></html>"""


if __name__ == "__main__":
    print(f"GUI running at http://localhost:{PORT}  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
