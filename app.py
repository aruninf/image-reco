import sys
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from face_recognizer import FaceRecognizerSystem

PORT = 8000
system = FaceRecognizerSystem()


def capture_stdout(fn, *args, **kwargs):
    """Run fn while capturing anything it prints, so the GUI can show it."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _send(self, text, status=200, ctype="application/json"):
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

        try:
            if parsed.path == "/capture":
                out = capture_stdout(
                    system.capture_face_data, name=name, target_count=60)
            elif parsed.path == "/train":
                out = capture_stdout(system.train_model)
            elif parsed.path == "/recognize":
                out = capture_stdout(
                    system.recognize_faces, confidence_threshold=70)
            else:
                out = "Unknown action"
        except Exception as e:  # surface errors to the GUI instead of crashing
            out = f"ERROR: {e}"

        self._send(out or "Done.")

    def log_message(self, *args):
        pass


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Face Recognition</title>
<style>
  body{font-family:system-ui,Arial;max-width:640px;margin:40px auto;padding:0 16px;color:#222}
  h1{font-size:22px}
  .row{display:flex;gap:8px;align-items:center;margin:10px 0}
  input{padding:8px;border:1px solid #ccc;border-radius:6px;flex:1}
  button{padding:9px 16px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer;font-size:14px}
  button.alt{background:#16a34a}
  button.rec{background:#dc2626}
  button:disabled{opacity:.5;cursor:default}
  #log{background:#0b1020;color:#9fe;padding:12px;border-radius:8px;height:240px;overflow:auto;white-space:pre-wrap;font-family:monospace;font-size:12px;margin-top:12px}
</style></head>
<body>
  <h1>Face Recognition System</h1>
  <div class="row">
    <input id="name" placeholder="Person name (used by Capture)">
    <button onclick="act('/capture?name='+encodeURIComponent(document.getElementById('name').value))">1. Capture Face</button>
  </div>
  <div class="row">
    <button class="alt" onclick="act('/train')">2. Train Model</button>
    <button class="rec" onclick="act('/recognize')">3. Recognize (webcam)</button>
  </div>
  <div id="log">Ready. Click a button. Webcam windows open on your desktop.\n</div>

<script>
function log(t){const el=document.getElementById('log');el.textContent+=t+"\\n";el.scrollTop=el.scrollHeight}
function act(url){
  log("> running "+url);
  const btns=document.querySelectorAll('button');btns.forEach(b=>b.disabled=true);
  fetch(url,{method:'POST'}).then(r=>r.json()).then(d=>{
    log(d.msg||'(no output)');
  }).catch(e=>log("NETWORK ERROR: "+e)).finally(()=>{
    btns.forEach(b=>b.disabled=false);
  });
}
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"GUI running at http://localhost:{PORT}  (Ctrl+C to stop)")
    # Single-threaded on purpose: OpenCV's GUI (cv2.imshow) must run on the
    # main thread on macOS, and a worker thread raises "Unknown C++ exception".
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
