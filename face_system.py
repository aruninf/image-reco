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
