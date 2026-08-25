import sys, os, tempfile, pathlib
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
    far = [-1.0] + [0.0]*127  # far from both enc1 and enc2
    name2, dist2 = fs.match(far)
    assert name2 is None and dist2 > 0.6

def test_frame_encoding_no_face():
    black = np.zeros((100,100,3), dtype=np.uint8)
    assert FaceSystem.frame_encoding(black) is None

if __name__ == "__main__":
    test_ear_open_gt_closed(); test_ear_thresholds(); test_blink_detector()
    with tempfile.TemporaryDirectory() as d:
        test_match_known_and_unknown(tmp_path=pathlib.Path(d))
    test_frame_encoding_no_face()
    print("ALL TESTS PASSED")
