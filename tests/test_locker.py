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
