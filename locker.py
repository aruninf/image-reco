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
