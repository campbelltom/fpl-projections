"""Turn raw model output into the JSON the site reads, and encrypt it with the site password."""
from __future__ import annotations

import base64
import json
import os

PBKDF2_ITER = 250_000


def encrypt_json(obj: dict, password: str) -> dict:
    """AES-256-GCM, key = PBKDF2-HMAC-SHA256(password, salt). Decrypted in the browser with WebCrypto."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=PBKDF2_ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, json.dumps(obj, separators=(",", ":")).encode(), None)
    b = lambda x: base64.b64encode(x).decode()  # noqa: E731
    return {"v": 1, "iter": PBKDF2_ITER, "salt": b(salt), "iv": b(iv), "ct": b(ct)}
