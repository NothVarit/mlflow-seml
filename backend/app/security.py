import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

SESSION_COOKIE_NAME = "app_session"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(7 * 24 * 60 * 60)))
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LENGTH = 64
FALLBACK_SESSION_SECRET = os.urandom(32)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _session_secret() -> bytes:
    configured_secret = os.getenv("SESSION_SECRET")
    if configured_secret:
        return configured_secret.encode("utf-8")
    return FALLBACK_SESSION_SECRET


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=KEY_LENGTH,
    )
    return "$".join(
        [
            "scrypt",
            str(SCRYPT_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            _b64encode(salt),
            _b64encode(digest),
        ]
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, n_value, r_value, p_value, salt, expected = password_hash.split("$", 5)
        if algorithm != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_b64decode(salt),
            n=int(n_value),
            r=int(r_value),
            p=int(p_value),
            dklen=KEY_LENGTH,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(_b64encode(digest), expected)


def create_session_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "expires_at": int(time.time()) + SESSION_TTL_SECONDS,
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(_session_secret(), body, hashlib.sha256).digest()
    return f"{_b64encode(body)}.{_b64encode(signature)}"


def read_session_token(token: str | None) -> dict[str, Any] | None:
    if not token or "." not in token:
        return None
    body_part, signature_part = token.split(".", 1)
    try:
        body = _b64decode(body_part)
        expected_signature = hmac.new(_session_secret(), body, hashlib.sha256).digest()
        actual_signature = _b64decode(signature_part)
        if not hmac.compare_digest(expected_signature, actual_signature):
            return None
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    if int(payload.get("expires_at", 0)) < int(time.time()):
        return None
    user_id = payload.get("user_id")
    if not isinstance(user_id, int):
        return None
    return payload
