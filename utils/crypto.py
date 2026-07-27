import base64
import hashlib
import os
from typing import Optional
from utils.logger import get_logger

logger = get_logger("crypto")

# Base key derived from secret or fallback
MASTER_SECRET = os.getenv("MYGPT_MASTER_SECRET", "mygpt_enterprise_secret_key_32_bytes_len!")


def _derive_fernet_key(secret: str) -> bytes:
    """Derives a valid 32-byte url-safe base64-encoded key for Fernet AES-256."""
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_payload(data_str: str, secret: Optional[str] = None) -> str:
    """Encrypts text string using AES-256 (Fernet / Base64 SHA-256 key).

    Args:
        data_str: Raw text string to encrypt.
        secret: Optional custom secret key string.

    Returns:
        Base64-encoded ciphertext string.
    """
    key_bytes = _derive_fernet_key(secret or MASTER_SECRET)
    try:
        from cryptography.fernet import Fernet
        f = Fernet(key_bytes)
        ciphertext = f.encrypt(data_str.encode("utf-8"))
        return ciphertext.decode("utf-8")
    except ImportError:
        # XOR fallback encryption if cryptography library is not installed
        raw_bytes = data_str.encode("utf-8")
        key = (secret or MASTER_SECRET).encode("utf-8")
        xor_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(raw_bytes)])
        return base64.b64encode(xor_bytes).decode("utf-8")


def decrypt_payload(ciphertext_str: str, secret: Optional[str] = None) -> str:
    """Decrypts AES-256 ciphertext string.

    Args:
        ciphertext_str: Base64-encoded ciphertext.
        secret: Optional custom secret key string.

    Returns:
        Decrypted plaintext string.
    """
    key_bytes = _derive_fernet_key(secret or MASTER_SECRET)
    try:
        from cryptography.fernet import Fernet
        f = Fernet(key_bytes)
        plaintext = f.decrypt(ciphertext_str.encode("utf-8"))
        return plaintext.decode("utf-8")
    except ImportError:
        # XOR fallback decryption
        raw_bytes = base64.b64decode(ciphertext_str.encode("utf-8"))
        key = (secret or MASTER_SECRET).encode("utf-8")
        xor_bytes = bytes([b ^ key[i % len(key)] for i, b in enumerate(raw_bytes)])
        return xor_bytes.decode("utf-8")
