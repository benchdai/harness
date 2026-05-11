"""Local Ed25519 signing using PyNaCl for development and testing."""

import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder


@dataclass
class SignedManifest:
    """A manifest with its cryptographic signature."""

    manifest: dict
    manifest_hash: str  # SHA-256 of canonical manifest JSON
    signature: str  # hex-encoded Ed25519 signature
    public_key: str  # hex-encoded public key
    signing_key_fingerprint: str  # first 16 chars of SHA-256 of public key
    signed_at: str  # ISO datetime
    signing_mode: str  # "local" or "verifiedstate"

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _canonical_json(manifest: dict) -> bytes:
    """Produce canonical JSON bytes: sorted keys, no whitespace."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    """Return hex-encoded SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def _fingerprint_from_public_key_hex(public_key_hex: str) -> str:
    """First 16 chars of SHA-256 of the public key hex string."""
    return hashlib.sha256(public_key_hex.encode("utf-8")).hexdigest()[:16]


class LocalSigner:
    """Ed25519 signer using local keypair for development/testing."""

    def __init__(self, private_key_path: Optional[Path] = None):
        """Load or generate a keypair.

        Args:
            private_key_path: Path to a hex-encoded private key file.
                If None or the file does not exist, a new keypair is generated
                in memory (not persisted).
        """
        if private_key_path is not None:
            path = Path(private_key_path)
            if path.exists():
                key_hex = path.read_text().strip()
                self._signing_key = SigningKey(
                    key_hex.encode("utf-8"), encoder=HexEncoder
                )
            else:
                self._signing_key = SigningKey.generate()
        else:
            self._signing_key = SigningKey.generate()

        self._verify_key = self._signing_key.verify_key

    def sign_manifest(self, manifest: dict) -> SignedManifest:
        """Sign a manifest dictionary.

        1. Serialize manifest to canonical JSON (sorted keys, no whitespace)
        2. SHA-256 hash the canonical JSON
        3. Ed25519 sign the hash bytes
        4. Return SignedManifest with all fields populated
        """
        canonical = _canonical_json(manifest)
        manifest_hash = _sha256_hex(canonical)
        hash_bytes = manifest_hash.encode("utf-8")

        signed = self._signing_key.sign(hash_bytes, encoder=HexEncoder)
        signature_hex = signed.signature.decode("utf-8")

        return SignedManifest(
            manifest=manifest,
            manifest_hash=manifest_hash,
            signature=signature_hex,
            public_key=self.public_key_hex,
            signing_key_fingerprint=self.fingerprint,
            signed_at=datetime.now(timezone.utc).isoformat(),
            signing_mode="local",
        )

    @property
    def public_key_hex(self) -> str:
        """Hex-encoded public key."""
        return self._verify_key.encode(encoder=HexEncoder).decode("utf-8")

    @property
    def fingerprint(self) -> str:
        """First 16 chars of SHA-256 of the public key hex string."""
        return _fingerprint_from_public_key_hex(self.public_key_hex)


def generate_keypair(output_dir: Path) -> Tuple[Path, Path]:
    """Generate a new Ed25519 keypair and save to disk.

    Saves ``private.key`` and ``public.key`` as hex-encoded files in
    *output_dir*.

    Returns:
        (private_key_path, public_key_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signing_key = SigningKey.generate()
    verify_key = signing_key.verify_key

    private_path = output_dir / "private.key"
    public_path = output_dir / "public.key"

    private_hex = signing_key.encode(encoder=HexEncoder).decode("utf-8")
    public_hex = verify_key.encode(encoder=HexEncoder).decode("utf-8")

    private_path.write_text(private_hex + "\n")
    public_path.write_text(public_hex + "\n")

    # Restrict permissions on private key
    private_path.chmod(0o600)

    return private_path, public_path


def verify_signature(signed_manifest: dict) -> bool:
    """Verify a SignedManifest dict.

    1. Extract manifest, signature, and public_key
    2. Recompute SHA-256 of canonical manifest JSON
    3. Verify Ed25519 signature against the hash

    Args:
        signed_manifest: A dictionary with at least ``manifest``,
            ``signature``, and ``public_key`` keys (as produced by
            ``SignedManifest.to_dict()``).

    Returns:
        True if the signature is valid, False otherwise.
    """
    try:
        manifest = signed_manifest["manifest"]
        signature_hex = signed_manifest["signature"]
        public_key_hex = signed_manifest["public_key"]

        verify_key = VerifyKey(
            public_key_hex.encode("utf-8"), encoder=HexEncoder
        )

        canonical = _canonical_json(manifest)
        manifest_hash = _sha256_hex(canonical)
        hash_bytes = manifest_hash.encode("utf-8")

        signature_bytes = bytes.fromhex(signature_hex)
        verify_key.verify(hash_bytes, signature_bytes)
        return True
    except Exception:
        return False


def load_signed_manifest(path: Path) -> SignedManifest:
    """Load a signed manifest from a JSON file.

    Args:
        path: Path to a JSON file containing a serialized SignedManifest.

    Returns:
        A SignedManifest instance.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    return SignedManifest(
        manifest=data["manifest"],
        manifest_hash=data["manifest_hash"],
        signature=data["signature"],
        public_key=data["public_key"],
        signing_key_fingerprint=data["signing_key_fingerprint"],
        signed_at=data["signed_at"],
        signing_mode=data["signing_mode"],
    )
