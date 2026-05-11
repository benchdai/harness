"""Tests for benchd_harness.signing.local."""

import re

from benchd_harness.signing.local import (
    LocalSigner,
    generate_keypair,
    verify_signature,
)


class TestGenerateKeypair:
    def test_generate_keypair(self, tmp_path):
        private_path, public_path = generate_keypair(tmp_path)
        assert private_path.exists()
        assert public_path.exists()

        # Both files should contain valid hex strings
        private_hex = private_path.read_text().strip()
        public_hex = public_path.read_text().strip()
        assert re.fullmatch(r"[0-9a-f]+", private_hex)
        assert re.fullmatch(r"[0-9a-f]+", public_hex)


class TestSignAndVerify:
    def test_sign_and_verify(self):
        signer = LocalSigner()
        manifest = {"benchmark": "smoke-memory-v0", "score": 42}
        signed = signer.sign_manifest(manifest)
        assert verify_signature(signed.to_dict()) is True

    def test_verify_tampered_manifest(self):
        signer = LocalSigner()
        manifest = {"benchmark": "smoke-memory-v0", "score": 42}
        signed = signer.sign_manifest(manifest)

        tampered = signed.to_dict()
        tampered["manifest"]["score"] = 99
        assert verify_signature(tampered) is False

    def test_verify_tampered_signature(self):
        signer = LocalSigner()
        manifest = {"benchmark": "smoke-memory-v0", "score": 42}
        signed = signer.sign_manifest(manifest)

        tampered = signed.to_dict()
        # Flip the first hex digit of the signature
        sig = tampered["signature"]
        flipped = ("1" if sig[0] == "0" else "0") + sig[1:]
        tampered["signature"] = flipped
        assert verify_signature(tampered) is False


class TestFingerprint:
    def test_fingerprint_format(self):
        signer = LocalSigner()
        fp = signer.fingerprint
        assert len(fp) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", fp)


class TestLocalSignerFromFile:
    def test_load_keypair_from_file(self, tmp_path):
        private_path, _ = generate_keypair(tmp_path)
        signer = LocalSigner(private_key_path=private_path)
        manifest = {"test": True}
        signed = signer.sign_manifest(manifest)
        assert verify_signature(signed.to_dict()) is True
