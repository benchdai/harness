"""Signing module for the Bench'd harness.

Two modes:
- Local dev mode: Ed25519 via PyNaCl for offline testing/development
- VerifiedState production mode: delegates to VerifiedState API (stub)
"""

from .local import LocalSigner, SignedManifest, generate_keypair, verify_signature

__all__ = ["LocalSigner", "SignedManifest", "generate_keypair", "verify_signature"]
