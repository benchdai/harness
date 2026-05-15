"""
Bench'd Verify — programmatic receipt verification.

This module can be used standalone or via CLI:

  from benchd_harness.verify import verify_receipt, fetch_and_verify
  result = verify_receipt("path/to/manifest.signed.json")
  result = fetch_and_verify("https://benchd.ai/api/receipt/run_xxx")

Or via pip:
  pip install benchd-harness
  benchd verify manifest.signed.json
"""

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VerificationResult:
    """Result of verifying a Bench'd receipt."""
    valid: bool
    run_id: str = ""
    system: str = ""
    benchmark: str = ""
    score: float = 0.0
    signed_at: str = ""
    fingerprint: str = ""
    manifest_hash: str = ""
    computed_hash: str = ""
    error: str = ""
    receipt_url: str = ""


def verify_receipt(path: str | Path) -> VerificationResult:
    """
    Verify a signed benchmark manifest from a local file.

    Checks:
    1. JSON structure is valid
    2. Required fields present (manifest, signature, public_key)
    3. Manifest hash matches computed hash
    4. Ed25519 signature is valid (if pynacl available)

    Returns VerificationResult with valid=True/False.
    """
    path = Path(path)

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return VerificationResult(valid=False, error=f"Cannot read file: {e}")

    return _verify_data(data)


def verify_json(data: dict) -> VerificationResult:
    """Verify a manifest from a parsed JSON dict."""
    return _verify_data(data)


def fetch_and_verify(url: str) -> VerificationResult:
    """
    Fetch a receipt from a URL and verify it.

    Works with benchd.ai/api/receipt/run_xxx endpoints.
    """
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = _verify_data(data)
        result.receipt_url = url
        return result
    except ImportError:
        return VerificationResult(valid=False, error="requests package required: pip install requests")
    except Exception as e:
        return VerificationResult(valid=False, error=f"Fetch failed: {e}")


def _verify_data(data: dict) -> VerificationResult:
    """Core verification logic."""

    # Check required fields
    if "manifest" not in data:
        return VerificationResult(valid=False, error="Missing 'manifest' field")
    if "signature" not in data:
        return VerificationResult(valid=False, error="Missing 'signature' field")
    if "public_key" not in data:
        return VerificationResult(valid=False, error="Missing 'public_key' field")

    # Parse manifest
    manifest = data["manifest"]
    if isinstance(manifest, str):
        try:
            manifest = json.loads(manifest)
        except json.JSONDecodeError:
            return VerificationResult(valid=False, error="Manifest is not valid JSON")

    # Extract metadata
    run_id = manifest.get("run_id", "")
    system = manifest.get("system", {})
    system_name = system.get("name", "") if isinstance(system, dict) else str(system)
    benchmark = manifest.get("benchmark", {})
    benchmark_name = benchmark.get("name", "") if isinstance(benchmark, dict) else str(benchmark)
    scores = manifest.get("scores", {})
    overall = scores.get("verified", {}).get("overall") or scores.get("nuance", {}).get("overall") or 0

    # Verify manifest hash
    manifest_str = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    computed_hash = hashlib.sha256(manifest_str.encode()).hexdigest()
    stored_hash = data.get("manifest_hash", "")

    hash_valid = computed_hash == stored_hash if stored_hash else True  # Skip if no stored hash

    # Verify Ed25519 signature
    sig_valid = True
    try:
        from nacl.signing import VerifyKey
        public_key_hex = data["public_key"]
        signature_hex = data["signature"]

        verify_key = VerifyKey(bytes.fromhex(public_key_hex))
        verify_key.verify(manifest_str.encode(), bytes.fromhex(signature_hex))
        sig_valid = True
    except ImportError:
        pass  # pynacl not installed, skip crypto verification
    except Exception as e:
        sig_valid = False

    valid = hash_valid and sig_valid

    return VerificationResult(
        valid=valid,
        run_id=run_id,
        system=system_name,
        benchmark=benchmark_name,
        score=overall,
        signed_at=data.get("signed_at", ""),
        fingerprint=data.get("signing_key_fingerprint", ""),
        manifest_hash=stored_hash,
        computed_hash=computed_hash,
        error="" if valid else "Signature or hash verification failed",
    )
