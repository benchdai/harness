"""VerifiedState production signer stub.

This module will be connected to the VerifiedState API via MCP once
the production signing pipeline is ready.
"""

from typing import Optional

try:
    from .local import SignedManifest
except ImportError:
    # Allow this module to be imported even if PyNaCl is not installed.
    SignedManifest = None  # type: ignore[assignment,misc]


class VerifiedStateSigner:
    """Production signer that delegates to VerifiedState.

    In production, this will:
    1. Ingest the manifest into VS memory
    2. Run VS verification ladder
    3. Return the VS-signed receipt with Merkle root
    4. Link to VS explorer for independent verification

    For now, this is a stub that raises NotImplementedError.
    """

    def __init__(self, namespace_id: Optional[str] = None):
        self.namespace_id = namespace_id

    def sign_manifest(self, manifest: dict) -> "SignedManifest":
        """Sign a manifest via VerifiedState (not yet implemented).

        Raises:
            NotImplementedError: Always, until the VS integration is connected.
        """
        raise NotImplementedError(
            "VerifiedState signing is not yet connected. "
            "Use LocalSigner for development or set BENCHD_SIGNING_MODE=local"
        )
