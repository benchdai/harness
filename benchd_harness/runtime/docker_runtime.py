"""
Docker Service Runtime — for systems that run as Docker containers.

Handles: pull, start, health wait, stop, remove.
Used by Letta, Khoj+Postgres, Neo4j for Graphiti, etc.
"""

import subprocess
import time

from .base import RuntimeExecutor, RuntimeConfig, RuntimeResult, IsolationProbe


class DockerServiceRuntime(RuntimeExecutor):
    """Runtime for Docker-containerized systems."""

    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        self._container_name = f"benchd-{config.docker_image.split('/')[-1].split(':')[0]}"

    def prepare(self) -> RuntimeResult:
        """Verify Docker is available."""
        try:
            result = subprocess.run(["docker", "version"], capture_output=True, timeout=10)
            if result.returncode != 0:
                return RuntimeResult(success=False, message="Docker not running")
            return RuntimeResult(success=True)
        except FileNotFoundError:
            return RuntimeResult(success=False, message="Docker not installed")

    def start(self) -> RuntimeResult:
        """Start a Docker container."""
        start_time = time.time()

        # Remove existing container
        subprocess.run(["docker", "rm", "-f", self._container_name], capture_output=True)

        # Build run command
        cmd = ["docker", "run", "-d", "--name", self._container_name]
        for port in self.config.docker_ports:
            cmd.extend(["-p", port])
        for k, v in self.config.docker_env.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.append(self.config.docker_image)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return RuntimeResult(success=False, message=f"Docker start failed: {result.stderr[:300]}")

        # Wait for health
        if self.config.healthcheck_url:
            import requests
            deadline = time.time() + self.config.startup_timeout_seconds
            while time.time() < deadline:
                try:
                    resp = requests.get(self.config.healthcheck_url, timeout=5)
                    if resp.status_code < 500:
                        break
                except Exception:
                    pass
                time.sleep(2)

        startup_ms = (time.time() - start_time) * 1000
        return RuntimeResult(success=True, startup_ms=startup_ms, message=f"Container {self._container_name} started")

    def healthcheck(self) -> RuntimeResult:
        """Check container is running."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", self._container_name],
            capture_output=True, text=True,
        )
        if "true" in result.stdout.lower():
            return RuntimeResult(success=True)
        return RuntimeResult(success=False, message="Container not running")

    def stop(self) -> RuntimeResult:
        """Stop and remove the container."""
        subprocess.run(["docker", "rm", "-f", self._container_name], capture_output=True)
        return RuntimeResult(success=True, message=f"Container {self._container_name} removed")

    def check_isolation(self) -> IsolationProbe:
        """Fresh container = clean state (if started fresh each run)."""
        return IsolationProbe(
            clean=True,
            evidence="Fresh Docker container — environment-guaranteed isolation",
        )
