"""Utilities for invoking auxiliary Node.js scripts.

The liquidity planner relies on the official Meteora DLMM SDK, which is
maintained in TypeScript.  Rather than re-implementing every deposit strategy
from scratch, we shell out to a small Node.js helper that is bundled with this
project.  The helper must be installed via ``npm install`` inside the
``node_bridge`` directory before it can be used.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class NodeBridgeError(RuntimeError):
    """Raised when a Node.js helper script fails."""


class NodeBridge:
    """Thin wrapper around the Node.js helper scripts."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        if base_dir is not None:
            candidate = Path(base_dir).expanduser().resolve()
        else:
            candidate = self._discover_default_base_dir()
        self._base_dir = candidate

    def _discover_default_base_dir(self) -> Path:
        """Return the node helper directory bundled with the repository."""

        current = Path(__file__).resolve()
        for parent in current.parents:
            helper_dir = parent / "node_bridge"
            if (helper_dir / "package.json").exists():
                return helper_dir
        raise NodeBridgeError(
            "Unable to locate node_bridge helpers relative to the package; "
            "ensure the repository root is intact or provide base_dir explicitly."
        )

    def _ensure_environment(self) -> None:
        node_modules = self._base_dir / "node_modules"
        package_json = self._base_dir / "package.json"
        if not package_json.exists():
            raise NodeBridgeError(
                "Node helper package.json not found at "
                f"{package_json}; the repository layout looks unexpected"
            )
        if not node_modules.exists():
            raise NodeBridgeError(
                "Node dependencies missing. Run 'npm install' inside the node_bridge directory."
            )

    def run(self, script_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_environment()
        script_path = self._base_dir / script_name
        if not script_path.exists():
            raise NodeBridgeError(f"Node script {script_name} not found in {self._base_dir}")
        try:
            result = subprocess.run(
                ["node", script_path.name],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                cwd=self._base_dir,
                check=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - error path
            stderr = exc.stderr.strip()
            try:
                details = json.loads(stderr)
                message = details.get("error", stderr)
            except json.JSONDecodeError:
                message = stderr or str(exc)
            raise NodeBridgeError(message) from exc
        if not result.stdout:
            return {}
        return json.loads(result.stdout)


__all__ = ["NodeBridge", "NodeBridgeError"]
