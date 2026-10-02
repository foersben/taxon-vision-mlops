"""Scientific and architectural invariants verifying pure Pixi isolation and VSCode integration."""

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_no_virtualenv_or_uv_artifacts() -> None:
    """Invariant: The workspace must not contain a .venv/ directory or uv.lock."""
    venv_dir = ROOT / ".venv"
    assert not venv_dir.exists(), f"Stray virtualenv found at {venv_dir}. Pure Pixi environment violated."

    uv_lock = ROOT / "uv.lock"
    assert not uv_lock.exists(), f"Stray uv.lock found at {uv_lock}. Pure Pixi environment violated."


def test_pixi_lock_and_manifest_exist() -> None:
    """Invariant: pixi.lock and pyproject.toml with [tool.pixi] must exist."""
    pixi_lock = ROOT / "pixi.lock"
    assert pixi_lock.exists(), "Missing pixi.lock file."

    pyproject = ROOT / "pyproject.toml"
    assert pyproject.exists(), "Missing pyproject.toml file."

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert "tool" in data and "pixi" in data["tool"], "pyproject.toml must contain [tool.pixi] configuration."
    assert "dev" in data["tool"]["pixi"].get("environments", {}), "Missing 'dev' Pixi environment."


def test_vscode_interpreter_and_venv_routing() -> None:
    """Invariant: VSCode settings must route directly to .pixi/envs/dev/bin/python and exclude .venv."""
    settings_file = ROOT / ".vscode/settings.json"
    assert settings_file.exists(), "Missing .vscode/settings.json."

    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    assert settings.get("python.defaultInterpreterPath") == "${workspaceFolder}/.pixi/envs/dev/bin/python"
    assert settings.get("python.venvPath") == "${workspaceFolder}/.pixi/envs"
    assert settings.get("python.venvFolders") == [".pixi/envs"]
    assert settings.get("python.createEnvironment.trigger") == "off"
    assert settings.get("pixi.environment") == "dev"


def test_no_uv_in_active_pre_commit_hooks() -> None:
    """Invariant: Git pre-commit hooks and config must not invoke uv."""
    pre_commit_config = ROOT / ".pre-commit-config.yaml"
    assert pre_commit_config.exists()
    content = pre_commit_config.read_text(encoding="utf-8")
    assert "uv run" not in content, "Pre-commit config contains forbidden 'uv run' entry."
    assert "uvx" not in content, "Pre-commit config contains forbidden 'uvx' entry."
