"""System configuration: config.yaml with environment-variable overrides.

The testbed is parameterised through one YAML file, ``config.yaml`` at the
repository root, validated against ``docs/schemas/config.schema.json``. Every
key can be overridden at runtime by the environment variable
``TESTBED_<KEY IN UPPER CASE>``. Secrets are never stored in the file: the SSH
password, when used, is read from the environment variable named by
``ssh_password_env``.

Usage::

    from src.config import load_config
    config = load_config()                  # config.yaml at the repository root, if present
    config = load_config("/path/to/config.yaml")
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "docs" / "schemas" / "config.schema.json"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_PREFIX = "TESTBED_"
CONFIG_PATH_ENV = "TESTBED_CONFIG"

# Values used when a key is absent from the file and the environment.
DEFAULTS: dict[str, Any] = {
    "ssh_username": "pi",
    "ssh_key_path": None,
    "ssh_password_env": "TESTBED_SSH_PASSWORD",
    "ssh_connect_timeout_s": 10.0,
    "ssh_known_hosts_path": None,
    "ssh_trust_new_hosts": False,
    "base_ip_prefix": "192.168.0.",
    "management_ip_prefix": None,
    "network_interface": "wlan0",
    "management_interface": "eth0",
    "bs_root": ".",
    "bs_scripts_dir": "./scripts",
    "bs_results_dir": "./results",
    "bs_models_dir": "./models",
    "ed_root": "/home/pi/testbed",
    "ed_scripts_dir": None,
    "ed_models_dir": None,
    "ed_data_dir": None,
    "ed_results_dir": None,
    "bs_computation_capacity": 2.0e10,
    "ntp_offset_threshold_ms": 1.0,
    "probe_period_s": 2.0,
    "nnpm_host": "0.0.0.0",
    "nnpm_port": 8001,
    "nnpm_basic_auth_env": None,
    "map_tiles_url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "map_offline": False,
    "headsplit_repo_url": "https://github.com/Dimitrios-Kafetzis/headsplit.git",
    "headsplit_commit": None,
    "headsplit_dir": None,
    "headsplit_master_port": 29500,
}


class ConfigError(ValueError):
    """Raised when the configuration file or an override is invalid."""


@dataclass(frozen=True)
class Config:
    """Validated configuration. Field names equal the YAML keys."""

    ssh_username: str
    ssh_key_path: str | None
    ssh_password_env: str | None
    ssh_connect_timeout_s: float
    ssh_known_hosts_path: str | None
    ssh_trust_new_hosts: bool
    base_ip_prefix: str
    management_ip_prefix: str | None
    network_interface: str
    management_interface: str
    bs_root: str
    bs_scripts_dir: str
    bs_results_dir: str
    bs_models_dir: str
    ed_root: str
    ed_scripts_dir: str | None
    ed_models_dir: str | None
    ed_data_dir: str | None
    ed_results_dir: str | None
    bs_computation_capacity: float
    ntp_offset_threshold_ms: float
    probe_period_s: float
    nnpm_host: str
    nnpm_port: int
    nnpm_basic_auth_env: str | None
    map_tiles_url: str
    map_offline: bool
    headsplit_repo_url: str
    headsplit_commit: str | None
    headsplit_dir: str | None
    headsplit_master_port: int
    # Directory that config.yaml was loaded from (or the repository root).
    config_dir: Path

    # --- Edge-server paths (resolved against config_dir) ---------------------

    def _bs_path(self, value: str) -> Path:
        root = (self.config_dir / self.bs_root).resolve()
        return (root / value).resolve() if not Path(value).is_absolute() else Path(value)

    @property
    def bs_root_path(self) -> Path:
        return (self.config_dir / self.bs_root).resolve()

    @property
    def bs_scripts_path(self) -> Path:
        return self._bs_path(self.bs_scripts_dir)

    @property
    def bs_results_path(self) -> Path:
        return self._bs_path(self.bs_results_dir)

    @property
    def bs_models_path(self) -> Path:
        return self._bs_path(self.bs_models_dir)

    # --- Edge-device paths (POSIX strings; the devices are remote) ----------

    @property
    def ed_code_dir(self) -> str:
        return f"{self.ed_root.rstrip('/')}/code"

    @property
    def ed_venv_dir(self) -> str:
        return f"{self.ed_root.rstrip('/')}/venv"

    @property
    def ed_scripts_path(self) -> str:
        return self.ed_scripts_dir or f"{self.ed_code_dir}/scripts/end_device"

    @property
    def ed_models_path(self) -> str:
        return self.ed_models_dir or f"{self.ed_root.rstrip('/')}/models"

    @property
    def ed_data_path(self) -> str:
        return self.ed_data_dir or f"{self.ed_root.rstrip('/')}/data"

    @property
    def ed_results_path(self) -> str:
        return self.ed_results_dir or f"{self.ed_root.rstrip('/')}/results"

    @property
    def headsplit_path(self) -> str:
        return self.headsplit_dir or f"{self.ed_root.rstrip('/')}/headsplit"

    @property
    def management_prefix(self) -> str:
        return self.management_ip_prefix or self.base_ip_prefix

    def env_overrides(self) -> dict[str, str]:
        """The configuration as TESTBED_* environment variables, for device-side scripts."""
        out: dict[str, str] = {}
        for f in fields(self):
            if f.name == "config_dir":
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            out[env_name(f.name)] = str(value).lower() if isinstance(value, bool) else str(value)
        out["TESTBED_ED_SCRIPTS_DIR"] = self.ed_scripts_path
        out["TESTBED_ED_MODELS_DIR"] = self.ed_models_path
        out["TESTBED_ED_DATA_DIR"] = self.ed_data_path
        out["TESTBED_ED_RESULTS_DIR"] = self.ed_results_path
        return out


def env_name(key: str) -> str:
    """Environment variable that overrides ``key``."""
    return ENV_PREFIX + key.upper()


def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        schema: dict[str, Any] = json.load(fh)
    return schema


def _coerce(key: str, raw: str, schema_types: list[str]) -> Any:
    """Convert an environment-variable string to the type the schema expects."""
    if raw == "" or raw.lower() == "null":
        if "null" in schema_types:
            return None
        raise ConfigError(f"{env_name(key)} may not be empty")
    if "boolean" in schema_types:
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"{env_name(key)}: expected a boolean, got {raw!r}")
    if "integer" in schema_types:
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"{env_name(key)}: expected an integer, got {raw!r}") from exc
    if "number" in schema_types:
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"{env_name(key)}: expected a number, got {raw!r}") from exc
    return raw


def _schema_types(prop: Mapping[str, Any]) -> list[str]:
    declared = prop.get("type", "string")
    return list(declared) if isinstance(declared, list) else [declared]


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Config:
    """Load, overlay and validate the configuration.

    Precedence: environment variable > configuration file > default. The file
    is optional; when ``config_path`` is None, ``$TESTBED_CONFIG`` and then
    ``<repository root>/config.yaml`` are tried.
    """
    env = os.environ if environ is None else environ
    schema = load_schema()
    properties: Mapping[str, Mapping[str, Any]] = schema["properties"]

    if config_path is None:
        config_path = env.get(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH
    path = Path(config_path)

    file_values: dict[str, Any] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path}: the configuration must be a mapping")
        file_values = loaded
    elif config_path is not None and Path(config_path) != DEFAULT_CONFIG_PATH:
        raise ConfigError(f"configuration file not found: {path}")

    merged: dict[str, Any] = dict(DEFAULTS)
    merged.update(file_values)
    for key, prop in properties.items():
        types = _schema_types(prop)
        # YAML 1.1 reads "2.0e10" as a string; accept numeric strings for numeric keys.
        if isinstance(merged.get(key), str) and ("number" in types or "integer" in types):
            with contextlib.suppress(ConfigError):  # the schema check below names the key
                merged[key] = _coerce(key, merged[key], types)
        raw = env.get(env_name(key))
        if raw is not None:
            merged[key] = _coerce(key, raw, types)

    try:
        jsonschema.validate(merged, schema)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise ConfigError(f"invalid configuration at {location}: {exc.message}") from exc

    config_dir = path.parent.resolve() if path.exists() else REPO_ROOT
    return Config(config_dir=config_dir, **merged)


def get_ssh_password(config: Config, environ: Mapping[str, str] | None = None) -> str | None:
    """The SSH password from the environment, or None when keys are used."""
    if not config.ssh_password_env:
        return None
    env = os.environ if environ is None else environ
    return env.get(config.ssh_password_env) or None


def main() -> None:
    """Print the effective configuration as JSON (paths resolved)."""
    import argparse

    parser = argparse.ArgumentParser(description="Show the effective testbed configuration.")
    parser.add_argument("--config", help="path to config.yaml (default: repository root)")
    parser.add_argument(
        "--env", action="store_true", help="print TESTBED_* assignments for device-side scripts"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.env:
        for name, value in sorted(config.env_overrides().items()):
            print(f"{name}={value}")
        return
    payload = {f.name: getattr(config, f.name) for f in fields(config) if f.name != "config_dir"}
    payload["config_dir"] = str(config.config_dir)
    payload["derived"] = {
        "bs_results_path": str(config.bs_results_path),
        "ed_scripts_path": config.ed_scripts_path,
        "ed_models_path": config.ed_models_path,
        "ed_results_path": config.ed_results_path,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
