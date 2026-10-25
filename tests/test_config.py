from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src import config as config_module
from src.config import DEFAULTS, ConfigError, env_name, get_ssh_password, load_config

pytestmark = pytest.mark.usefixtures("clean_env")


def write_config(tmp_path: Path, **values: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_defaults_when_no_file(tmp_path: Path) -> None:
    # The repository-root default path may not exist: load_config then uses the defaults.
    config = load_config(config_module.DEFAULT_CONFIG_PATH, environ={})
    assert config.ssh_username == DEFAULTS["ssh_username"]
    assert config.base_ip_prefix == "192.168.0."
    assert config.ed_scripts_path == "/home/pi/testbed/code/scripts/end_device"
    assert config.ed_models_path == "/home/pi/testbed/models"
    assert config.ed_results_path == "/home/pi/testbed/results"
    assert config.management_prefix == config.base_ip_prefix


def test_explicit_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", environ={})


def test_file_values_override_defaults(tmp_path: Path) -> None:
    path = write_config(tmp_path, ssh_username="dimitris", nnpm_port=9000, ed_root="/opt/tb")
    config = load_config(path, environ={})
    assert config.ssh_username == "dimitris"
    assert config.nnpm_port == 9000
    assert config.ed_scripts_path == "/opt/tb/code/scripts/end_device"
    assert config.config_dir == tmp_path.resolve()
    assert config.bs_results_path == (tmp_path / "results").resolve()


def test_example_file_validates(repo_root: Path) -> None:
    config = load_config(repo_root / "config.example.yaml", environ={})
    assert config.nnpm_port == 8001
    assert config.ed_root == "/home/pi/testbed"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, ssh_user="pi")
    with pytest.raises(ConfigError, match="ssh_user"):
        load_config(path, environ={})


def test_wrong_type_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, nnpm_port="eight")
    with pytest.raises(ConfigError, match="nnpm_port"):
        load_config(path, environ={})


def test_every_key_has_an_env_override(tmp_path: Path) -> None:
    """Every schema key can be overridden by TESTBED_<KEY>, with type coercion."""
    schema = config_module.load_schema()
    samples: dict[str, tuple[str, object]] = {}
    for key, prop in schema["properties"].items():
        types = prop["type"] if isinstance(prop["type"], list) else [prop["type"]]
        if "boolean" in types:
            samples[key] = ("true", True)
        elif "integer" in types:
            samples[key] = ("4242", 4242)
        elif "number" in types:
            samples[key] = ("3.5", 3.5)
        elif key.endswith("_prefix"):
            samples[key] = ("10.9.", "10.9.")
        elif key == "headsplit_commit":
            samples[key] = ("abcdef1", "abcdef1")
        else:
            samples[key] = ("value-x", "value-x")
    path = write_config(tmp_path)
    environ = {env_name(k): raw for k, (raw, _) in samples.items()}
    config = load_config(path, environ=environ)
    for key, (_, expected) in samples.items():
        assert getattr(config, key) == expected, key


def test_env_null_clears_nullable_key(tmp_path: Path) -> None:
    path = write_config(tmp_path, ssh_key_path="/x")
    config = load_config(path, environ={"TESTBED_SSH_KEY_PATH": "null"})
    assert config.ssh_key_path is None


def test_env_bad_boolean(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    with pytest.raises(ConfigError, match="boolean"):
        load_config(path, environ={"TESTBED_MAP_OFFLINE": "maybe"})


def test_config_path_from_environment(tmp_path: Path) -> None:
    path = write_config(tmp_path, ssh_username="fromenv")
    config = load_config(environ={"TESTBED_CONFIG": str(path)})
    assert config.ssh_username == "fromenv"


def test_ssh_password_only_from_environment(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    config = load_config(path, environ={})
    assert get_ssh_password(config, environ={}) is None
    assert get_ssh_password(config, environ={"TESTBED_SSH_PASSWORD": "s3cret"}) == "s3cret"


def test_env_overrides_export(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, map_offline=True), environ={})
    exported = config.env_overrides()
    assert exported["TESTBED_MAP_OFFLINE"] == "true"
    assert exported["TESTBED_ED_SCRIPTS_DIR"] == config.ed_scripts_path
    assert "TESTBED_SSH_KEY_PATH" not in exported  # None values are not exported


def test_schema_and_example_cover_the_same_keys(repo_root: Path) -> None:
    schema = config_module.load_schema()
    example = yaml.safe_load((repo_root / "config.example.yaml").read_text(encoding="utf-8"))
    assert set(example) == set(schema["properties"]) == set(DEFAULTS)


def test_cli_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config(tmp_path, ssh_username="cli")
    monkeypatch.setattr("sys.argv", ["config", "--config", str(path)])
    config_module.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["ssh_username"] == "cli"
    assert payload["derived"]["ed_scripts_path"].endswith("scripts/end_device")
