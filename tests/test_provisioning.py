"""Provisioning: derived networks, rendered templates, mirror manifest, dry-run plans.

The scripts are run in dry-run mode with fake `ssh`, `ssh-keyscan` and `hostname`
binaries on PATH; the printed command plans are compared with golden files under
``tests/golden/provision/``. Set ``UPDATE_GOLDEN=1`` to rewrite the golden files
after an intended change, then review the diff.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from src.common.nodes import Node, Topology, load_topology
from src.config import load_config
from src.provisioning import mirror, networks, render

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "provision"
SCRIPTS = REPO / "scripts" / "provision"
PASSPHRASE = "synthetic-passphrase"
PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE edge"
)


def compare_golden(name: str, content: str) -> None:
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN") == "1" or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if os.environ.get("UPDATE_GOLDEN") != "1":
            pytest.fail(f"golden file {path.relative_to(REPO)} created; review and re-run")
    assert content == path.read_text(encoding="utf-8"), f"differs from {path.relative_to(REPO)}"


@pytest.fixture
def example(tmp_path: Path) -> tuple[object, object, dict[str, str]]:
    key = tmp_path / "id_ed25519.pub"
    key.write_text(PUBLIC_KEY + "\n")
    environ = {"TESTBED_AP_PASSPHRASE": PASSPHRASE, "USER": "edge"}
    config = load_config(
        REPO / "config.example.yaml",
        environ={"TESTBED_SSH_PUBLIC_KEY_PATH": str(key), "TESTBED_BS_ROOT": "/opt/testbed"},
    )
    topology = load_topology(REPO / "nodes.example.json")
    return config, topology, environ


# --- networks ----------


def test_network_derivation() -> None:
    assert networks.network_cidr("192.168.0.") == "192.168.0.0/24"
    assert networks.network_cidr("10.0.") == "10.0.0.0/16"
    assert networks.prefix_bits("10.") == 8
    assert networks.dhcp_range("192.168.0.") == ("192.168.0.100", "192.168.0.199")
    assert networks.dhcp_range("10.0.") == ("10.0.100.100", "10.0.100.199")
    assert networks.in_prefix("192.168.0.11", "192.168.0.")
    assert not networks.in_prefix("192.168.10.11", "192.168.0.")
    with pytest.raises(ValueError):
        networks.network_cidr("300.1.")
    assert networks.vht_centre_index(36) == 42 and networks.vht_centre_index(149) == 155
    assert networks.ht40_direction(36) == "[HT40+]" and networks.ht40_direction(40) == "[HT40-]"
    with pytest.raises(ValueError):
        networks.vht_centre_index(1)


# --- templates ----------


def test_edge_server_templates_match_golden(example: tuple) -> None:
    config, topology, environ = example
    rendered = render.render_edge_server(config, topology, environ)
    assert rendered.warnings == []
    assert set(rendered.files) == set(render.EDGE_SERVER_FILES.values())
    for rel, content in rendered.files.items():
        compare_golden(f"edge_server/{rel}", content)
    hostapd = rendered.files["etc/hostapd/hostapd.conf"]
    assert "hw_mode=a" in hostapd and "ieee80211ac=1" in hostapd and "channel=36" in hostapd
    assert f"wpa_passphrase={PASSPHRASE}" in hostapd
    dnsmasq = rendered.files["etc/dnsmasq.d/testbed.conf"]
    assert "no-resolv" in dnsmasq and "server=" not in dnsmasq  # no upstream resolver
    assert "dhcp-host=device5,10.0.0.15" in dnsmasq
    chrony = rendered.files["etc/chrony/conf.d/testbed.conf"]
    assert "allow 192.168.0.0/24" in chrony and "allow 10.0.0.0/24" in chrony
    assert (
        "--bind 10.0.0.1 --directory /opt/testbed/mirror"
        in (rendered.files["etc/systemd/system/testbed-mirror.service"])
    )
    assert (
        "Neural Network Architecture Partitioning (NNPM) manager"
        in (rendered.files["etc/systemd/system/nnpm.service"])
    )


def test_device_templates_match_golden(example: tuple) -> None:
    config, topology, environ = example
    rendered = render.render_device(config, topology, "device1", environ)
    assert rendered.warnings == []
    assert set(rendered.files) == set(render.DEVICE_FILES.values())
    for rel, content in rendered.files.items():
        compare_golden(f"device1/{rel}", content)
    toml = rendered.files["boot/firmware/custom.toml"]
    assert 'hostname = "device1"' in toml and PUBLIC_KEY in toml and 'ssid = "testbed"' in toml
    sudoers = rendered.files["etc/sudoers.d/testbed"]
    commands = [line for line in sudoers.splitlines() if "NOPASSWD" in line]
    assert len(commands) == 2
    assert commands[0].endswith("/usr/sbin/tc")
    assert commands[1].endswith("/home/pi/testbed/venv/bin/python -m src.testbed.heterogeneity *")
    wifi = rendered.files["etc/NetworkManager/system-connections/testbed-wlan.nmconnection"]
    assert "address1=192.168.0.11/24" in wifi and "powersave=2" in wifi
    eth = rendered.files["etc/NetworkManager/system-connections/testbed-mgmt.nmconnection"]
    assert "address1=10.0.0.11/24" in eth and "dns=10.0.0.1;" in eth
    assert "server 10.0.0.1 iburst" in rendered.files["etc/chrony/conf.d/testbed.conf"]
    pip = rendered.files["etc/pip.conf"]
    assert "no-index = true" in pip and "find-links = http://10.0.0.1:8080/wheels/aarch64/" in pip
    assert (
        "deb [trusted=yes] http://10.0.0.1:8080/apt ./"
        in (rendered.files["etc/apt/sources.list.d/testbed-mirror.list"])
    )


def test_single_network_drops_the_wired_profile_and_warns_on_missing_secrets(
    tmp_path: Path,
) -> None:
    config = load_config(
        REPO / "config.example.yaml",
        environ={
            "TESTBED_MANAGEMENT_IP_PREFIX": "null",
            "TESTBED_SSH_PUBLIC_KEY_PATH": str(tmp_path / "missing.pub"),
        },
    )
    # A single-network nodes.json has no management addresses.
    topology = Topology(
        [
            Node(n.id, n.role, n.ip, n.lat, n.lon)
            for n in load_topology(REPO / "nodes.example.json").nodes
        ]
    )
    rendered = render.render_device(config, topology, "device2", environ={})
    assert render.DEVICE_FILES["nm_eth.nmconnection"] not in rendered.files
    assert any("passphrase" in w for w in rendered.warnings)
    assert any("public key" in w for w in rendered.warnings)
    toml = rendered.files["boot/firmware/custom.toml"]
    assert render.PASSPHRASE_PLACEHOLDER in toml and render.PUBLIC_KEY_PLACEHOLDER in toml
    # Without a separate management network the mirror sits on the data-plane address.
    assert render.mirror_url(config, topology) == "http://192.168.0.1:8080"


def test_render_errors(example: tuple) -> None:
    config, topology, environ = example
    with pytest.raises(render.RenderError, match="unknown node"):
        render.render_device(config, topology, "device9", environ)
    with pytest.raises(render.RenderError, match="edge server"):
        render.render_device(config, topology, "edge-server", environ)
    with pytest.raises(render.RenderError, match="placeholder"):
        render.render_template("hostapd.conf", {"AP_INTERFACE": "x"})


def test_render_cli_exports_and_writes(example: tuple, tmp_path: Path) -> None:
    config, _, _ = example
    env = dict(
        os.environ,
        TESTBED_CONFIG=str(REPO / "config.example.yaml"),
        TESTBED_AP_PASSPHRASE=PASSPHRASE,
        TESTBED_SSH_PUBLIC_KEY_PATH=config.ssh_public_key_path,
    )
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.provisioning.render",
            "--nodes",
            str(REPO / "nodes.example.json"),
            "--role",
            "device",
            "--node",
            "device3",
            "--vars",
            "--out",
            str(out),
            "--var",
            "DEVICE_PASSWORD_HASH=$6$salt$hash",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "export DEVICE_DATA_IP=192.168.0.13" in result.stdout
    assert "export AP_PASSPHRASE=synthetic-passphrase" in result.stdout
    assert 'password = "$6$salt$hash"' in (out / "boot/firmware/custom.toml").read_text()


# --- mirror manifest ----------


def test_mirror_manifest_and_verify(tmp_path: Path) -> None:
    mirror_dir = tmp_path / "mirror"
    (mirror_dir / "wheels" / "aarch64").mkdir(parents=True)
    (mirror_dir / "wheels" / "aarch64" / "a-1.0-py3-none-any.whl").write_bytes(b"wheel")
    (mirror_dir / "apt").mkdir()
    (mirror_dir / "apt" / "Packages.gz").write_bytes(b"gz")
    path = mirror.write_manifest(mirror_dir, REPO)
    manifest = json.loads(path.read_text())
    assert manifest["file_count"] == 2 and "requirements/edge_device.lock" in manifest["lock_files"]
    assert mirror.verify(mirror_dir) == []
    (mirror_dir / "apt" / "Packages.gz").write_bytes(b"changed")
    (mirror_dir / "apt" / "extra.deb").write_bytes(b"x")
    (mirror_dir / "wheels" / "aarch64" / "a-1.0-py3-none-any.whl").unlink()
    problems = mirror.verify(mirror_dir)
    assert problems == [
        "missing: wheels/aarch64/a-1.0-py3-none-any.whl",
        "unexpected: apt/extra.deb",
        "changed: apt/Packages.gz",
    ]
    assert mirror.verify(tmp_path / "nowhere")[0].endswith("run `manifest` first")


# --- scripts in dry-run mode ----------


def fake_bin(tmp_path: Path, uname: str = "aarch64", hostname: str = "raspberrypi") -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    tools = {
        "ssh": f'echo "ssh $*" >> {log}\ncase "$*" in *"uname -m"*) echo {uname} ;; esac\n',
        "ssh-keyscan": f'echo "ssh-keyscan $*" >> {log}\necho "$2 ssh-ed25519 AAAA"\n',
        "hostname": f"echo {hostname}\n",
    }
    for name, body in tools.items():
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\n" + body)
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def run_script(
    tmp_path: Path,
    script: str,
    *args: str,
    env_extra: dict[str, str] | None = None,
    uname: str = "aarch64",
) -> subprocess.CompletedProcess[str]:
    env = dict(
        os.environ,
        PATH=f"{fake_bin(tmp_path, uname=uname)}{os.pathsep}{os.environ['PATH']}",
        HOME=str(tmp_path / "home"),
        TESTBED_CONFIG=str(REPO / "config.example.yaml"),
        TESTBED_BS_ROOT="/opt/testbed",
        TESTBED_AP_PASSPHRASE=PASSPHRASE,
        TESTBED_SSH_PUBLIC_KEY_PATH=str(tmp_path / "id.pub"),
        TESTBED_SSH_KNOWN_HOSTS_PATH=str(tmp_path / "known_hosts"),
        TESTBED_PYTHON=sys.executable,  # the same interpreter on every machine
        DRY_RUN="1",
    )
    env.pop("TESTBED_DEVICE_PASSWORD", None)
    env.update(env_extra or {})
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "id.pub").write_text(PUBLIC_KEY + "\n")
    # The scripts read nodes.json from the repository root; point them at the example.
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )


def normalise(plan: str, tmp_path: Path) -> str:
    plan = plan.replace(str(tmp_path), "<TMP>").replace(str(REPO), "<REPO>")
    plan = re.sub(r"/tmp/tmp\.[A-Za-z0-9]+", "<RENDERED>", plan)
    return plan.replace(sys.executable, "<PYTHON>")


@pytest.fixture
def example_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scripts load nodes.json from the repository root: use the example file."""
    monkeypatch.setenv("TESTBED_NODES", str(REPO / "nodes.example.json"))


def test_provision_device_plan(tmp_path: Path, example_nodes: None) -> None:
    result = run_script(tmp_path, "provision_device.sh", "device1", "--llm")
    assert result.returncode == 0, result.stderr
    plan = normalise(result.stdout, tmp_path)
    compare_golden("plan_provision_device.txt", plan)
    assert "bash -c 'ssh-keyscan -H 10.0.0.11 >> <TMP>/known_hosts'" in plan
    assert plan.count("pi@10.0.0.11") >= 8
    assert "edge_device,edge_device_llm" in plan
    assert plan.rstrip().splitlines()[-1].endswith(" network'")  # addresses change last
    assert "https://" not in plan  # nothing upstream: everything through the mirror
    assert "ssh -o BatchMode=yes" in (tmp_path / "calls.log").read_text()  # uname was queried


def test_provision_device_refuses_a_32_bit_kernel(tmp_path: Path, example_nodes: None) -> None:
    result = run_script(tmp_path, "provision_device.sh", "device2", uname="armv7l")
    assert result.returncode == 1
    assert "armv7l" in result.stderr and "64-bit" in result.stderr
    assert "device_setup.sh" not in result.stdout  # nothing was planned after the check


def test_provision_device_unknown_node(tmp_path: Path, example_nodes: None) -> None:
    result = run_script(tmp_path, "provision_device.sh", "device9")
    assert result.returncode == 1 and "could not read" in result.stderr


def test_provision_edge_server_plan(tmp_path: Path, example_nodes: None) -> None:
    result = run_script(tmp_path, "provision_edge_server.sh", "--allow-upstream")
    assert result.returncode == 0, result.stderr
    plan = normalise(result.stdout, tmp_path)
    compare_golden("plan_provision_edge_server.txt", plan)
    assert "systemctl enable --now testbed-ap-address hostapd dnsmasq chrony" in plan
    assert "/etc/hostapd/hostapd.conf" in plan and "600 " in plan
    assert "ssh-keygen -q -t ed25519" in plan
    mirror_plan = run_script(tmp_path, "provision_edge_server.sh", "--skip-apt").stdout
    assert "--no-index --find-links /opt/testbed/mirror/wheels/x86_64" in mirror_plan
    assert "apt-get" not in mirror_plan


def test_build_mirror_plan(tmp_path: Path, example_nodes: None) -> None:
    result = run_script(
        tmp_path,
        "build_mirror.sh",
        "all",
        "device1",
        env_extra={"TESTBED_HEADSPLIT_COMMIT": "abc1234"},
    )
    assert result.returncode == 0, result.stderr
    plan = normalise(result.stdout, tmp_path)
    compare_golden("plan_build_mirror.txt", plan)
    assert plan.count("pip download") == 5  # four lock files and HeadSplit
    assert "--platform manylinux_2_28_aarch64" in plan and "--require-hashes" in plan
    assert "apt-get install --download-only --reinstall" in plan
    assert "git clone --quiet --bare https://github.com/Dimitrios-Kafetzis/headsplit.git" in plan
    assert "src.provisioning.mirror manifest" in plan
    skipped = run_script(tmp_path, "build_mirror.sh", "headsplit")
    assert skipped.returncode == 0 and "not mirrored" in skipped.stderr


def test_prepare_image_plan(tmp_path: Path, example_nodes: None) -> None:
    boot = tmp_path / "boot"
    boot.mkdir()
    (boot / "config.txt").write_text("")
    (boot / "kernel8.img").write_bytes(b"")
    result = run_script(tmp_path, "prepare_image.sh", "device4", str(boot))
    assert result.returncode == 0, result.stderr
    plan = normalise(result.stdout, tmp_path)
    compare_golden("plan_prepare_image.txt", plan)
    assert "custom.toml <TMP>/boot/custom.toml" in plan and "touch <TMP>/boot/ssh" in plan
    assert "password login locked" in result.stderr
    (boot / "kernel7l.img").write_bytes(b"")
    refused = run_script(tmp_path, "prepare_image.sh", "device4", str(boot))
    assert refused.returncode == 1 and "32-bit image" in refused.stderr


def test_device_setup_plan(tmp_path: Path, example: tuple) -> None:
    """The device-side script, run locally against a fake file-system root."""
    config, topology, environ = example
    rendered = tmp_path / "rendered"
    render.write_rendered(render.render_device(config, topology, "device1", environ), rendered)
    root = tmp_path / "root"
    (root / "etc" / "apt" / "sources.list.d").mkdir(parents=True)
    (root / "etc" / "apt" / "sources.list").write_text(
        "deb http://deb.debian.org/debian bookworm main\n"
    )
    (root / "etc" / "apt" / "sources.list.d" / "raspi.list").write_text(
        "deb http://archive.raspberrypi.com/debian/ bookworm main\n"
    )
    (root / "etc" / "chrony").mkdir()
    (root / "etc" / "chrony" / "chrony.conf").write_text("pool 2.debian.pool.ntp.org iburst\n")
    args = [
        "--rendered",
        str(rendered),
        "--node",
        "device1",
        "--code",
        str(REPO),
        "--venv",
        "/home/pi/testbed/venv",
        "--locks",
        "edge_device",
        "--mirror",
        "http://10.0.0.1:8080",
        "--headsplit-dir",
        "/home/pi/testbed/headsplit",
        "--headsplit-commit",
        "abc1234",
        "hostname",
        "apt",
        "chrony",
        "sudoers",
        "pip",
        "venv",
        "headsplit",
        "verify",
        "network",
    ]
    result = run_script(
        tmp_path, "device_setup.sh", *args, env_extra={"TESTBED_FS_ROOT": str(root)}
    )
    assert result.returncode == 0, result.stderr
    plan = normalise(result.stdout, tmp_path)
    compare_golden("plan_device_setup.txt", plan)
    assert "hostnamectl set-hostname device1" in plan  # the fake hostname differs
    assert "sources.list.d/raspi.list <TMP>/root/etc/apt/sources.upstream.disabled/" in plan
    assert "testbed-mirror.list" in plan and "visudo -cf" in plan
    assert "--no-index --find-links http://10.0.0.1:8080/wheels/aarch64/ --require-hashes" in plan
    assert "--no-install-recommends python3 python3-venv" in plan
    assert "git clone --quiet http://10.0.0.1:8080/src/headsplit.git" in plan
    assert plan.rstrip().splitlines()[-1].endswith("testbed-mgmt")
    bad = run_script(tmp_path, "device_setup.sh", *args[:-9], "reboot")
    assert bad.returncode == 1 and "unknown step" in bad.stderr


def test_no_upstream_url_outside_the_mirror_builder() -> None:
    """H-02: every provisioning script installs from the mirror; only build_mirror.sh
    (and the scripts' comments) may name an upstream source."""
    offences = []
    for path in sorted(SCRIPTS.rglob("*")):
        if not path.is_file() or path.name == "build_mirror.sh":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"https?://[A-Za-z0-9]", code):  # a literal host, not a variable
                offences.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert offences == []


def test_scripts_are_executable_and_have_usage() -> None:
    for name in [
        "prepare_image.sh",
        "build_mirror.sh",
        "provision_edge_server.sh",
        "provision_device.sh",
        "device_setup.sh",
    ]:
        path = SCRIPTS / name
        assert path.stat().st_mode & stat.S_IXUSR, name
        assert "Usage:" in path.read_text(encoding="utf-8"), name
