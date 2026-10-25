from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from src.common.nodes import END_DEVICE_ROLE, Node
from src.config import load_config
from src.network import collection
from src.network.ssh_executor import Executor
from src.network.transport import FakeTransport, LocalTransport

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "end_device" / "transfer_files.sh"


def nodes(*ids: str) -> list[Node]:
    return [
        Node(id=i, role=END_DEVICE_ROLE, ip=f"192.0.2.{10 + k}", lat=0, lon=0)
        for k, i in enumerate(ids)
    ]


def fake_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    if bin_dir.exists():
        return bin_dir
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    for name in ("scp", "sshpass"):
        script = bin_dir / name
        script.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\n')
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    host = bin_dir / "hostname"
    host.write_text("#!/bin/sh\necho device7\n")
    host.chmod(host.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def run_script(
    tmp_path: Path, *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PATH=f"{fake_bin(tmp_path)}{os.pathsep}{os.environ['PATH']}")
    env.pop("SSHPASS", None)
    env.update(env_extra or {})
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env)


def test_transfer_files_prefixes_hostname_and_uses_key_by_default(tmp_path: Path) -> None:
    result_file = tmp_path / "benchmark_results.json"
    result_file.write_text("{}")
    done = run_script(tmp_path, "user", "192.0.2.1", "/results/run1", str(result_file))
    assert done.returncode == 0, done.stderr
    calls = (tmp_path / "calls.log").read_text()
    assert "scp -q -P 22 -o BatchMode=yes" in calls
    assert "/device7_benchmark_results.json user@192.0.2.1:/results/run1/" in calls
    assert "transferred" in done.stdout
    assert result_file.exists()  # the original is not renamed


def test_transfer_files_password_mode_and_missing_file(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}")
    done = run_script(
        tmp_path,
        "user",
        "192.0.2.1",
        "/r",
        str(tmp_path / "a.json"),
        str(tmp_path / "missing.json"),
        env_extra={"SSHPASS": "secret"},
    )
    assert done.returncode == 1
    calls = (tmp_path / "calls.log").read_text()
    assert calls.startswith("sshpass -e scp")
    assert "secret" not in calls
    assert "missing" in done.stderr
    usage = run_script(tmp_path, "user")
    assert usage.returncode == 1 and "Usage" in usage.stderr


def test_pull_collection_with_fake_transport(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.files[("d1", "/home/x/results/r.json")] = '{"d": 1}'
    transport.files[("d2", "/home/x/results/r.json")] = '{"d": 2}'
    executor = Executor(transport)
    collected = collection.collect(
        executor,
        nodes("d1", "d2"),
        ["/home/x/results/r.json"],
        tmp_path / "raw",
        hostnames={"d1": "host1"},
    )
    assert [c.local_path.name for c in collected] == ["host1_r.json", "d2_r.json"]
    assert (tmp_path / "raw" / "host1_r.json").read_text() == '{"d": 1}'


def test_push_collection_runs_transfer_files_on_each_node(tmp_path: Path) -> None:
    config = load_config(environ={"TESTBED_ED_ROOT": "/opt/tb", "TESTBED_SSH_USERNAME": "bs"})
    transport = FakeTransport()
    executor = Executor(transport)
    collected = collection.collect(
        executor,
        nodes("d1"),
        ["/opt/tb/results/r.json"],
        tmp_path / "raw",
        config=config,
        edge_server_ip="192.0.2.1",
        mode="push",
    )
    assert transport.calls == [
        (
            "d1",
            "/opt/tb/code/scripts/end_device/transfer_files.sh bs 192.0.2.1 "
            f"{tmp_path / 'raw'} /opt/tb/results/r.json",
        )
    ]
    assert collected[0].local_path.name == "d1_r.json"


def test_local_transport_forces_pull(tmp_path: Path) -> None:
    workdir = tmp_path / "d1"
    workdir.mkdir()
    (workdir / "r.json").write_text("{}")
    n = Node(id="d1", role=END_DEVICE_ROLE, ip="192.0.2.11", lat=0, lon=0, workdir=str(workdir))
    executor = Executor(LocalTransport({"d1": workdir}))
    collected = collection.collect(executor, [n], ["r.json"], tmp_path / "raw", mode="push")
    assert collected[0].local_path.read_text() == "{}"
