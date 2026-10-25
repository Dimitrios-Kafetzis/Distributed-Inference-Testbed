from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common import metadata as md
from src.common import probe as pr
from src.common.nodes import END_DEVICE_ROLE, Node
from src.common.runner import CommandResult, FakeRunner, contains, fail, ok, prefix, substring
from src.common.sysfs import SysFs
from src.config import load_config
from src.network.ssh_executor import Executor
from src.network.transport import FakeTransport, ScriptedOutcome

CHRONYC = """Reference ID    : C0A80001 (edge-server)
Stratum         : 2
Ref time (UTC)  : Sat Oct 25 10:00:00 2026
System time     : 0.000312 seconds fast of NTP time
Last offset     : -0.000104 seconds
Leap status     : Normal
"""


def make_tree(sysfs: SysFs) -> None:
    sysfs.write(
        "/etc/os-release", 'NAME="Raspbian"\nPRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\n'
    )
    sysfs.write("/proc/cpuinfo", "processor\t: 0\nmodel name\t: Cortex-A72\n")
    sysfs.write("/proc/meminfo", "MemTotal:        3884848 kB\nMemFree:  100 kB\n")
    sysfs.write("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", 1800000)


def test_probe_reads_facts_from_sysfs_and_commands(sysfs: SysFs) -> None:
    make_tree(sysfs)
    runner = FakeRunner().on(prefix("chronyc"), ok(CHRONYC))
    runner.on(substring("import tensorflow"), ok("2.15.1\n"))
    runner.on(substring("import torch"), fail(1, "ModuleNotFoundError"))
    facts = pr.probe(sysfs, runner)
    assert facts["os_release"] == "Debian GNU/Linux 12 (bookworm)"
    assert facts["cpu_model"] == "Cortex-A72" and facts["max_frequency_khz"] == 1800000
    assert facts["total_memory_mb"] == pytest.approx(3793.8)
    assert facts["tensorflow"] == "2.15.1" and facts["torch"] is None
    assert facts["ntp"]["synchronised"] is True
    assert facts["ntp"]["system_time_ms"] == pytest.approx(0.312)
    assert facts["ntp"]["last_offset_ms"] == pytest.approx(-0.104)
    assert facts["kernel"] and facts["python"]


def test_probe_without_chrony_or_files(sysfs: SysFs) -> None:
    runner = FakeRunner().on(prefix("chronyc"), CommandResult(("chronyc",), 127))
    facts = pr.probe(sysfs, runner, frameworks=False)
    assert facts["ntp"] is None and facts["os_release"] is None and facts["tensorflow"] is None


def test_probe_cli_runs(capsys: pytest.CaptureFixture[str]) -> None:
    pr.main(["--no-frameworks"])
    facts = json.loads(capsys.readouterr().out)
    assert facts["hostname"] and facts["cores"] >= 1


def test_git_state_with_fake_runner() -> None:
    runner = FakeRunner().on(contains("rev-parse"), ok("abc123\n"))
    runner.on(contains("status"), ok(" M src/x.py\n"))
    runner.on(contains("describe"), ok("v1.0.0-3-gabc123\n"))
    state = md.git_state(runner)
    assert state == md.GitState("abc123", True, "v1.0.0-3-gabc123")
    assert md.git_state(FakeRunner().on(contains("rev-parse"), fail(128))).commit is None


def executor_with_probe() -> tuple[Executor, FakeTransport]:
    transport = FakeTransport().on(
        None,
        "src.common.probe",
        ScriptedOutcome(
            0, ['{"hostname": "h", "kernel": "6.6.0", "tensorflow": "2.15.1", "torch": null}']
        ),
    )
    return Executor(transport), transport


def test_collect_run_metadata_records_everything(tmp_path: Path, sysfs: SysFs) -> None:
    executor, transport = executor_with_probe()
    local = (
        FakeRunner(strict=False)
        .on(contains("rev-parse"), ok("deadbeef\n"))
        .on(contains("status"), ok(""))
    )
    local.on(contains("describe"), ok("deadbeef\n")).on(prefix("chronyc"), fail(127))
    config = load_config(environ={"TESTBED_ED_ROOT": "/opt/tb"})
    nodes = [Node(id="d1", role=END_DEVICE_ROLE, ip="192.0.2.11", lat=0, lon=0)]
    meta = md.collect_run_metadata(
        run_id="r1",
        scenario={"name": "s"},
        config=config,
        executor=executor,
        nodes=nodes,
        seeds={"scenario": 7},
        local_runner=local,
        frameworks=False,
    )
    path = meta.save(tmp_path / "metadata.json")
    data = json.loads(path.read_text())
    assert data["testbed"]["git_commit"] == "deadbeef" and data["testbed"]["git_dirty"] is False
    assert (
        data["nodes"]["d1"]["kernel"] == "6.6.0" and data["nodes"]["d1"]["tensorflow"] == "2.15.1"
    )
    assert data["seeds"] == {"scenario": 7} and data["scenario"] == {"name": "s"}
    assert data["testbed"]["config"]["ed_root"] == "/opt/tb"
    assert (
        transport.calls[0][1]
        == "cd /opt/tb/code && /opt/tb/venv/bin/python -m src.common.probe --no-frameworks"
    )


def test_dirty_tree_is_refused_unless_overridden() -> None:
    executor, _ = executor_with_probe()
    dirty = (
        FakeRunner(strict=False)
        .on(contains("rev-parse"), ok("abc\n"))
        .on(contains("status"), ok(" M x\n"))
    )
    config = load_config(environ={})
    nodes = [Node(id="d1", role=END_DEVICE_ROLE, ip="192.0.2.11", lat=0, lon=0)]
    with pytest.raises(md.DirtyTreeError):
        md.collect_run_metadata(
            run_id="r",
            scenario={},
            config=config,
            executor=executor,
            nodes=nodes,
            local_runner=dirty,
            frameworks=False,
        )
    meta = md.collect_run_metadata(
        run_id="r",
        scenario={},
        config=config,
        executor=executor,
        nodes=nodes,
        local_runner=dirty,
        frameworks=False,
        allow_dirty=True,
    )
    assert meta.testbed["dirty_override"] is True


def test_probe_output_must_be_json() -> None:
    transport = FakeTransport().on(None, "probe", ScriptedOutcome(0, ["garbage"]))
    config = load_config(environ={})
    with pytest.raises(RuntimeError, match="not JSON"):
        md.probe_nodes(
            Executor(transport),
            [Node(id="d1", role=END_DEVICE_ROLE, ip="192.0.2.11", lat=0, lon=0)],
            config,
        )
