from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from src.common.clock import FakeClock
from src.common.runner import FakeRunner, ok, prefix
from src.testbed import latency_injector as li
from src.testbed.latency_injector import NetemParams, TcPlan, VariationRunner

REPO = Path(__file__).resolve().parents[1]


def commands(plan_commands: list[list[str]]) -> list[str]:
    return [shlex.join(c) for c in plan_commands]


def test_delay_only_uses_the_documented_command() -> None:
    plan = TcPlan("wlan0", FakeRunner())
    assert commands(plan.commands_all(NetemParams(delay_ms=10))) == [
        "sudo tc qdisc add dev wlan0 root netem delay 10ms"
    ]
    assert commands(
        TcPlan("wlan0", FakeRunner(), sudo=False).commands_all(NetemParams(delay_ms=7.5))
    ) == ["tc qdisc add dev wlan0 root netem delay 7.5ms"]


def test_jitter_loss_and_rate_commands() -> None:
    plan = TcPlan("wlan0", FakeRunner())
    assert commands(plan.commands_all(NetemParams(delay_ms=20, jitter_ms=5, loss_percent=0.1))) == [
        "sudo tc qdisc add dev wlan0 root netem delay 20ms 5ms distribution uniform loss 0.1%"
    ]
    assert commands(plan.commands_all(NetemParams(delay_ms=10, rate_mbps=25))) == [
        "sudo tc qdisc add dev wlan0 root handle 1: netem delay 10ms",
        "sudo tc qdisc add dev wlan0 parent 1: handle 10: tbf rate 25mbit burst 32kbit "
        "latency 400ms",
    ]
    assert commands(plan.commands_all(NetemParams(loss_percent=1))) == [
        "sudo tc qdisc add dev wlan0 root netem loss 1%"
    ]
    assert commands(plan.commands_clear()) == ["sudo tc qdisc del dev wlan0 root"]


def test_params_validation() -> None:
    with pytest.raises(ValueError, match="negative"):
        NetemParams(delay_ms=-1).validate()
    with pytest.raises(ValueError, match="exceed 100"):
        NetemParams(loss_percent=150).validate()
    with pytest.raises(ValueError, match="jitter needs a delay"):
        NetemParams(jitter_ms=1).validate()
    assert NetemParams().is_empty


def test_per_destination_plan_is_an_htb_tree_with_filters() -> None:
    plan = TcPlan("wlan0", FakeRunner())
    shaping = {
        "192.0.2.12": NetemParams(delay_ms=25, rate_mbps=10),
        "192.0.2.11": NetemParams(delay_ms=15),
    }
    got = commands(plan.commands_per_destination(shaping))
    assert got == [
        "sudo tc qdisc add dev wlan0 root handle 1: htb default 999",
        "sudo tc class add dev wlan0 parent 1: classid 1:999 htb rate 10000mbit",
        "sudo tc class add dev wlan0 parent 1: classid 1:1 htb rate 10000mbit ceil 10000mbit",
        "sudo tc qdisc add dev wlan0 parent 1:1 handle 2: netem delay 15ms",
        "sudo tc filter add dev wlan0 protocol ip parent 1: prio 1 u32 match ip dst "
        "192.0.2.11/32 flowid 1:1",
        "sudo tc class add dev wlan0 parent 1: classid 1:2 htb rate 10mbit ceil 10mbit",
        "sudo tc qdisc add dev wlan0 parent 1:2 handle 3: netem delay 25ms",
        "sudo tc filter add dev wlan0 protocol ip parent 1: prio 1 u32 match ip dst "
        "192.0.2.12/32 flowid 1:2",
    ]


def test_apply_is_idempotent_and_clears_existing_qdisc() -> None:
    runner = FakeRunner(strict=False)
    runner.on(
        prefix("sudo", "tc", "qdisc", "show"),
        ok("qdisc netem 8001: root refcnt 2 limit 1000 delay 10ms\n"),
    )
    plan = TcPlan("wlan0", runner)
    plan.apply_all(NetemParams(delay_ms=30))
    assert runner.commands() == [
        "sudo tc qdisc show dev wlan0",
        "sudo tc qdisc del dev wlan0 root",
        "sudo tc qdisc add dev wlan0 root netem delay 30ms",
    ]
    fresh = FakeRunner(strict=False).on(
        prefix("sudo", "tc", "qdisc", "show"), ok("qdisc pfifo_fast 0: root refcnt 2\n")
    )
    TcPlan("wlan0", fresh).apply_all(NetemParams(delay_ms=30))
    assert "sudo tc qdisc del dev wlan0 root" not in fresh.commands()


def test_mode_aliases_and_draws() -> None:
    assert li.normalise_mode("once") == "constant"
    assert li.normalise_mode("continuous") == "uniform"
    assert li.normalise_mode("periodic") == "square"
    with pytest.raises(ValueError):
        li.normalise_mode("random")
    import random

    assert 5 <= li.draw_delay((5, 20), random.Random(0)) <= 20
    assert li.draw_delay((7, 7), random.Random(0)) == 7
    with pytest.raises(ValueError):
        li.draw_delay((9, 1), random.Random(0))


def applied_delays(runner: FakeRunner) -> list[str]:
    return [c for c in runner.commands() if "netem delay" in c]


def test_constant_profile_applies_once_and_clears_after_duration() -> None:
    runner = FakeRunner(strict=False)
    clock = FakeClock(start_time=0.0)
    v = VariationRunner(
        TcPlan("wlan0", runner), NetemParams(), (10, 10), "constant", 0, 30, clock=clock
    )
    assert v.run() == [10.0, None]
    assert applied_delays(runner) == ["sudo tc qdisc add dev wlan0 root netem delay 10ms"]
    assert runner.commands()[-1] == "sudo tc qdisc del dev wlan0 root"
    assert clock.time() >= 30.0


def test_constant_profile_without_duration_leaves_shaping_in_place() -> None:
    runner = FakeRunner(strict=False)
    v = VariationRunner(
        TcPlan("wlan0", runner), NetemParams(loss_percent=0.5), (5, 5), "once", 0, None
    )
    assert v.run() == [5.0]
    assert runner.commands()[-1] == "sudo tc qdisc add dev wlan0 root netem delay 5ms loss 0.5%"


def test_uniform_profile_redraws_every_period() -> None:
    runner = FakeRunner(strict=False)
    clock = FakeClock(start_time=0.0)
    v = VariationRunner(
        TcPlan("wlan0", runner), NetemParams(), (5, 20), "uniform", 10, 35, clock=clock, seed=1
    )
    applied = v.run()
    assert len(applied) == 5 and applied[-1] is None  # 4 draws + final clear
    assert all(5 <= d <= 20 for d in applied[:-1] if d is not None)
    assert len(set(applied[:-1])) > 1


def test_square_profile_alternates_on_and_off() -> None:
    runner = FakeRunner(strict=False)
    clock = FakeClock(start_time=0.0)
    v = VariationRunner(
        TcPlan("wlan0", runner), NetemParams(), (8, 8), "periodic", 5, 20, clock=clock
    )
    assert v.run() == [8.0, None, 8.0, None]
    with pytest.raises(ValueError, match="positive period"):
        VariationRunner(TcPlan("wlan0", runner), NetemParams(), (8, 8), "square", 0, 20)


def test_per_destination_variation_uses_htb_tree() -> None:
    runner = FakeRunner(strict=False)
    v = VariationRunner(
        TcPlan("wlan0", runner),
        NetemParams(),
        (12, 12),
        "constant",
        0,
        None,
        destinations=["192.0.2.11"],
    )
    v.run()
    assert any("u32 match ip dst 192.0.2.11/32" in c for c in runner.commands())


def test_cli_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    li.main(["wlan0", "--delay", "10", "--jitter", "2", "--dry-run"])
    out = capsys.readouterr().out
    assert (
        "[dry-run] sudo tc qdisc add dev wlan0 root netem delay 10ms 2ms distribution uniform"
        in out
    )
    li.main(["wlan0", "--clear", "--dry-run"])
    assert "[dry-run] sudo tc qdisc del dev wlan0 root" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="nothing to apply"):
        li.main(["wlan0", "--dry-run"])


def test_apply_latency_script_keeps_the_historical_interface() -> None:
    script = REPO / "scripts" / "end_device" / "apply_latency.sh"
    env = dict(os.environ, TESTBED_PYTHON=os.environ.get("TESTBED_PYTHON", "python3"))
    env["PYTHONPATH"] = str(REPO)
    once = subprocess.run(
        ["bash", str(script), "wlan0", "10", "20", "--once", "--dry-run", "--seed", "3"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    assert once.returncode == 0, once.stderr
    assert re.search(
        r"\[dry-run\] sudo tc qdisc add dev wlan0 root netem delay 1\d(\.\d+)?ms", once.stdout
    )
    clear = subprocess.run(
        ["bash", str(script), "wlan0", "--clear", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    assert "[dry-run] sudo tc qdisc del dev wlan0 root" in clear.stdout
    missing = subprocess.run(
        ["bash", str(script), "wlan0", "10", "20", "--continuous"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    assert missing.returncode == 1 and "need --duration" in missing.stderr
    usage = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, cwd=REPO)
    assert usage.returncode == 1 and "Usage" in usage.stdout
