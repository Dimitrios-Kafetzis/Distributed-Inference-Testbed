"""NNPM web application: routes, probes, dialogs' endpoints, runs in local mode."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

from gui import app as gui_app
from gui.state import AppState, Prober, parse_ping_summary, status_from_replies
from src.common.nodes import Node, Topology
from src.common.runner import CommandResult, FakeRunner, prefix
from src.orchestration import local_mode

REPO = Path(__file__).resolve().parents[1]

PING_OK = (
    "PING x\n64 bytes from x: icmp_seq=1 time=1 ms\n\n--- x ping statistics ---\n"
    "3 packets transmitted, 3 received, 0% packet loss\n"
)
PING_LOSSY = (
    "PING x\n\n--- x ping statistics ---\n3 packets transmitted, 2 received, 33% packet loss\n"
)


@pytest.fixture
def state(tmp_path: Path) -> Iterator[AppState]:
    runner = FakeRunner(strict=False).on(prefix("ping"), CommandResult(("ping",), 0, PING_OK))
    config, _ = local_mode.create_local_testbed(tmp_path / "tb", devices=2)
    st = AppState(
        config,
        tmp_path / "tb" / "nodes.json",
        local=True,
        results_dir=tmp_path / "tb" / "results",
        scenario_root=tmp_path / "tb" / "scenarios",
        probe_runner=runner,
    )
    yield st
    st.prober.stop()
    st.runs.wait(timeout=120)


@pytest.fixture
def app(state: AppState) -> Flask:
    application = gui_app.create_app(state, start_prober=False)
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def test_probe_rule_green_only_when_all_three_pings_succeed() -> None:
    assert status_from_replies(3, 3) == "green"
    assert status_from_replies(3, 2) == "red"
    assert status_from_replies(3, 0) == "red"
    assert parse_ping_summary(PING_OK) == (3, 3)
    assert parse_ping_summary(PING_LOSSY) == (3, 2)
    assert parse_ping_summary("garbage") == (3, 0)


def test_index_page_has_the_thesis_title_and_vendored_leaflet(client: FlaskClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "<title>Neural Network Architecture Partitioning (NNPM) manager</title>" in html
    assert "vendor/leaflet-1.9.4/leaflet.js" in html and "unpkg" not in html
    assert "Create scenario" in html and "Benchmarking" in html and "Optimise" in html
    assert client.get("/static/vendor/leaflet-1.9.4/leaflet.js").status_code == 200
    assert client.get("/healthz").get_json()["nodes"] == 3


def test_map_rule_functions_live_in_gui_app() -> None:
    assert gui_app.calculate_latency(0.0) == 5.0
    assert gui_app.calculate_latency(20_000.0) == 25.0
    assert gui_app.calculate_distance(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)


def test_nodes_crud_and_links(client: FlaskClient, state: AppState) -> None:
    nodes = client.get("/api/nodes").get_json()["nodes"]
    assert {n["id"] for n in nodes} == {"edge-server", "device1", "device2"}
    created = client.post(
        "/api/nodes", json={"id": "device3", "ip": "192.0.2.13", "lat": 37.99, "lon": 23.74}
    )
    assert created.status_code == 201
    assert (
        client.post(
            "/api/nodes", json={"id": "device3", "ip": "192.0.2.13", "lat": 0, "lon": 0}
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/nodes", json={"id": "x", "ip": "not-an-ip", "lat": 0, "lon": 0}
        ).status_code
        == 400
    )
    moved = client.put("/api/nodes/device3", json={"lat": 38.0, "lon": 23.75})
    assert moved.get_json()["node"]["lat"] == 38.0
    saved = json.loads(state.nodes_path.read_text())
    assert any(n["id"] == "device3" and n["lat"] == 38.0 for n in saved["nodes"])
    links = client.get("/api/links").get_json()["links"]
    assert len(links) == 6  # four nodes, every pair
    assert all(5.0 <= link["latency_ms"] <= 50.0 for link in links)
    derived = client.post("/api/latency-from-map").get_json()["latency_ms"]
    assert set(derived) == {"device1", "device2", "device3"} and "edge_server" in derived["device1"]
    # Moving a marker changes the value the next launch would inject (C-07).
    before = derived["device3"]["edge_server"]
    client.put("/api/nodes/device3", json={"lat": 38.3, "lon": 24.1})
    after = client.post("/api/latency-from-map").get_json()["latency_ms"]["device3"]["edge_server"]
    assert after > before
    info = client.get("/api/nodes/device3").get_json()["node"]
    assert info["ip"] == "192.0.2.13"
    assert client.delete("/api/nodes/device3").status_code == 200
    assert client.get("/api/nodes/device3").status_code == 404
    assert client.delete("/api/nodes/device3").status_code == 404


def test_status_probe_and_event_stream(client: FlaskClient, state: AppState) -> None:
    snapshot = client.post("/api/status/probe").get_json()
    assert snapshot["round"] == 1
    assert {k: v["status"] for k, v in snapshot["nodes"].items()} == {
        "edge-server": "green",
        "device1": "green",
        "device2": "green",
    }
    state.prober.runner = FakeRunner(strict=False).on(
        prefix("ping"), CommandResult(("ping",), 1, PING_LOSSY)
    )
    snapshot = client.post("/api/status/probe").get_json()
    assert all(v["status"] == "red" for v in snapshot["nodes"].values())
    response = client.get("/events/status?limit=1")
    body = response.get_data(as_text=True)
    assert (
        response.mimetype == "text/event-stream"
        and "event: status" in body
        and '"round": 2' in body
    )


def test_scenario_endpoints(client: FlaskClient) -> None:
    bad = client.post(
        "/api/scenarios/validate",
        json={"name": "x", "model": "fake", "load": {"default": {"mode": "no"}}},
    ).get_json()
    assert bad["valid"] is False
    good = client.post("/api/scenarios/validate", json={"name": "x", "model": "fake"}).get_json()
    assert good["valid"] is True and good["scenario"]["repetitions"] == 1
    saved = client.post("/api/scenarios", json={"name": "s1", "model": "fake", "repetitions": 2})
    assert saved.status_code == 201 and saved.get_json()["version"] == 1
    again = client.post(
        "/api/scenarios", json={"name": "s1", "model": "fake", "repetitions": 3}
    ).get_json()
    assert again["version"] == 2
    listing = client.get("/api/scenarios").get_json()["scenarios"]
    assert listing[0]["name"] == "s1" and len(listing[0]["versions"]) == 2
    assert client.get("/api/scenarios/s1/1").get_json()["scenario"]["repetitions"] == 2
    assert client.get("/api/scenarios/s1").get_json()["version"] == 2
    assert client.get("/api/scenarios/nope").status_code == 404
    assert client.post("/api/scenarios", json=["not", "an", "object"]).status_code == 400


def wait_for_run(client: FlaskClient, run_id: str, timeout_s: float = 180.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        detail = client.get(f"/api/runs/{run_id}").get_json()
        if detail.get("status") in ("completed", "aborted"):
            return detail
        time.sleep(0.5)
    raise AssertionError("the run did not finish in time")


def test_run_benchmarking_and_optimise_through_the_api(
    client: FlaskClient, state: AppState
) -> None:
    scenario = {
        "name": "api_bench",
        "model": "fake",
        "repetitions": 1,
        "warmup": 0,
        "settle_s": 0,
        "start_lead_s": 0.2,
        "monitoring": {"enabled": True, "interval_s": 0.1, "latency_pings": 1},
        "optimisation": {"enabled": False},
        "edge_server": {"profile_capacity": True},
    }
    started = client.post(
        "/api/runs", json={"scenario": scenario, "pipeline": "benchmarking", "run_id": "api1"}
    )
    assert started.status_code == 202, started.get_json()
    assert client.post("/api/runs", json={"name": "api_bench"}).status_code == 409  # one at a time
    detail = wait_for_run(client, "api1")
    assert detail["status"] == "completed", detail.get("error")
    assert "total_averages_of_benchmark_results.json" in detail["files"]
    log = client.get("/events/runs/api1?limit=3").get_data(as_text=True)
    assert "event: log" in log and "[nnpm]" in log
    runs = client.get("/api/runs").get_json()
    assert (
        runs["busy"] is False
        and runs["runs"][0]["run_id"] == "api1"
        and runs["runs"][0]["status"] == "completed"
    )
    # The `latest` link in the results directory must not produce a second entry.
    assert [run["run_id"] for run in runs["runs"]] == ["api1"]
    assert client.get("/api/runs/api1/files/summary.json").status_code == 200
    assert client.get("/api/runs/api1/files/../../etc/passwd").status_code == 404
    result = client.post(
        "/api/optimise",
        json={
            "run_id": "api1",
            "algorithm": "greedy",
            "policy": "RR",
            "objective": "total_delay",
            "initial_scheme": "50%",
        },
    )
    assert result.status_code == 200, result.get_json()
    payload = result.get_json()["result"]
    assert set(payload["by_node"]) == {"device1", "device2"} and payload["policy"] == "RR"
    assert (state.results_dir / "api1" / payload["search_log"].split("api1/")[-1]).exists()
    assert client.post("/api/optimise", json={"run_id": "nope"}).status_code == 404
    assert (
        client.post("/api/optimise", json={"run_id": "api1", "algorithm": "magic"}).status_code
        == 400
    )
    # The topology statistics were written by the pipeline and are visible in the node panel.
    info = client.get("/api/nodes/device1").get_json()["node"]
    assert info["statistics"]["run_id"] == "api1"


def test_stop_aborts_a_run_and_tears_down(client: FlaskClient, state: AppState) -> None:
    scenario = {
        "name": "api_stop",
        "model": "fake",
        "repetitions": 20,
        "warmup": 0,
        "settle_s": 3,
        "start_lead_s": 0.2,
        "load": {"default": {"mode": "fixed", "cpu_percent": 5, "memory_percent": 1, "cycle_s": 1}},
        "monitoring": {"enabled": False},
        "optimisation": {"enabled": False},
    }
    started = client.post("/api/runs", json={"scenario": scenario, "run_id": "stop1"})
    assert started.status_code == 202
    time.sleep(0.5)
    assert client.post("/api/runs/stop1/stop").get_json()["stopping"] is True
    detail = wait_for_run(client, "stop1")
    assert detail["status"] == "aborted" and "stopped by the user" in (detail.get("error") or "")
    journal = json.loads((state.results_dir / "stop1" / "journal.json").read_text())
    assert journal["entries"] and all(e["undone"] for e in journal["entries"])
    assert client.post("/api/runs/stop1/stop").status_code == 404


def test_catalog(client: FlaskClient) -> None:
    catalog = client.get("/api/catalog").get_json()
    assert "fake" in catalog["models"] and "greedy" in catalog["algorithms"]
    assert "input_elements" in catalog["model_parameters"]["fake"]["properties"]
    assert catalog["regimes"]["heavy"] == {"cpu": [60.0, 80.0], "memory": [50.0, 60.0]}
    assert any(e["name"] == "baseline" for e in catalog["examples"])


def test_basic_auth_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NNPM_AUTH", "user:secret")
    monkeypatch.setenv("TESTBED_NNPM_BASIC_AUTH_ENV", "NNPM_AUTH")
    config, _ = local_mode.create_local_testbed(tmp_path / "tb", devices=1)
    st = AppState(
        config,
        tmp_path / "tb" / "nodes.json",
        local=True,
        results_dir=tmp_path / "tb" / "results",
        scenario_root=tmp_path / "tb" / "scenarios",
    )
    app = gui_app.create_app(st, start_prober=False)
    client = app.test_client()
    assert client.get("/healthz").status_code == 401
    import base64

    token = base64.b64encode(b"user:secret").decode()
    assert client.get("/healthz", headers={"Authorization": f"Basic {token}"}).status_code == 200


def test_a_round_starts_every_period_even_when_a_node_is_slow() -> None:
    """J-14: a probe that outlasts the period keeps its last colour and finishes later."""
    topology = Topology(
        [
            Node("edge-server", "Base Station", "192.0.2.1", 37.98, 23.73),
            Node("device1", "End Device", "192.0.2.11", 37.98, 23.74),
            Node("device2", "End Device", "192.0.2.12", 37.99, 23.73),
        ]
    )
    slow_ip = "192.0.2.12"

    def reply(argv: tuple[str, ...]) -> CommandResult:
        if argv[-1] == slow_ip:
            time.sleep(1.2)
            return CommandResult(argv, 1, PING_LOSSY)
        return CommandResult(argv, 0, PING_OK)

    runner = FakeRunner(strict=False).on(prefix("ping"), reply)
    prober = Prober(lambda: topology, period_s=0.4, runner=runner)
    started = time.monotonic()
    first = prober.probe_round()
    assert time.monotonic() - started < 1.0  # the round did not wait for the slow node
    assert {k: v.status for k, v in first.items()} == {"edge-server": "green", "device1": "green"}
    assert "device2" not in prober.snapshot()["nodes"]  # unknown: no colour yet
    second = prober.probe_round()
    assert "device2" not in second
    assert sum(1 for c in runner.calls if c[-1] == slow_ip) == 1  # no second probe in flight
    time.sleep(1.2)
    prober.probe_round()
    assert prober.snapshot()["nodes"]["device2"]["status"] == "red"
    assert prober.round == 3


def test_index_page_lists_the_runtimes_and_the_catalogue_describes_them(
    client: FlaskClient,
) -> None:
    html = client.get("/").get_data(as_text=True)
    assert 'id="btn-runtimes"' in html and 'id="scenario-runtime"' in html
    assert 'value="headsplit"' in html and 'value="testbed"' in html
    catalog = client.get("/api/catalog").get_json()
    runtimes = {r["name"]: r for r in catalog["runtimes"]}
    assert catalog["default_runtime"] == "testbed"
    assert runtimes["headsplit"]["external"] is True
    assert runtimes["headsplit"]["schema"]["properties"]["controller"]["x-nnpm"] == {
        "widget": "node"
    }
    assert [a["name"] for a in runtimes["headsplit"]["actions"]] == [
        "install",
        "profile",
        "checkpoint",
    ]
    assert runtimes["testbed"]["schema"] == {} and runtimes["testbed"]["actions"] == []
    listing = client.get("/api/runtimes").get_json()
    assert {r["name"] for r in listing["runtimes"]} == {"headsplit", "testbed"}
    assert client.get("/api/runtimes/nope/status").status_code == 404
    assert client.post("/api/runtimes/testbed/actions/nope", json={}).status_code == 404


def test_sweep_runs_its_variants_in_sequence_with_a_summary_page(
    client: FlaskClient, state: AppState
) -> None:
    scenario = {
        "name": "api_sweep",
        "model": "fake",
        "repetitions": 1,
        "warmup": 0,
        "settle_s": 0,
        "start_lead_s": 0.2,
        "monitoring": {"enabled": False},
        "optimisation": {"enabled": True, "algorithm": "greedy", "initial_scheme": "50%"},
        "edge_server": {"profile_capacity": True, "nominal_frequency": False},
        "sweep": {"path": "edge_server.policy", "values": ["RR", "SJF"]},
    }
    started = client.post("/api/runs", json={"scenario": scenario})
    assert started.status_code == 202, started.get_json()
    first = started.get_json()["run_id"]
    listing = client.get("/api/runs").get_json()
    assert listing["busy"] is True and len(listing["runs"]) == 2
    assert {r["status"] for r in listing["runs"]} <= {"queued", "running"}
    assert all(r["sweep_origin"]["name"] == "api_sweep" for r in listing["runs"])
    # Wait for the whole group.
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline and client.get("/api/runs").get_json()["busy"]:
        time.sleep(0.5)
    runs = client.get("/api/runs").get_json()["runs"]
    assert [r["status"] for r in runs] == ["completed", "completed"], runs
    values = sorted(r["sweep_origin"]["value"] for r in runs)
    assert values == ["RR", "SJF"]
    second = next(r["run_id"] for r in runs if r["run_id"] != first)
    detail = client.get(f"/api/runs/{second}").get_json()
    assert detail["outputs"]["optimised_objective"]["policy"] in ("RR", "SJF")
    page = client.get(f"/runs/{second}")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Sweep api_sweep: edge_server.policy" in html
    assert html.count('class="status-completed"') >= 2
    assert "total_averages_of_benchmark_results.json" in html
    assert "optimised_vector" in html and first in html
    assert client.get("/runs/nope").status_code == 404


def test_headsplit_through_the_api_with_checks_actions_and_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.integration.test_headsplit_runtime import make_stub

    stub = make_stub(tmp_path, with_json=True)
    monkeypatch.setenv("TESTBED_HEADSPLIT_DIR", str(stub))
    monkeypatch.setenv("TESTBED_HEADSPLIT_COMMIT", "abcdef1")
    runner = FakeRunner(strict=False).on(prefix("ping"), CommandResult(("ping",), 0, PING_OK))
    config, _ = local_mode.create_local_testbed(tmp_path / "tb", devices=3)
    st = AppState(
        config,
        tmp_path / "tb" / "nodes.json",
        local=True,
        results_dir=tmp_path / "tb" / "results",
        scenario_root=tmp_path / "tb" / "scenarios",
        probe_runner=runner,
    )
    app = gui_app.create_app(st, start_prober=False)
    app.config["TESTING"] = True
    client = app.test_client()
    try:
        status = client.get("/api/runtimes/headsplit/status").get_json()
        assert set(status["nodes"]) == {"edge-server", "device1", "device2", "device3"}
        assert status["nodes"]["edge-server"] == []
        checks = {c["name"]: c["status"] for c in status["nodes"]["device1"]}
        assert checks == {
            "checkout": "ok",
            "environment": "ok",
            "profile": "warn",
            "checkpoint": "warn",
        }
        # The profile action runs as a task whose log streams like a run's log.
        started = client.post(
            "/api/runtimes/headsplit/actions/profile",
            json={"controller": "device1", "chunks": 3},
        )
        assert started.status_code == 202, started.get_json()
        task_id = started.get_json()["task_id"]
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            task = client.get(f"/api/tasks/{task_id}").get_json()
            if task["status"] != "running":
                break
            time.sleep(0.2)
        assert task["status"] == "completed", task
        assert any("profiled" in line for line in task["log"])
        events = client.get(f"/events/tasks/{task_id}?limit=2").get_data(as_text=True)
        assert "event: log" in events
        assert json.loads(Path(stub, "artifacts", "profile.json").read_text())["chunks"] == 3
        status = client.get("/api/runtimes/headsplit/status").get_json()
        assert {c["name"]: c["status"] for c in status["nodes"]["device1"]}["profile"] == "ok"
        tasks = client.get("/api/tasks").get_json()
        assert tasks["busy"] is False and tasks["tasks"][0]["action"] == "profile"
        # A HeadSplit experiment through the API, then its results on the summary page.
        scenario = {
            "name": "api_hs",
            "runtime": "headsplit",
            "model": "tinyllama",
            "repetitions": 1,
            "settle_s": 0,
            "edge_server": {"nominal_frequency": False, "profile_capacity": False},
            "headsplit": {
                "controller": "device1",
                "workers": ["device2", "device3"],
                "mode": "all",
            },
            "monitoring": {"enabled": False},
            "optimisation": {"enabled": False},
        }
        run = client.post("/api/runs", json={"scenario": scenario, "run_id": "hsapi"})
        assert run.status_code == 202, run.get_json()
        detail = wait_for_run(client, "hsapi")
        assert detail["status"] == "completed", detail.get("error")
        assert detail["runtime"] == "headsplit"
        assert detail["outputs"]["headsplit"]["headsplit"] == 41000.0
        html = client.get("/runs/hsapi").get_data(as_text=True)
        assert "HeadSplit results" in html and "layer_based" in html and "JSON results" in html
        assert "41000.0" in html and "1.52" in html
    finally:
        st.runs.wait(timeout=120)
        st.tasks.wait(timeout=60)
