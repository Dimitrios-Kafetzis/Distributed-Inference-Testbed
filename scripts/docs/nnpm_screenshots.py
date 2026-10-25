"""Screenshots of every NNPM view for the tutorial, taken in local mode.

Starts NNPM with five simulated devices and a stub HeadSplit checkout, drives the
page with Playwright (dialogs, runs, summary pages, the Runtimes dialog) and writes
PNG files to ``docs/images/nnpm/``. Nothing here needs hardware or the internet
apart from the map tiles; with ``--offline`` the map draws its grid instead.

Usage (from the repository root, with the documentation environment)::

    python scripts/docs/nnpm_screenshots.py [--out docs/images/nnpm] [--server-python PY]
        [--offline] [--port 8019]

The server runs with ``--server-python`` (the testbed's own environment, default:
the interpreter of this script) while Playwright and Pillow come from
``requirements/docs.in``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIEWPORT = {"width": 1440, "height": 1000}
TALL = {"width": 1440, "height": 2400}  # dialogs scroll internally: show them whole

CONTROLLER_OUTPUT = """controller: RPC group of 5 (127.0.0.1:29500) ...
measuring device capabilities and links ...
speculating layers: [2, 5, 9]
running layer_based ...
layer_based   total   62400.0 ms  prefill   1200.5 ms  decode  1224.0 ms/token  payload   12.34 MB
  The history of Athens begins
running head_only ...
head_only     total   45100.0 ms  prefill   900.0 ms  decode   884.0 ms/token  payload   40.10 MB
running headsplit ...
headsplit     total   40000.0 ms  prefill   880.0 ms  decode   782.4 ms/token  payload   41.00 MB  \
speculation 44/50 accepted  re-placements 1

speedup vs layer-based:
  head_only    1.38 times
  headsplit    1.56 times
"""


def make_headsplit_stub(root: Path) -> Path:
    """A fake HeadSplit checkout whose scripts print what the real ones print
    (the same stub the tests use), so the Runtimes dialog and a HeadSplit run work
    without the real runtime."""
    scripts = root / "headsplit" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "run_worker.py").write_text(
        "import argparse, time\n"
        "p = argparse.ArgumentParser()\n"
        "for a in ('--name', '--rank', '--world-size', '--master-addr', '--master-port',"
        " '--model'):\n"
        "    p.add_argument(a)\n"
        "a = p.parse_args()\n"
        "print(f'{a.name}: joining RPC group ({a.master_addr}:{a.master_port}) ...', flush=True)\n"
        "print(f'{a.name}: ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    (scripts / "run_controller.py").write_text(
        "import argparse, json, sys\n"
        "p = argparse.ArgumentParser()\n"
        "for a in ('--config', '--profile', '--master-addr', '--master-port', '--prompt',"
        " '--tokens', '--mode', '--json'):\n"
        "    p.add_argument(a)\n"
        "a = p.parse_args()\n"
        f"sys.stdout.write({CONTROLLER_OUTPUT!r})\n"
        "if a.json:\n"
        "    json.dump({'modes': {'layer_based': {'total_ms': 62400.0, 'prefill_ms': 1200.5,"
        " 'decode_ms_per_token': 1224.0, 'payload_mb': 12.34},"
        " 'head_only': {'total_ms': 45100.0, 'prefill_ms': 900.0, 'decode_ms_per_token': 884.0,"
        " 'payload_mb': 40.1},"
        " 'headsplit': {'total_ms': 40000.0, 'prefill_ms': 880.0, 'decode_ms_per_token': 782.4,"
        " 'payload_mb': 41.0, 'speculation_accepted': 44, 'speculation_attempts': 50,"
        " 'replacements': 1}}, 'speedup_vs_layer_based': {'head_only': 1.38, 'headsplit': 1.56},"
        " 'rtt_ms': {'worker1': 9.5}}, open(a.json, 'w'))\n"
    )
    (scripts / "profile_offline.py").write_text(
        "import argparse, json, os\n"
        "p = argparse.ArgumentParser()\n"
        "for a in ('--model', '--chunks', '--chunk-len', '--epsilon', '--out'):\n"
        "    p.add_argument(a)\n"
        "a = p.parse_args()\n"
        "os.makedirs(os.path.dirname(a.out), exist_ok=True)\n"
        "profile = {'model': a.model, 'alpha': {str(i): 0.9 for i in range(22)}}\n"
        "json.dump(profile, open(a.out, 'w'))\n"
        "print('profiled', a.model)\n"
    )
    for script in scripts.iterdir():
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return root / "headsplit"


class Server:
    def __init__(self, python: str, root: Path, port: int, offline: bool, env: dict[str, str]):
        self.base = f"http://127.0.0.1:{port}"
        environment = {**os.environ, **env}
        if offline:
            environment["TESTBED_MAP_OFFLINE"] = "true"
        self.process = subprocess.Popen(
            [
                python,
                "-m",
                "gui.app",
                "--local",
                str(root),
                "--devices",
                "5",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=REPO,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{self.base}/healthz", timeout=1)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("NNPM did not start")

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=10) as resp:
            return json.loads(resp.read())

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def wait_runs(self, limit_s: float = 600) -> None:
        deadline = time.monotonic() + limit_s
        while time.monotonic() < deadline and self.get("/api/runs")["busy"]:
            time.sleep(0.5)

    def wait_task(self, task_id: str, limit_s: float = 120) -> dict:
        deadline = time.monotonic() + limit_s
        while time.monotonic() < deadline:
            task = self.get(f"/api/tasks/{task_id}")
            if task["status"] != "running":
                return task
            time.sleep(0.3)
        raise RuntimeError("task did not finish")

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(10)
        except subprocess.TimeoutExpired:
            self.process.kill()


def save(page_or_element, path: Path, **kwargs) -> None:
    """Screenshot, then quantise to 255 colours so the repository stays small."""
    path.parent.mkdir(parents=True, exist_ok=True)
    page_or_element.screenshot(path=str(path), **kwargs)
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        image.quantize(255, dither=Image.Dither.FLOYDSTEINBERG).save(path, optimize=True)
    except ImportError:
        pass
    print(f"wrote {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "images" / "nnpm")
    ap.add_argument("--server-python", default=sys.executable)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--port", type=int, default=8019)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    work = Path(tempfile.mkdtemp(prefix="nnpm-shots-"))
    stub = make_headsplit_stub(work)
    root = work / "testbed"
    out = args.out
    server = Server(
        args.server_python,
        root,
        args.port,
        args.offline,
        {"TESTBED_HEADSPLIT_DIR": str(stub), "TESTBED_HEADSPLIT_COMMIT": "4864a22"},
    )
    try:
        time.sleep(5)  # the first probe rounds, so that every marker is green
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
            page.goto(server.base)
            page.wait_for_selector(".leaflet-tooltip.status-green", timeout=20000)
            page.wait_for_timeout(2500)  # map tiles
            save(page, out / "01_main_view.png")

            # Node panel
            page.click('.leaflet-marker-icon[title="device2"]')
            page.wait_for_selector("#node-info menu")
            save(page.locator("#side-panel"), out / "02_node_panel.png")

            # Edit network
            page.click("#btn-edit-network")
            page.wait_for_selector("#dlg-network[open]")
            save(page.locator("#dlg-network"), out / "03_edit_network.png")
            page.click("#dlg-network [data-close]")

            # Add node by clicking the map
            page.click("#btn-add-mode")
            box = page.locator("#map").bounding_box()
            page.mouse.click(box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.3)
            page.wait_for_selector("#dlg-network[open]")
            page.fill('#form-node [name="id"]', "device6")
            page.fill('#form-node [name="ip"]', "192.0.2.16")
            save(page.locator("#dlg-network"), out / "04_add_node.png")
            page.click("#dlg-network [data-close]")
            page.click("#btn-add-mode")  # leave add mode

            # Create scenario: testbed runtime, model parameters, per node, sweep
            page.set_viewport_size(TALL)
            page.click("#btn-scenario")
            page.wait_for_selector("#dlg-scenario[open]")
            page.fill('#form-scenario [name="name"]', "tutorial_alexnet")
            page.select_option("#scenario-model", "fake")
            page.fill('#form-scenario [name="sweep_path"]', "edge_server.policy")
            page.fill('#form-scenario [name="sweep_values"]', "RR, SJF")
            page.click("#btn-derive-latency")
            page.select_option('#form-scenario [name="loadmode_device2"]', "regime")
            page.fill('#form-scenario [name="freq_device3"]', "1200")
            page.check('#form-scenario [name="inference_enabled"]')
            page.fill('#form-scenario [name="partition_points"]', "0,2,5")
            page.wait_for_timeout(300)
            save(page.locator("#dlg-scenario"), out / "05_create_scenario.png")

            # HeadSplit runtime block
            page.select_option("#scenario-runtime", "headsplit")
            page.wait_for_selector("#runtime-block .schema-block")
            page.select_option('#form-scenario [name="rt__controller"]', "device1")
            for worker in ("device2", "device3", "device4", "device5"):
                page.check(f'#form-scenario input[name="rt__workers"][value="{worker}"]')
            page.select_option('#form-scenario [name="rt__mode"]', "all")
            save(page.locator("#runtime-block"), out / "06_create_scenario_headsplit.png")
            page.select_option("#scenario-runtime", "testbed")
            page.click("#dlg-scenario [data-close]")
            page.set_viewport_size(VIEWPORT)

            # Benchmarking dialog, then a run started from it
            page.click("#btn-benchmark")
            page.wait_for_selector("#dlg-benchmark[open]")
            page.select_option('#form-benchmark [name="model"]', "fake")
            page.fill('#form-benchmark [name="repetitions"]', "2")
            save(page.locator("#dlg-benchmark"), out / "07_benchmarking.png")
            page.click('#form-benchmark button[type="submit"]')
            page.wait_for_selector(".indicator.running", timeout=20000)
            page.wait_for_timeout(6000)
            save(page, out / "08_run_in_progress.png")
            server.wait_runs()
            page.wait_for_timeout(1500)
            runs = server.get("/api/runs")["runs"]
            benchmark_run = runs[0]["run_id"]
            save(page, out / "09_run_finished.png")

            # Run summary page
            page.goto(f"{server.base}/runs/{benchmark_run}")
            page.wait_for_selector("table.facts")
            save(page, out / "10_run_summary.png", full_page=True)
            page.goto(server.base)
            page.wait_for_selector(".leaflet-tooltip.status-green", timeout=20000)

            # Optimise dialog with a result
            page.click("#btn-optimise")
            page.wait_for_selector("#dlg-optimise[open]")
            page.wait_for_timeout(500)
            page.select_option("#optimise-runs", benchmark_run)
            page.select_option('#form-optimise [name="policy"]', "RR")
            page.fill('#form-optimise [name="initial_scheme"]', "50%")
            page.click('#form-optimise button[type="submit"]')
            page.wait_for_selector("#optimise-result table", timeout=60000)
            save(page.locator("#dlg-optimise"), out / "11_optimise.png")
            page.click("#dlg-optimise [data-close]")

            # Runtimes dialog: checks before and after the profile action
            page.set_viewport_size(TALL)
            page.click("#btn-runtimes")
            page.wait_for_selector("#dlg-runtimes[open]")
            page.select_option("#runtimes-select", "headsplit")
            page.click("#btn-runtimes-refresh")
            page.wait_for_selector("#runtimes-status table", timeout=60000)
            save(page.locator("#dlg-runtimes"), out / "12_runtimes_checks.png")
            page.select_option(
                '#dlg-runtimes form[data-action="profile"] [name="act_profile__controller"]',
                "device1",
            )
            page.click('#dlg-runtimes form[data-action="profile"] button[type="submit"]')
            page.wait_for_timeout(500)
            tasks = server.get("/api/tasks")
            server.wait_task(tasks["tasks"][-1]["task_id"])
            page.wait_for_timeout(1000)
            page.click("#btn-runtimes-refresh")
            page.wait_for_selector("#runtimes-status table", timeout=60000)
            page.wait_for_timeout(500)
            save(page.locator("#dlg-runtimes"), out / "13_runtimes_after_profile.png")
            page.click("#dlg-runtimes [data-close]")
            page.set_viewport_size(VIEWPORT)
            save(page.locator("#terminal-panel"), out / "14_terminal_task_log.png")

            # A HeadSplit experiment (stub runtime) and its summary page
            scenario = {
                "name": "tutorial_headsplit",
                "runtime": "headsplit",
                "model": "tinyllama",
                "repetitions": 1,
                "settle_s": 0,
                "edge_server": {"nominal_frequency": False, "profile_capacity": False},
                "headsplit": {
                    "controller": "device1",
                    "workers": ["device2", "device3", "device4", "device5"],
                    "mode": "all",
                },
                "monitoring": {"enabled": False},
                "optimisation": {"enabled": False},
            }
            started = server.post("/api/runs", {"scenario": scenario})
            server.wait_runs()
            page.goto(f"{server.base}/runs/{started['run_id']}")
            page.wait_for_selector("table.results")
            save(page, out / "15_headsplit_summary.png", full_page=True)

            # A sweep (two runs) and its summary page
            sweep = {
                "name": "tutorial_sweep",
                "model": "fake",
                "repetitions": 1,
                "warmup": 0,
                "settle_s": 0,
                "start_lead_s": 0.2,
                "monitoring": {"enabled": False},
                "edge_server": {"profile_capacity": True, "nominal_frequency": False},
                "optimisation": {"enabled": True, "algorithm": "greedy", "initial_scheme": "50%"},
                "sweep": {"path": "edge_server.policy", "values": ["RR", "SJF"]},
            }
            started = server.post("/api/runs", {"scenario": sweep})
            server.wait_runs()
            runs = server.get("/api/runs")["runs"]
            sweep_run = next(r["run_id"] for r in runs if r.get("sweep_origin"))
            page.goto(f"{server.base}/runs/{sweep_run}")
            page.wait_for_selector("table.results")
            save(page, out / "16_sweep_summary.png", full_page=True)
            page.goto(server.base)
            page.wait_for_selector("#runs-list li")
            page.wait_for_timeout(1500)
            save(page.locator("#runs-panel"), out / "17_runs_panel.png")
            browser.close()
    finally:
        server.stop()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
