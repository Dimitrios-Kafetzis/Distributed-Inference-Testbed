# Hardware validation checklist

Everything in this repository is tested without hardware: with fake command runners, fake clocks, a fake sysfs tree and a local mode that simulates all nodes on one machine. The items below are the physical checks that only the real testbed can answer. Each item names the requirement it closes (IDs from the thesis requirement catalogue), the steps, and the expected outcome. Record the outcome and the date next to each item.

## Software stack (A-07, J-19)

- [ ] On one Raspberry Pi 4 (Raspberry Pi OS Bookworm 64-bit, Python 3.11): create a virtual environment and install `requirements/edge_device.lock` with `pip install --require-hashes -r requirements/edge_device.lock`. Expected: every wheel is found for `aarch64` and `cp311`; `python -c "import tensorflow; print(tensorflow.__version__)"` prints `2.15.1`.
- [ ] Same device: install `requirements/edge_device_llm.lock`. Expected: `torch` 2.1.2 and `transformers` 4.36.2 import; `torch.get_num_threads()` returns 4.
- [ ] Note the wall-clock time of each installation (J-15; the thesis states about 25 minutes per device for the whole deployment).

## Process control (C-02)

- [ ] On a device, run `python -m src.config --env` through SSH and confirm the printed `TESTBED_*` values match `config.yaml`.
- [ ] Start a tagged process over SSH with the start command of `src/common/process_tags.py` (for example a `sleep 60`), close the SSH session, reconnect and run the check command. Expected: `running`. Run the stop command. Expected: `stopped`, no process with the tag in `ps`, PID file removed.

(Items are added with every work package.)
