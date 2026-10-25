"""Render the provisioning templates from ``config.yaml`` and ``nodes.json``.

Templates live in ``scripts/provision/templates/`` and use ``${NAME}``
placeholders (``string.Template``). Rendering is strict: a placeholder without a
value is an error, so a template can never be installed half-filled.

Usage::

    python -m src.provisioning.render --role edge-server --out DIR
    python -m src.provisioning.render --role device --node device1 --out DIR
    python -m src.provisioning.render --vars [--node ID]        # shell exports
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from src.common.nodes import DEFAULT_NODES_PATH, Node, Topology, load_topology
from src.config import REPO_ROOT, Config, load_config
from src.provisioning import networks

TEMPLATE_DIR = REPO_ROOT / "scripts" / "provision" / "templates"
PASSPHRASE_PLACEHOLDER = "CHANGE-ME-SET-THE-PASSPHRASE-VARIABLE"
PUBLIC_KEY_PLACEHOLDER = "ssh-ed25519 AAAA-REPLACE-WITH-THE-EDGE-SERVER-PUBLIC-KEY"
PASSWORD_HASH_PLACEHOLDER = "!"  # a locked password: key-based SSH only
DOMAIN = "testbed"

# template name -> installed path (relative to the target root)
EDGE_SERVER_FILES: dict[str, str] = {
    "hostapd.conf": "etc/hostapd/hostapd.conf",
    "dnsmasq.conf": "etc/dnsmasq.d/testbed.conf",
    "hosts": "etc/testbed-hosts",
    "chrony_server.conf": "etc/chrony/conf.d/testbed.conf",
    "nm_unmanaged.conf": "etc/NetworkManager/conf.d/testbed-unmanaged.conf",
    "nm_eth_edge.nmconnection": "etc/NetworkManager/system-connections/testbed-mgmt.nmconnection",
    "ap_address.service": "etc/systemd/system/testbed-ap-address.service",
    "mirror.service": "etc/systemd/system/testbed-mirror.service",
    "nnpm.service": "etc/systemd/system/nnpm.service",
}
DEVICE_FILES: dict[str, str] = {
    "custom.toml": "boot/firmware/custom.toml",
    "chrony_client.conf": "etc/chrony/conf.d/testbed.conf",
    "sudoers": "etc/sudoers.d/testbed",
    "pip.conf": "etc/pip.conf",
    "sources.list": "etc/apt/sources.list.d/testbed-mirror.list",
    "nm_wifi.nmconnection": "etc/NetworkManager/system-connections/testbed-wlan.nmconnection",
    "nm_eth.nmconnection": "etc/NetworkManager/system-connections/testbed-mgmt.nmconnection",
}


class RenderError(ValueError):
    """A template could not be rendered (missing node, missing variable)."""


@dataclass
class Rendered:
    files: dict[str, str]  # installed path -> content
    warnings: list[str]


def edge_server_node(topology: Topology) -> Node:
    for node in topology.nodes:
        if node.role == "Base Station":
            return node
    raise RenderError("nodes.json has no edge server (role 'Base Station')")


def mirror_url(config: Config, topology: Topology) -> str:
    if config.mirror_url:
        return config.mirror_url.rstrip("/")
    return f"http://{edge_server_node(topology).management_address}:{config.mirror_port}"


def _public_key(config: Config, warnings: list[str]) -> str:
    path = Path(config.ssh_public_key_path or "~/.ssh/id_ed25519.pub").expanduser()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    warnings.append(f"public key {path} not found; custom.toml carries a placeholder")
    return PUBLIC_KEY_PLACEHOLDER


def _passphrase(config: Config, environ: Mapping[str, str], warnings: list[str]) -> str:
    name = config.ap_passphrase_env
    if name and environ.get(name):
        return environ[name]
    warnings.append(
        f"passphrase variable {name or '(ap_passphrase_env)'} is not set; "
        "the rendered files carry a placeholder"
    )
    return PASSPHRASE_PLACEHOLDER


def common_variables(
    config: Config, topology: Topology, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Variables shared by every template, and the warnings raised while collecting them."""
    env = dict(os.environ if environ is None else environ)
    warnings: list[str] = []
    edge = edge_server_node(topology)
    data_prefix = config.base_ip_prefix
    mgmt_prefix = config.management_prefix
    data_range = networks.dhcp_range(data_prefix)
    mgmt_range = networks.dhcp_range(mgmt_prefix)
    devices = [n for n in topology.nodes if n.role != "Base Station"]
    url = mirror_url(config, topology)
    host = url.split("://", 1)[1].split("/", 1)[0].rsplit(":", 1)[0]
    reservations = "\n".join(f"dhcp-host={n.id},{n.management_address}" for n in devices)
    hosts = "\n".join(f"{n.management_address} {n.id} {n.id}.{DOMAIN}" for n in [edge, *devices])
    variables = {
        "SSH_USERNAME": config.ssh_username,
        "SSH_KNOWN_HOSTS": config.ssh_known_hosts_path or "",
        "SSH_CONNECT_TIMEOUT": f"{config.ssh_connect_timeout_s:g}",
        "BS_RESULTS_DIR": str(config.bs_results_path),
        "HEADSPLIT_REPO_URL": config.headsplit_repo_url,
        "BS_USER": env.get("SUDO_USER") or env.get("USER") or config.ssh_username,
        "BS_ROOT": str(config.bs_root_path),
        "BS_MODELS_DIR": str(config.bs_models_path),
        "ED_ROOT": config.ed_root.rstrip("/"),
        "ED_CODE_DIR": config.ed_code_dir,
        "ED_VENV_DIR": config.ed_venv_dir,
        "ED_MODELS_DIR": config.ed_models_path,
        "DATA_INTERFACE": config.network_interface,
        "MGMT_INTERFACE": config.management_interface,
        "BS_MGMT_INTERFACE": config.bs_management_interface,
        "AP_INTERFACE": config.ap_interface,
        "AP_SSID": config.ap_ssid,
        "AP_CHANNEL": str(config.ap_channel),
        "AP_COUNTRY": config.ap_country,
        "AP_HT40": networks.ht40_direction(config.ap_channel),
        "AP_VHT_CENTRE": str(networks.vht_centre_index(config.ap_channel)),
        "AP_PASSPHRASE": _passphrase(config, env, warnings),
        "DATA_NETWORK": networks.network_cidr(data_prefix),
        "DATA_PREFIX_BITS": str(networks.prefix_bits(data_prefix)),
        "DATA_DHCP_RANGE": ",".join(data_range),
        "MGMT_NETWORK": networks.network_cidr(mgmt_prefix),
        "MGMT_PREFIX_BITS": str(networks.prefix_bits(mgmt_prefix)),
        "MGMT_DHCP_RANGE": ",".join(mgmt_range),
        "SEPARATE_MANAGEMENT": "yes" if config.management_ip_prefix else "no",
        "EDGE_DATA_IP": edge.data_address,
        "EDGE_MGMT_IP": edge.management_address,
        "NTP_SERVER": edge.management_address,
        "DHCP_HOSTS": reservations,
        "HOSTS": hosts,
        "DOMAIN": DOMAIN,
        "MIRROR_URL": url,
        "MIRROR_HOST": host,
        "MIRROR_PORT": str(config.mirror_port),
        "MIRROR_DIR": str(config.mirror_path),
        "NNPM_HOST": config.nnpm_host,
        "NNPM_PORT": str(config.nnpm_port),
        "SSH_PUBLIC_KEY": _public_key(config, warnings),
        "TIMEZONE": config.device_timezone,
        "HEADSPLIT_COMMIT": config.headsplit_commit or "",
        "HEADSPLIT_DIR": config.headsplit_path,
        "DEVICE_PASSWORD_HASH": PASSWORD_HASH_PLACEHOLDER,
    }
    return variables, warnings


def device_variables(
    config: Config,
    topology: Topology,
    node_id: str,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    variables, warnings = common_variables(config, topology, environ)
    try:
        node = topology.get(node_id)
    except KeyError as exc:
        raise RenderError(f"unknown node {node_id!r} in nodes.json") from exc
    if node.role == "Base Station":
        raise RenderError(f"{node_id} is the edge server; use --role edge-server")
    if not networks.in_prefix(node.data_address, config.base_ip_prefix):
        warnings.append(
            f"{node_id}: data address {node.data_address} is outside {config.base_ip_prefix}"
        )
    variables.update(
        {
            "NODE_ID": node.id,
            "DEVICE_DATA_IP": node.data_address,
            "DEVICE_MGMT_IP": node.management_address,
        }
    )
    variables.update(overrides or {})
    return variables, warnings


def render_template(name: str, variables: Mapping[str, str]) -> str:
    path = TEMPLATE_DIR / f"{name}.tmpl"
    if not path.exists():
        raise RenderError(f"template {path} not found")
    try:
        return Template(path.read_text(encoding="utf-8")).substitute(variables)
    except KeyError as exc:
        raise RenderError(f"{name}.tmpl: no value for placeholder {exc}") from exc


def render_edge_server(
    config: Config, topology: Topology, environ: Mapping[str, str] | None = None
) -> Rendered:
    variables, warnings = common_variables(config, topology, environ)
    files = {dst: render_template(name, variables) for name, dst in EDGE_SERVER_FILES.items()}
    return Rendered(files, warnings)


def render_device(
    config: Config,
    topology: Topology,
    node_id: str,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> Rendered:
    variables, warnings = device_variables(config, topology, node_id, environ, overrides)
    files = {dst: render_template(name, variables) for name, dst in DEVICE_FILES.items()}
    if variables["SEPARATE_MANAGEMENT"] == "no":
        # One network: the wired profile would duplicate the wireless address.
        files.pop(DEVICE_FILES["nm_eth.nmconnection"])
    return Rendered(files, warnings)


def write_rendered(rendered: Rendered, out_dir: Path) -> list[Path]:
    written = []
    for rel, content in rendered.files.items():
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written


def shell_exports(variables: Mapping[str, str]) -> str:
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in sorted(variables.items()))


def _parse_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.isidentifier():
            raise RenderError(f"--var expects NAME=value, got {item!r}")
        out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--nodes", default=None, help=f"default: {DEFAULT_NODES_PATH}")
    parser.add_argument("--role", choices=["edge-server", "device"], default="edge-server")
    parser.add_argument("--node", default=None, help="node id (role device)")
    parser.add_argument("--out", default=None, help="directory to write the rendered files to")
    parser.add_argument("--vars", action="store_true", help="print the variables as exports")
    parser.add_argument("--var", action="append", default=[], metavar="NAME=value")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    topology = load_topology(args.nodes)
    try:
        overrides = _parse_overrides(args.var)
        if args.role == "device":
            if not args.node:
                raise RenderError("--node is required with --role device")
            variables, warnings = device_variables(config, topology, args.node, overrides=overrides)
            rendered = render_device(config, topology, args.node, overrides=overrides)
        else:
            variables, warnings = common_variables(config, topology)
            variables.update(overrides)
            rendered = render_edge_server(config, topology)
    except RenderError as exc:
        print(f"render: {exc}", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"render: warning: {warning}", file=sys.stderr)
    if args.vars:
        print(shell_exports(variables))
    if args.out:
        for path in write_rendered(rendered, Path(args.out)):
            print(path)
    return 0


def rendered_as_dict(rendered: Rendered) -> dict[str, Any]:
    return {"files": rendered.files, "warnings": rendered.warnings}


if __name__ == "__main__":
    sys.exit(main())
