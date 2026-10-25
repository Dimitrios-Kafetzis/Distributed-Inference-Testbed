"""Addresses derived from the configured prefixes.

``base_ip_prefix`` and ``management_ip_prefix`` are dotted prefixes with one to
three octets (``"192.168.0."``). The network is the prefix padded with zeros and
a mask of eight bits per octet given; the DHCP pool used before a device is
provisioned (and by any laptop plugged into the management switch) is a fixed
range inside that network, above the static addresses of ``nodes.json``.
"""

from __future__ import annotations

# Centre-frequency segment index of the 80 MHz block that holds each 5 GHz channel
# (hostapd ``vht_oper_centr_freq_seg0_idx``).
_VHT80_CENTRES = {
    42: (36, 40, 44, 48),
    58: (52, 56, 60, 64),
    106: (100, 104, 108, 112),
    122: (116, 120, 124, 128),
    138: (132, 136, 140, 144),
    155: (149, 153, 157, 161),
}


def prefix_octets(prefix: str) -> list[int]:
    parts = prefix.strip().rstrip(".").split(".")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        raise ValueError(f"invalid address prefix {prefix!r}")
    return [int(p) for p in parts]


def prefix_bits(prefix: str) -> int:
    return 8 * len(prefix_octets(prefix))


def network_cidr(prefix: str) -> str:
    octets = prefix_octets(prefix) + [0] * (4 - len(prefix_octets(prefix)))
    return ".".join(str(o) for o in octets) + f"/{prefix_bits(prefix)}"


def dhcp_range(prefix: str) -> tuple[str, str]:
    """Pool for unprovisioned devices: ``.100`` to ``.199`` of the last octet."""
    octets = prefix_octets(prefix)
    missing = 4 - len(octets)
    head = ".".join(str(o) for o in octets + [100] * (missing - 1))
    return f"{head}.100", f"{head}.199"


def in_prefix(address: str, prefix: str) -> bool:
    return address.startswith(prefix if prefix.endswith(".") else prefix + ".")


def vht_centre_index(channel: int) -> int:
    for centre, channels in _VHT80_CENTRES.items():
        if channel in channels:
            return centre
    raise ValueError(f"channel {channel} is not a 5 GHz 802.11ac channel")


def ht40_direction(channel: int) -> str:
    """``[HT40+]`` for the lower channel of each 40 MHz pair, ``[HT40-]`` otherwise."""
    return "[HT40+]" if (channel // 4) % 2 == 1 else "[HT40-]"
