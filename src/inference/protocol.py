"""Wire protocol between a device and the edge-server inference service.

A message is a JSON header and a binary payload, each prefixed with its
length as a 4-byte big-endian unsigned integer::

    | len(header) | header JSON | len(payload) | payload bytes |

Message kinds (``header["kind"]``):

* ``request`` (device -> server): run id, node id, model, partition point s,
  repetition, the client's send timestamps; payload = the cut tensor
  (NumPy ``.npy`` bytes);
* ``ack`` (server -> device): sent as soon as the last payload byte has been
  read, with the server's arrival timestamps; the device measures the
  transmission delay from its first byte sent to this acknowledgement;
* ``result`` (server -> device): service start and end timestamps, queue wait,
  remote computation time; payload = the output tensor;
* ``error`` (server -> device): a message.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 1 << 31
_LEN = struct.Struct("!I")


class ProtocolError(RuntimeError):
    pass


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(min(1 << 20, size - len(chunks)))
        if not chunk:
            raise ProtocolError("connection closed while receiving")
        chunks.extend(chunk)
    return bytes(chunks)


def encode(header: dict[str, Any], payload: bytes = b"") -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _LEN.pack(len(header_bytes)) + header_bytes + _LEN.pack(len(payload)) + payload


def send_message(sock: socket.socket, header: dict[str, Any], payload: bytes = b"") -> int:
    """Send; return the number of bytes written."""
    data = encode(header, payload)
    sock.sendall(data)
    return len(data)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    (header_len,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    if header_len > MAX_HEADER_BYTES:
        raise ProtocolError(f"header too large: {header_len} bytes")
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    (payload_len,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload too large: {payload_len} bytes")
    payload = _recv_exact(sock, payload_len) if payload_len else b""
    if not isinstance(header, dict) or "kind" not in header:
        raise ProtocolError("malformed header")
    return header, payload
