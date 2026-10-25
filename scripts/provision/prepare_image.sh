#!/usr/bin/env bash
#
# Writes the first-boot configuration of one device onto a flashed Raspberry Pi OS
# Bookworm 64-bit image (its boot partition, mounted at <boot-mountpoint>):
# hostname (the node id), the device user with the edge server's SSH key, WiFi to
# the testbed access point, time zone. Everything else happens over SSH with
# provision_device.sh after the first boot.
#
# Usage: prepare_image.sh <node-id> <boot-mountpoint> [--dry-run]
#
# The device user's password comes from TESTBED_DEVICE_PASSWORD when set; otherwise
# the account gets a locked password and only the SSH key opens it.
#
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

take_common_flags "$@"
[ "${#ARGS[@]}" -eq 2 ] || {
    sed -n '3,13p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
}
NODE=${ARGS[0]}
BOOT=${ARGS[1]}

[ -d "$BOOT" ] || die "$BOOT is not a directory (mount the boot partition first)"
[ -f "$BOOT/config.txt" ] || die "$BOOT does not look like a Raspberry Pi boot partition"
# The 32-bit image ships kernel7l.img next to kernel8.img; the 64-bit image does not.
[ ! -f "$BOOT/kernel7l.img" ] ||
    die "32-bit image: the LLM experiments need the 64-bit Raspberry Pi OS (Chapter 3)"
[ -f "$BOOT/kernel8.img" ] || die "no kernel8.img: not a 64-bit Raspberry Pi OS image"

load_variables --node "$NODE"

if [ -n "${TESTBED_DEVICE_PASSWORD:-}" ]; then
    HASH=$(openssl passwd -6 "$TESTBED_DEVICE_PASSWORD")
else
    HASH='!'
    log "TESTBED_DEVICE_PASSWORD is not set: password login locked, SSH key only"
fi

OUT=$(render_files --role device --node "$NODE" --var "DEVICE_PASSWORD_HASH=$HASH")
trap 'rm -rf "$OUT"' EXIT

run install -m 644 "$OUT/boot/firmware/custom.toml" "$BOOT/custom.toml"
run touch "$BOOT/ssh"
log "$NODE: first-boot configuration written to $BOOT (hostname $NODE, user $SSH_USERNAME)"
log "boot the device on the management switch; it registers as $NODE at $DEVICE_MGMT_IP"
timing_log "prepare_image $NODE"
