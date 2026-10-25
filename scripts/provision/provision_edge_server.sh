#!/usr/bin/env bash
#
# Provisions the edge server (Ubuntu 22.04 LTS): packages, SSH key, the isolated
# 5 GHz access point (hostapd), DHCP and DNS for both testbed networks (dnsmasq),
# the NTP server (chrony), the static management address, the package mirror
# service, the Python environment from the lock file and the NNPM service.
#
# Usage: provision_edge_server.sh [--dry-run] [--allow-upstream] [--skip-apt]
#
# --allow-upstream installs the Python environment from PyPI instead of the mirror
# (needed the first time, before build_mirror.sh has run on this machine).
#
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

ALLOW_UPSTREAM=0
SKIP_APT=0
take_common_flags "$@"
for arg in "${ARGS[@]+"${ARGS[@]}"}"; do
    case "$arg" in
        --allow-upstream) ALLOW_UPSTREAM=1 ;;
        --skip-apt) SKIP_APT=1 ;;
        *) die "unknown argument $arg" ;;
    esac
done

[ -f "${TESTBED_CONFIG:-$CODE_DIR/config.yaml}" ] || die "copy config.example.yaml to config.yaml and edit it first"
[ -f "${TESTBED_NODES:-$CODE_DIR/nodes.json}" ] || die "copy nodes.example.json to nodes.json and edit it first"
load_variables

if [ "$SKIP_APT" != 1 ]; then
    # shellcheck disable=SC2046
    run sudo apt-get update
    # shellcheck disable=SC2046
    run sudo apt-get install -y --no-install-recommends $(package_list "$PROVISION_DIR/apt_packages_edge_server.txt")
fi

# SSH key that prepare_image.sh installs on every device.
KEY="${SSH_PUBLIC_KEY_FILE:-$HOME/.ssh/id_ed25519}"
if [ ! -f "$KEY" ]; then
    run mkdir -p "$(dirname "$KEY")"
    run ssh-keygen -q -t ed25519 -N "" -f "$KEY" -C "testbed edge server"
fi

OUT=$(render_files --role edge-server)
trap 'rm -rf "$OUT"' EXIT
install_file "$OUT/etc/hostapd/hostapd.conf" /etc/hostapd/hostapd.conf 600
install_file "$OUT/etc/dnsmasq.d/testbed.conf" /etc/dnsmasq.d/testbed.conf 644
install_file "$OUT/etc/testbed-hosts" /etc/testbed-hosts 644
install_file "$OUT/etc/chrony/conf.d/testbed.conf" /etc/chrony/conf.d/testbed.conf 644
install_file "$OUT/etc/NetworkManager/conf.d/testbed-unmanaged.conf" \
    /etc/NetworkManager/conf.d/testbed-unmanaged.conf 644
install_file "$OUT/etc/NetworkManager/system-connections/testbed-mgmt.nmconnection" \
    /etc/NetworkManager/system-connections/testbed-mgmt.nmconnection 600
install_file "$OUT/etc/systemd/system/testbed-ap-address.service" \
    /etc/systemd/system/testbed-ap-address.service 644
install_file "$OUT/etc/systemd/system/testbed-mirror.service" \
    /etc/systemd/system/testbed-mirror.service 644
install_file "$OUT/etc/systemd/system/nnpm.service" /etc/systemd/system/nnpm.service 644

run sudo sed -i 's|^#\?DAEMON_CONF=.*|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd
run sudo rfkill unblock wlan
run sudo systemctl unmask hostapd
run sudo systemctl daemon-reload
run sudo nmcli connection reload
run sudo systemctl enable --now testbed-ap-address hostapd dnsmasq chrony

run mkdir -p "$MIRROR_DIR/wheels/x86_64" "$MIRROR_DIR/wheels/aarch64" "$MIRROR_DIR/apt" "$MIRROR_DIR/src"
run sudo systemctl enable --now testbed-mirror

VENV="$CODE_DIR/.venv"
[ -x "$VENV/bin/python" ] || run python3.11 -m venv "$VENV"
if [ "$ALLOW_UPSTREAM" = 1 ]; then
    run "$VENV/bin/pip" install --require-hashes -r "$CODE_DIR/requirements/edge_server.lock"
else
    run "$VENV/bin/pip" install --no-index --find-links "$MIRROR_DIR/wheels/x86_64" \
        --require-hashes -r "$CODE_DIR/requirements/edge_server.lock"
fi
run mkdir -p "$BS_RESULTS_DIR" "$BS_MODELS_DIR"
run sudo systemctl enable --now nnpm
log "NNPM: http://$EDGE_MGMT_IP:$NNPM_PORT/  mirror: $MIRROR_URL  access point: $AP_SSID on $AP_INTERFACE"
timing_log provision_edge_server
