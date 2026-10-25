#!/usr/bin/env bash
#
# Device-side steps of the provisioning, run on the device by provision_device.sh
# (the code has been copied first). Each step is idempotent.
#
# Usage: device_setup.sh --rendered DIR --node ID --code DIR --venv DIR --locks NAME[,NAME]
#                        --mirror URL [--headsplit-dir DIR --headsplit-commit SHA] <step>...
# Steps: hostname apt chrony sudoers pip venv headsplit verify network
#
# TESTBED_FS_ROOT (tests only) prefixes every destination path.
#
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

RENDERED="" NODE="" CODE="" VENV="" LOCKS="" MIRROR="" HS_DIR="" HS_COMMIT=""
STEPS=()
# DRY_RUN is read by the helpers in lib.sh.
# shellcheck disable=SC2034
while [ "$#" -gt 0 ]; do
    case "$1" in
        --rendered) RENDERED=$2; shift 2 ;;
        --node) NODE=$2; shift 2 ;;
        --code) CODE=$2; shift 2 ;;
        --venv) VENV=$2; shift 2 ;;
        --locks) LOCKS=$2; shift 2 ;;
        --mirror) MIRROR=$2; shift 2 ;;
        --headsplit-dir) HS_DIR=$2; shift 2 ;;
        --headsplit-commit) HS_COMMIT=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) STEPS+=("$1"); shift ;;
    esac
done
[ -n "$RENDERED" ] && [ -n "$NODE" ] && [ -n "$CODE" ] && [ -n "$VENV" ] && [ -n "$MIRROR" ] ||
    die "missing arguments (see the usage at the top of $0)"
[ "${#STEPS[@]}" -ge 1 ] || die "no step given"
ROOT="${TESTBED_FS_ROOT:-}"

step_hostname() {
    [ "$(hostname)" = "$NODE" ] || run sudo hostnamectl set-hostname "$NODE"
}

step_apt() {
    # The mirror replaces every upstream source; the upstream lists are kept aside.
    run sudo mkdir -p "$ROOT/etc/apt/sources.upstream.disabled"
    local f
    for f in "$ROOT/etc/apt/sources.list" "$ROOT"/etc/apt/sources.list.d/*; do
        [ -e "$f" ] || continue
        case "$f" in */testbed-mirror.list) continue ;; esac
        run sudo mv "$f" "$ROOT/etc/apt/sources.upstream.disabled/"
    done
    install_file "$RENDERED/etc/apt/sources.list.d/testbed-mirror.list" \
        "$ROOT/etc/apt/sources.list.d/testbed-mirror.list" 644
    run sudo apt-get update
    # shellcheck disable=SC2046
    run sudo apt-get install -y --no-install-recommends $(package_list "$CODE/scripts/provision/apt_packages_device.txt")
}

step_chrony() {
    # The edge server becomes the only time source.
    run sudo sed -i 's/^\(pool\|server\) /# \1 /' "$ROOT/etc/chrony/chrony.conf"
    install_file "$RENDERED/etc/chrony/conf.d/testbed.conf" "$ROOT/etc/chrony/conf.d/testbed.conf" 644
    run sudo systemctl restart chrony
}

step_sudoers() {
    run sudo visudo -cf "$RENDERED/etc/sudoers.d/testbed"
    install_file "$RENDERED/etc/sudoers.d/testbed" "$ROOT/etc/sudoers.d/testbed" 440
}

step_pip() {
    install_file "$RENDERED/etc/pip.conf" "$ROOT/etc/pip.conf" 644
}

step_venv() {
    [ -x "$VENV/bin/python" ] || run python3 -m venv "$VENV"
    local lock
    for lock in ${LOCKS//,/ }; do
        run "$VENV/bin/pip" install --no-index --find-links "$MIRROR/wheels/aarch64/" \
            --require-hashes -r "$CODE/requirements/$lock.lock"
    done
}

step_headsplit() {
    [ -n "$HS_DIR" ] && [ -n "$HS_COMMIT" ] || die "headsplit needs --headsplit-dir and --headsplit-commit"
    [ -d "$HS_DIR/.git" ] || run git clone --quiet "$MIRROR/src/headsplit.git" "$HS_DIR"
    run git -C "$HS_DIR" fetch --quiet
    run git -C "$HS_DIR" checkout --quiet "$HS_COMMIT"
    [ -x "$HS_DIR/.venv/bin/python" ] || run python3 -m venv "$HS_DIR/.venv"
    run "$HS_DIR/.venv/bin/pip" install --no-index --find-links "$MIRROR/wheels/aarch64-headsplit/" \
        -r "$HS_DIR/requirements.txt"
}

step_verify() {
    run bash -c "cd $CODE && $VENV/bin/python -c 'import src.config'"
    run chronyc tracking
    run uname -m
}

step_network() {
    local f
    for f in "$RENDERED"/etc/NetworkManager/system-connections/*.nmconnection; do
        install_file "$f" "$ROOT/etc/NetworkManager/system-connections/$(basename "$f")" 600
    done
    run sudo nmcli connection reload
    run sudo nmcli connection up testbed-wlan
    if [ -f "$RENDERED/etc/NetworkManager/system-connections/testbed-mgmt.nmconnection" ]; then
        run sudo nmcli connection up testbed-mgmt
    fi
}

for step in "${STEPS[@]}"; do
    case "$step" in
        hostname | apt | chrony | sudoers | pip | venv | headsplit | verify | network)
            log "$NODE: $step"
            "step_$step"
            ;;
        *) die "unknown step $step" ;;
    esac
done
