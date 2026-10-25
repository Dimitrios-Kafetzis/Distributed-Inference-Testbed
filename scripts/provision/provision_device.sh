#!/usr/bin/env bash
#
# Provisions one edge device from the edge server over SSH (management network):
# 64-bit check, code copy, packages and Python environment from the mirror, NTP
# client, sudoers rule, static addresses, model weights, optional HeadSplit.
#
# Usage: provision_device.sh <node-id> [--address A] [--llm] [--skip-apt] [--skip-models] [--dry-run]
#
# --address A   reach the device at A instead of its nodes.json management address
#               (for example the DHCP address shown by `journalctl -u dnsmasq`)
# --llm         also install the PyTorch environment (edge_device_llm.lock)
#
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

take_common_flags "$@"
[ "${#ARGS[@]}" -ge 1 ] || {
    sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
}
NODE=${ARGS[0]}
ADDRESS=""
LOCKS="edge_device"
SKIP_APT=0
SKIP_MODELS=0
i=1
while [ "$i" -lt "${#ARGS[@]}" ]; do
    case "${ARGS[$i]}" in
        --address)
            i=$((i + 1))
            ADDRESS=${ARGS[$i]}
            ;;
        --llm) LOCKS="edge_device,edge_device_llm" ;;
        --skip-apt) SKIP_APT=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
        *) die "unknown argument ${ARGS[$i]}" ;;
    esac
    i=$((i + 1))
done

load_variables --node "$NODE"
HOST=${ADDRESS:-$DEVICE_MGMT_IP}

# Record the host key once (first contact), then never accept a changed key.
KNOWN="${SSH_KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
if ! ssh-keygen -F "$HOST" -f "$KNOWN" >/dev/null 2>&1; then
    run mkdir -p "$(dirname "$KNOWN")"
    run bash -c "ssh-keyscan -H $HOST >> $KNOWN"
fi

ARCH=$(remote_query "$HOST" uname -m) || die "cannot reach $NODE at $HOST over SSH"
[ "$ARCH" = aarch64 ] ||
    die "$NODE runs a $ARCH kernel; the 64-bit Raspberry Pi OS is required (Chapter 3: the LLM experiments exceed the 3 GB per-process limit of 32-bit address spaces)"

OUT=$(render_files --role device --node "$NODE")
trap 'rm -rf "$OUT"' EXIT

remote "$HOST" "mkdir -p $ED_ROOT/provision $ED_CODE_DIR $ED_MODELS_DIR $ED_ROOT/results $ED_ROOT/data $ED_ROOT/run"
RSYNC_SSH="ssh $(ssh_options)"
run rsync -az --delete -e "$RSYNC_SSH" \
    --exclude .git --exclude .venv --exclude results --exclude mirror --exclude __pycache__ \
    --exclude .local_testbed --exclude config.yaml --exclude nodes.json \
    "$CODE_DIR/" "$SSH_USERNAME@$HOST:$ED_CODE_DIR/"
run rsync -az --delete -e "$RSYNC_SSH" "$OUT/" "$SSH_USERNAME@$HOST:$ED_ROOT/provision/rendered/"

# Paths come from config.yaml and carry no spaces, so the remote commands stay unquoted
# and the dry-run plan stays readable.
SETUP="bash $ED_CODE_DIR/scripts/provision/device_setup.sh --rendered $ED_ROOT/provision/rendered --node $NODE --code $ED_CODE_DIR --venv $ED_VENV_DIR --locks $LOCKS --mirror $MIRROR_URL"
if [ -n "$HEADSPLIT_COMMIT" ]; then
    SETUP="$SETUP --headsplit-dir $HEADSPLIT_DIR --headsplit-commit $HEADSPLIT_COMMIT"
fi

remote "$HOST" "$SETUP hostname"
[ "$SKIP_APT" = 1 ] || remote "$HOST" "$SETUP apt"
remote "$HOST" "$SETUP chrony sudoers pip venv"
[ -z "$HEADSPLIT_COMMIT" ] || remote "$HOST" "$SETUP headsplit"
if [ "$SKIP_MODELS" != 1 ] && [ -d "$BS_MODELS_DIR" ]; then
    run rsync -az -e "$RSYNC_SSH" "$BS_MODELS_DIR/" "$SSH_USERNAME@$HOST:$ED_MODELS_DIR/"
    remote "$HOST" "cd $ED_MODELS_DIR && if [ -f SHA256SUMS ]; then sha256sum -c --quiet SHA256SUMS; fi"
fi
remote "$HOST" "$SETUP verify"
# Last: the static addresses may move the device away from the address used so far.
remote "$HOST" "$SETUP network"
log "$NODE provisioned; management address $DEVICE_MGMT_IP, data-plane address $DEVICE_DATA_IP"
timing_log "provision_device $NODE"
