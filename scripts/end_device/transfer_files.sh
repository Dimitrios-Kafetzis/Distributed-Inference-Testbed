#!/usr/bin/env bash
#
# Copies result files from this device to the edge server with scp, prepending
# the device hostname to each filename to prevent collisions during aggregation.
#
# Usage: transfer_files.sh <username> <target_ip> <target_directory> <file1> [file2 ...]
#
# Authentication: an SSH key by default (TESTBED_SSH_IDENTITY names the key
# file when it is not the default one). When the environment variable SSHPASS
# is set, sshpass password authentication is used instead; the password never
# appears on the command line.
#
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <username> <target_ip> <target_directory> <file1> [file2 ...]" >&2
    exit 1
fi

USERNAME=$1
TARGET_IP=$2
TARGET_DIR=$3
shift 3

HOST=$(hostname)
PORT="${TESTBED_SSH_PORT:-22}"
SCP_OPTS=(-q -P "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
if [ -n "${TESTBED_SSH_IDENTITY:-}" ]; then
    SCP_OPTS+=(-i "$TESTBED_SSH_IDENTITY")
fi

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

status=0
for FILE in "$@"; do
    if [ ! -f "$FILE" ]; then
        echo "transfer_files: missing $FILE" >&2
        status=1
        continue
    fi
    NAME="${HOST}_$(basename "$FILE")"
    cp "$FILE" "$STAGING/$NAME"
    if [ -n "${SSHPASS:-}" ]; then
        sshpass -e scp "${SCP_OPTS[@]}" -o BatchMode=no "$STAGING/$NAME" "${USERNAME}@${TARGET_IP}:${TARGET_DIR}/"
    else
        scp "${SCP_OPTS[@]}" "$STAGING/$NAME" "${USERNAME}@${TARGET_IP}:${TARGET_DIR}/"
    fi
    echo "transferred $FILE as $NAME"
done
exit $status
