#!/usr/bin/env bash
#
# Builds the versioned local mirror on the edge server (mirror_dir in config.yaml):
#   wheels            Python wheels for every lock file (x86_64 and aarch64)
#   apt <node-id>     Debian packages for the devices, fetched through one reference
#                     device (the archives are not reachable from an x86_64 host)
#   headsplit         bare clone of HeadSplit at headsplit_commit plus its wheels
#                     (aarch64 for the devices, x86_64 for the profile step on the edge server)
#   models            every entry of the model catalogue (bs_models_dir/models.yaml) fetched,
#                     checksummed and copied into mirror/models/
#   manifest          MANIFEST.json with the SHA-256 of every file
#   all <node-id>     the five steps above
#
# Usage: build_mirror.sh <wheels|apt NODE|headsplit|models|manifest|all NODE> [--dry-run]
#
# This is the only provisioning script that contacts upstream sources (PyPI, the
# PyTorch index, the Debian and Raspberry Pi archives, the HeadSplit repository,
# the Keras and Hugging Face weight servers). Every other script installs from the mirror.
#
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

take_common_flags "$@"
[ "${#ARGS[@]}" -ge 1 ] || {
    sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
}
COMMAND=${ARGS[0]}
NODE=${ARGS[1]:-}

load_variables

TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
PIP_COMMON=(--only-binary=:all: --python-version 3.11 --implementation cp
    --abi cp311 --abi abi3 --abi none)
PLATFORMS_X86=(--platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64
    --platform manylinux_2_17_x86_64 --platform manylinux1_x86_64 --platform any)
PLATFORMS_ARM=(--platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64
    --platform manylinux_2_17_aarch64 --platform any)

download() { # dest lock platforms...
    local dest=$1 lock=$2
    shift 2
    run mkdir -p "$dest"
    run "$PYTHON" -m pip download --require-hashes "${PIP_COMMON[@]}" "$@" \
        --extra-index-url "$TORCH_CPU_INDEX" -d "$dest" -r "$CODE_DIR/requirements/$lock"
}

wheels() {
    download "$MIRROR_DIR/wheels/x86_64" edge_server.lock "${PLATFORMS_X86[@]}"
    download "$MIRROR_DIR/wheels/x86_64" dev.lock "${PLATFORMS_X86[@]}"
    download "$MIRROR_DIR/wheels/aarch64" edge_device.lock "${PLATFORMS_ARM[@]}"
    download "$MIRROR_DIR/wheels/aarch64" edge_device_llm.lock "${PLATFORMS_ARM[@]}"
}

apt_packages() {
    [ -n "$NODE" ] || die "apt needs a reference node id"
    load_variables --node "$NODE"
    local host=$DEVICE_MGMT_IP packages
    packages=$(package_list "$PROVISION_DIR/apt_packages_device.txt")
    run mkdir -p "$MIRROR_DIR/apt"
    # The device downloads from the upstream archives once; the files then live in the mirror.
    remote "$host" "sudo apt-get update"
    remote "$host" "sudo apt-get install --download-only --reinstall -y $packages"
    run rsync -a -e "ssh $(ssh_options)" \
        "$SSH_USERNAME@$host:/var/cache/apt/archives/*.deb" "$MIRROR_DIR/apt/"
    run bash -c "cd $MIRROR_DIR/apt && dpkg-scanpackages --multiversion . /dev/null | gzip -9c > Packages.gz"
}

headsplit() {
    if [ -z "$HEADSPLIT_COMMIT" ]; then
        log "headsplit_commit is null: HeadSplit is not mirrored"
        return 0
    fi
    local bare="$MIRROR_DIR/src/headsplit.git" reqs
    run mkdir -p "$MIRROR_DIR/src" "$MIRROR_DIR/wheels/aarch64-headsplit"
    if [ -d "$bare" ]; then
        run git --git-dir="$bare" fetch --quiet origin
    else
        run git clone --quiet --bare "$HEADSPLIT_REPO_URL" "$bare"
    fi
    run git --git-dir="$bare" update-server-info
    reqs=$(mktemp)
    run bash -c "git --git-dir=$bare show $HEADSPLIT_COMMIT:requirements.txt > $reqs"
    run "$PYTHON" -m pip download "${PIP_COMMON[@]}" "${PLATFORMS_ARM[@]}" \
        --extra-index-url "$TORCH_CPU_INDEX" -d "$MIRROR_DIR/wheels/aarch64-headsplit" -r "$reqs"
    run mkdir -p "$MIRROR_DIR/wheels/x86_64-headsplit"
    run "$PYTHON" -m pip download "${PIP_COMMON[@]}" "${PLATFORMS_X86[@]}" \
        --extra-index-url "$TORCH_CPU_INDEX" -d "$MIRROR_DIR/wheels/x86_64-headsplit" -r "$reqs"
    [ "$DRY_RUN" = 1 ] || rm -f "$reqs"
}

models() {
    if [ ! -f "$BS_MODELS_DIR/models.yaml" ]; then
        log "no $BS_MODELS_DIR/models.yaml: copy examples/models.yaml there and edit it"
        return 0
    fi
    run "$PYTHON" -m src.models.weights fetch --all --allow-upstream --models-dir "$BS_MODELS_DIR"
    run mkdir -p "$MIRROR_DIR/models"
    run rsync -a --delete "$BS_MODELS_DIR/" "$MIRROR_DIR/models/"
}

manifest() {
    run "$PYTHON" -m src.provisioning.mirror manifest --mirror "$MIRROR_DIR"
}

case "$COMMAND" in
    wheels) wheels ;;
    apt) apt_packages ;;
    headsplit) headsplit ;;
    models) models ;;
    manifest) manifest ;;
    all)
        wheels
        apt_packages
        headsplit
        models
        manifest
        ;;
    *) die "unknown command $COMMAND" ;;
esac
timing_log "build_mirror $COMMAND"
