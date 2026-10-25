# shellcheck shell=bash
#
# Shared helpers for the provisioning scripts. Source it; do not run it.
#
# Conventions:
#   * DRY_RUN=1 (or --dry-run) prints every state-changing command, one per line
#     and shell-quoted, instead of running it. Read-only queries still run.
#   * Addresses, users and paths come from config.yaml and nodes.json through
#     src/provisioning/render.py; nothing of that kind is written in the scripts.
#   * Every step is idempotent: files are installed only when their content
#     differs, environments are created only when absent.

set -euo pipefail

PROVISION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$PROVISION_DIR/../.." && pwd)"
PYTHON="${TESTBED_PYTHON:-$CODE_DIR/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
DRY_RUN="${DRY_RUN:-0}"
START_TIME=$(date +%s)

log() { printf '[provision] %s\n' "$*" >&2; }
die() { printf '[provision] error: %s\n' "$*" >&2; exit 1; }

# Takes --dry-run out of the arguments; the rest stays in ARGS.
ARGS=()
take_common_flags() {
    ARGS=()
    for arg in "$@"; do
        case "$arg" in
            --dry-run) DRY_RUN=1 ;;
            *) ARGS+=("$arg") ;;
        esac
    done
}

# Quotes an argument for display only when it needs quoting.
quote_arg() {
    local arg=$1 q="'\\''"
    if [[ "$arg" =~ ^[A-Za-z0-9_./:=@%+,*-]+$ ]]; then
        printf '%s' "$arg"
    else
        printf "'%s'" "${arg//\'/$q}"
    fi
}

# Prints the command when dry running, runs it otherwise.
run() {
    if [ "$DRY_RUN" = 1 ]; then
        local out="" arg
        for arg in "$@"; do
            out+="$(quote_arg "$arg") "
        done
        printf '%s\n' "${out% }"
    else
        "$@"
    fi
}

ssh_options() {
    printf -- '-o BatchMode=yes -o ConnectTimeout=%s' "${SSH_CONNECT_TIMEOUT:-10}"
    if [ -n "${SSH_KNOWN_HOSTS:-}" ]; then
        printf -- ' -o UserKnownHostsFile=%s' "$SSH_KNOWN_HOSTS"
    fi
}

# Runs a command on a device over SSH (management address). State-changing.
remote() {
    local host=$1
    shift
    # shellcheck disable=SC2046
    run ssh $(ssh_options) "$SSH_USERNAME@$host" "$@"
}

# Read-only query on a device; runs even in dry-run mode.
remote_query() {
    local host=$1
    shift
    # shellcheck disable=SC2046,SC2029
    ssh $(ssh_options) "$SSH_USERNAME@$host" "$@"
}

# Installs a file only when its content differs (root-owned destination).
install_file() { # src dst mode
    local src=$1 dst=$2 mode=$3
    if [ "$DRY_RUN" != 1 ] && [ -f "$dst" ] && cmp -s "$src" "$dst"; then
        log "unchanged: $dst"
        return 0
    fi
    run sudo install -D -m "$mode" "$src" "$dst"
}

# Loads the rendered configuration variables (exports) for this host or a node.
# shellcheck disable=SC2120
load_variables() { # [--node ID]
    local exports role=()
    [ "$#" -eq 0 ] || role=(--role device)
    exports=$(cd "$CODE_DIR" && "$PYTHON" -m src.provisioning.render --vars "${role[@]+"${role[@]}"}" "$@") ||
        die "could not read config.yaml and nodes.json (copy the example files first)"
    eval "$exports"
}

# Renders the files of a role into a fresh temporary directory (printed).
render_files() { # --role R [--node ID] [--var K=V]
    local out
    out=$(mktemp -d)
    (cd "$CODE_DIR" && "$PYTHON" -m src.provisioning.render --out "$out" "$@" >/dev/null) ||
        die "rendering failed"
    printf '%s\n' "$out"
}

# Package names from a list file (comments and blank lines removed).
package_list() { # file
    [ -f "$1" ] || die "package list $1 not found"
    grep -v '^\s*#' "$1" | grep -v '^\s*$' | tr '\n' ' '
}

# Records how long the script took (Chapter 3 quotes 25 min per device, 2 h in all).
timing_log() { # name
    local end
    end=$(date +%s)
    log "$1 finished in $((end - START_TIME)) s"
    if [ "$DRY_RUN" != 1 ] && [ -n "${BS_RESULTS_DIR:-}" ]; then
        mkdir -p "$BS_RESULTS_DIR/provisioning"
        printf '%s %s %s s\n' "$(date -u +%FT%TZ)" "$1" "$((end - START_TIME))" \
            >>"$BS_RESULTS_DIR/provisioning/timing.log"
    fi
}
