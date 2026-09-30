#!/bin/bash
# Replace data/ logs/ outputs/ results/ with symlinks to the locations named in
# .env, for keeping them on a scratch filesystem. Run scripts/setup.sh first and
# edit RES_DIR (or the individual *_DIR variables) there. Run this before the
# first training: the checkpoint manifests under logs/ carry absolute paths, so
# moving logs/ afterwards breaks eval.
#
#   bash scripts/symbolink.sh          # asks before destroying anything
#   bash scripts/symbolink.sh --force  # no questions
#
# DESTRUCTIVE: a real directory in the way is deleted with its contents. With
# RES_DIR left at setup.sh's "." the targets are those directories themselves,
# which this refuses to do.
set -euo pipefail

cd "$(dirname "$0")/.."

force=0
[ "${1:-}" != "--force" ] || force=1

if [ ! -f .env ]; then
    echo "no .env - run 'bash scripts/setup.sh' first" >&2
    exit 1
fi
# shellcheck disable=SC1091  # .env is generated, not checked in
source .env
: "${RES_DIR:?RES_DIR is not set in .env}"

# $1 = link to create in the repository, $2 = directory it should point at.
create_and_link_dir() {
    local dir_name="$1"
    local target_dir="$2"

    if [ "$(cd "$(dirname "$dir_name")" && pwd)/$(basename "$dir_name")" = \
         "$(cd "$(dirname "$target_dir")" 2>/dev/null && pwd)/$(basename "$target_dir")" ]; then
        echo "skipping $dir_name: it is its own target (set RES_DIR in .env)"
        return
    fi

    if [ ! -d "$target_dir" ]; then
        echo "Target directory $target_dir does not exist. Creating it."
        mkdir -p "$target_dir"
    fi

    if [ -L "$dir_name" ]; then
        echo "Found existing symbolic link $dir_name. It will be replaced."
        unlink "$dir_name"
    elif [ -d "$dir_name" ]; then
        if [ "$force" -eq 0 ] && [ -n "$(ls -A "$dir_name")" ]; then
            printf "%s is a non-empty directory. Delete it and its contents? [y/N] " "$dir_name"
            read -r reply
            case "$reply" in [yY]*) ;; *) echo "skipping $dir_name"; return ;; esac
        fi
        echo "Removing existing directory $dir_name."
        rm -rf "$dir_name"
    fi

    ln -s "$target_dir" "$dir_name"
}

create_and_link_dir data    "${DATA_DIR:-$RES_DIR/data}"
create_and_link_dir logs    "${LOG_DIR:-$RES_DIR/logs}"
create_and_link_dir outputs "${OUTPUT_DIR:-$RES_DIR/outputs}"
create_and_link_dir results "${RESULTS_DIR:-$RES_DIR/results}"

echo "Done."
