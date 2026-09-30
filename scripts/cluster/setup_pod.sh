#!/usr/bin/env bash
# Set up the checkout of this repository in a kd-session pod. Run it inside the pod, entered
# from lxplus with ~/ngt/kssh-adl1t kd-session-<n>, from any copy of the repository the pod
# can read; a rerun leaves the links and an existing .env as they are.
#
#   bash scripts/cluster/setup_pod.sh <git url>   # clone once; check out the branch afterwards
#   bash scripts/cluster/setup_pod.sh <dir>       # rsync a checkout (on /afs or /shared, or
#                                                 # host:path); rerun to update the code
#   bash scripts/cluster/setup_pod.sh             # links and .env only
#
# The checkout lives on the pod's local /scratch. Its data, logs, outputs and results are
# symlinks onto /shared, so every kd pod reads the same caches and writes one set of logs and
# one results.csv, and a checkpoint path a manifest records on one pod resolves on every other.
set -euo pipefail

repo=/scratch/distilling_symmetries
shared=/shared/deodagiu/distilling_symmetries

fetch() {
    case $1 in
        git@* | https://* | *.git) [ -d "$repo/.git" ] || git clone "$1" "$repo" ;;
        # The run directories are symlinks here and real directories in the source: never copy them.
        *)
            mkdir -p "$repo"
            rsync -a --exclude .git --exclude __pycache__ --exclude .env --exclude /data \
                --exclude /logs --exclude /outputs --exclude /results "$1/" "$repo/"
            ;;
    esac
}

link() {
    mkdir -p "$shared/$1"
    if [ -e "$repo/$1" ] && [ ! -L "$repo/$1" ]; then
        echo "$repo/$1 is a real directory; move its contents to $shared/$1 and rerun" >&2
        exit 1
    fi
    ln -sfn "$shared/$1" "$repo/$1"
}

# The keys scripts/setup.sh writes that are needed here, plus OMP_NUM_THREADS, which keeps each
# driver from spawning one thread per core of the pod. HF_HUB_CACHE is the hub cache assumed to
# be shared by the pods; edit .env to point it elsewhere.
write_env() {
    [ ! -e "$repo/.env" ] || return 0
    cat > "$repo/.env" << 'ENV'
PROJECT_ROOT="."
HF_HUB_CACHE="/shared/deodagiu/hf_cache"
MLFLOW_ALLOW_FILE_STORE="true"
OMP_NUM_THREADS="4"
ENV
}

[ $# -eq 0 ] || fetch "$1"
if [ ! -f "$repo/.project-root" ]; then
    echo "no checkout at $repo; pass a git url or a directory" >&2
    exit 1
fi
for d in data logs outputs results; do link "$d"; done
mkdir -p /shared/deodagiu/hf_cache
write_env
echo "ready: cd $repo"
