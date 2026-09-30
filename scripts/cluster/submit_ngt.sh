#!/bin/bash
# Run the blocks of a catalogue on an NGT pod, one submit.sh queue per GPU device.
#
# The selected (block x fold) commands, counted over blocks then folds, are dealt
# round-robin over the chosen devices, so on a pod with as many slices as folds
# fold k of a block runs on slice k. A multirun block (-m) is one command: its
# points run one after another on their device.
#
# The pod is the machine; there is no scheduler. NVIDIA_VISIBLE_DEVICES lists one
# UUID per MIG slice, or a single 'GPU-...' on a full-card pod, and CUDA
# enumerates only one slice per parent card, so trainer.devices=[i] cannot reach
# the others. Each queue is pinned with CUDA_VISIBLE_DEVICES=<uuid> instead
# (submit.sh --device) and keeps the catalogue's trainer.devices=[0].
#
# Usage (from the repository root, inside an NGT session):
#   bash scripts/cluster/submit_ngt.sh --dry-run --only teacher_fold1 --folds 1-5 \
#       scripts/jets/rundeepsets_train.sh
#   bash scripts/cluster/submit_ngt.sh --only teacher_fold1 --folds 1-5 \
#       scripts/jets/rundeepsets_train.sh
#
#   --only <sel>    passed to submit.sh: label substrings or 1-based indices
#   --folds <spec>  passed to submit.sh: '5', '2-5' or '1,3'
#   --gpus <list>   comma-separated indices into NVIDIA_VISIBLE_DEVICES; default
#                   is every device it lists, in order
#   --cpus <list>   the CPU ids the queues share, as ids and a-b ranges separated
#                   by commas: --cpus "$(cpus)" on a pod, where 'cpus' is an alias
#                   of the interactive shell that this script cannot see. Default
#                   is the output of a 'cpus' executable on PATH, else of
#                   scripts/cluster/cpus.sh; with neither, the queues run unpinned
#   --dry-run       print the plan and every command, launch nothing; off a pod the
#                   devices are placeholders, one per command
#
# Anything after the catalogue file is appended to every command as an override.
#
# A process tree started under 'kubectl exec' dies with the exec session, so each
# submit.sh runs under setsid and its nohup'd queue lands in a session of its own.
# Each queue is pinned with taskset to its own contiguous share of the pod's CPUs:
# the pods' cpusets are neither 0-based nor contiguous, so the ids are handed over
# explicitly, where a naive 'taskset -c 0-2' dies with "failed to set affinity".
set -euo pipefail

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

only=""
folds=""
gpus=""
cpus=""
dry=0

while [ $# -gt 0 ]; do
    case "$1" in
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --folds) folds="${2:?--folds needs a spec}"; shift 2 ;;
        --gpus) gpus="${2:?--gpus needs a list}"; shift 2 ;;
        --cpus) cpus="${2:?--cpus needs a list}"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit_ngt.sh [--only sel] [--folds spec] [--gpus i,j] [--cpus list]"
usage="$usage [--dry-run] <run*_{train,eval}.sh> [overrides...]"
file=${1:?$usage}
shift || true

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }

# The number of (block x fold) commands submit.sh will select, to leave no
# device with an empty share.
count_cmds() {
    local cmd i=0 blocks=0 n_folds
    while IFS= read -r cmd; do
        i=$((i + 1))
        if selected "$i" "$(block_label "$cmd")"; then blocks=$((blocks + 1)); fi
    done < <(extract_cmds "$file")
    n_folds=$(expand_folds "$folds" | grep -c . || true)
    echo $((blocks * (n_folds > 0 ? n_folds : 1)))
}
n_cmds=$(count_cmds)
[ "$n_cmds" -gt 0 ] || { echo "nothing matched --only '$only' in $file" >&2; exit 1; }

devices=${NVIDIA_VISIBLE_DEVICES:-}
if [ -z "$devices" ]; then
    if [ "$dry" -eq 0 ]; then
        echo "error: NVIDIA_VISIBLE_DEVICES is empty - this is not an NGT pod." >&2
        exit 1
    fi
    for i in $(seq 0 $((n_cmds - 1))); do devices="${devices:+$devices,}<gpu$i>"; done
fi

IFS=',' read -r -a all_devices <<< "$devices"
chosen=("${all_devices[@]}")
if [ -n "$gpus" ]; then
    chosen=()
    IFS=',' read -r -a wanted <<< "$gpus"
    for i in "${wanted[@]}"; do
        if [[ ! "$i" =~ ^[0-9]+$ ]] || [ "$i" -ge "${#all_devices[@]}" ]; then
            echo "bad --gpus index '$i': NVIDIA_VISIBLE_DEVICES lists ${#all_devices[@]}" >&2
            exit 2
        fi
        chosen+=("${all_devices[$i]}")
    done
fi
n_queues=$((n_cmds < ${#chosen[@]} ? n_cmds : ${#chosen[@]}))

# One id per line from a list of ids and a-b ranges, refusing anything else, so
# a 'cpus' that prints something unexpected stops the launch rather than pinning
# every queue to CPU 0.
cpu_ids() {
    local item c
    for item in $(printf '%s' "$1" | tr ',' ' '); do
        case "$item" in
            *-*-* | -* | *- | *[!0-9-]*) echo "bad CPU id '$item' in: $1" >&2; return 1 ;;
            *-*) for ((c = ${item%%-*}; c <= ${item##*-}; c++)); do echo "$c"; done ;;
            *) echo "$item" ;;
        esac
    done
}

if [ -z "$cpus" ] && command -v cpus >/dev/null 2>&1; then cpus=$(cpus); fi
if [ -z "$cpus" ] && [ -f scripts/cluster/cpus.sh ]; then cpus=$(bash scripts/cluster/cpus.sh 2>/dev/null || true); fi
ids=()
if [ -n "$cpus" ]; then
    # Through a substitution, whose status is the producer's; read -d '' rather than
    # mapfile keeps the dry run working under macOS's bash 3.2.
    id_lines=$(cpu_ids "$cpus") || exit 2
    read -r -d '' -a ids <<< "$id_lines" || true
    if [ "${#ids[@]}" -lt "$n_queues" ]; then
        echo "$n_queues queues but only ${#ids[@]} CPU id(s) in '$cpus'" >&2
        exit 2
    fi
else
    echo "note: no CPU ids from --cpus, 'cpus' or scripts/cluster/cpus.sh, so the queues are not pinned" >&2
fi

# Queue j's contiguous share of the CPU ids as a comma list; the first
# (ids mod queues) shares carry one id more. Empty when nothing is pinned.
pin_of() {
    local n=${#ids[@]} per rem first count
    [ "$n" -gt 0 ] || return 0
    per=$((n / n_queues)); rem=$((n % n_queues))
    first=$(($1 * per + ($1 < rem ? $1 : rem))); count=$((per + ($1 < rem ? 1 : 0)))
    local IFS=','
    printf '%s' "${ids[*]:$first:$count}"
}

logdir="logs/submit/ngt"
launch_log="$logdir/$(basename "$(dirname "$file")")_$(basename "$file" .sh)_$(date +%Y-%m-%d_%H-%M-%S).log"
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

common=()
[ -z "$only" ] || common+=(--only "$only")
[ -z "$folds" ] || common+=(--folds "$folds")

echo "$n_cmds command(s) over $n_queues queue(s): devices=[$(IFS=,; printf '%s' "${chosen[*]:0:$n_queues}")]" \
    "cpus=${cpus:-unpinned}"
for j in $(seq 0 $((n_queues - 1))); do
    args=(--share "$j/$n_queues" --device "${chosen[$j]}")
    pin=$(pin_of "$j")
    [ -z "$pin" ] || args+=(--cpus "$pin")
    echo
    if [ "$dry" -eq 1 ]; then
        bash scripts/cluster/submit.sh --dry-run "${args[@]}" ${common[@]+"${common[@]}"} "$file" "$@"
        continue
    fi
    setsid -w bash scripts/cluster/submit.sh "${args[@]}" ${common[@]+"${common[@]}"} "$file" "$@" 2>&1 |
        tee -a "$launch_log"
done

echo
if [ "$dry" -eq 1 ]; then
    echo "$n_queues queue(s) would start; nothing was launched."
    exit 0
fi
echo "$n_queues queue(s) running detached, safe to leave the pod (launch log: $launch_log)."
