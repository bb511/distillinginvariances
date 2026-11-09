#!/bin/bash
# Sweep one Optuna study with several concurrent workers on an NGT pod.
#
# The optuna sweeper opens its study with load_if_exists=True, so drivers that
# agree on the study name and the storage share one study: their trials
# accumulate and n_startup_trials is read from the study's own count, which is
# what makes a pool of short drivers equivalent to one long one. Two things must
# differ per worker, and supplying them is all this script does: the sampler
# seed, without which every worker draws the same random start-up points, and a
# tag, because logs/ on the pods is one volume shared by all of them.
#
# The pod is the machine; there is no scheduler. NVIDIA_VISIBLE_DEVICES lists one
# UUID per MIG slice, or a single 'GPU-...' on a full-card pod, and CUDA
# enumerates only one slice per parent card, so trainer.devices=[i] cannot reach
# the others. A worker is pinned with CUDA_VISIBLE_DEVICES=<uuid> instead and
# keeps the catalogue's trainer.devices=[0].
#
# Usage (from the repository root, inside an NGT session):
#
#   # four workers on one full card, 200 trials shared out between them
#   bash scripts/cluster/submit_search_ngt.sh --only pt_etarel_phirel --workers 4 \
#       --trials 200 --seed 1234 --tag s1 scripts/jets/runstudent_search.sh
#
#   --only <sel>    which block of the catalogue to sweep, passed through to
#                   submit_search.sh; required, since a pool sweeps one study
#   --gpus <list>   comma-separated indices into NVIDIA_VISIBLE_DEVICES; default
#                   is every device it lists, in order
#   --workers <k>   drivers to start (default: one per chosen device). More
#                   workers than devices take them round-robin, which is how a
#                   full card is time-shared
#   --trials <T>    trials in total, split into k near-equal parts; required,
#                   because hydra.sweeper.n_trials is per driver and accumulates
#   --seed <s>      sampler seed of worker 0; worker j gets s+j. Required: it is
#                   the one setting the workers must not share
#   --tag <t>       names the logs and the sweep directories (default: hostname)
#   --cpus <list>   the CPU ids the workers share, as ids and a-b ranges separated
#                   by commas: --cpus "$(cpus)" on a pod, where 'cpus' is an alias
#                   of the interactive shell that this script cannot see. Default
#                   is the output of a 'cpus' executable on PATH, else of
#                   scripts/cluster/cpus.sh; with neither, the workers run unpinned
#   --dry-run       print each worker's command, launch nothing
#
# Anything after the catalogue file is appended to every worker as an override.
#
# A process tree started under 'kubectl exec' dies with the exec session, so each
# worker is launched with setsid nohup and outlives the disconnection. Each is
# pinned with taskset to its own contiguous share of the pod's CPUs: the pods'
# cpusets are neither 0-based nor contiguous, so the ids are handed over
# explicitly, where a naive 'taskset -c 0-2' dies with "failed to set affinity".
set -eu

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Workers of one study race to create their study row in the same sqlite file.
STAGGER_SECONDS=3

only=""
gpus=""
workers=""
trials=""
seed=""
tag=""
cpus=""
dry=0

while [ $# -gt 0 ]; do
    case "$1" in
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --gpus) gpus="${2:?--gpus needs a list}"; shift 2 ;;
        --workers) workers="${2:?--workers needs a number}"; shift 2 ;;
        --trials) trials="${2:?--trials needs a number}"; shift 2 ;;
        --seed) seed="${2:?--seed needs a number}"; shift 2 ;;
        --tag) tag="${2:?--tag needs a value}"; shift 2 ;;
        --cpus) cpus="${2:?--cpus needs a list}"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit_search_ngt.sh --only <sel> --trials <T> --seed <s> [--gpus i,j]"
usage="$usage [--workers k] [--tag t] [--cpus list] [--dry-run] <run*_search.sh> [overrides...]"
file=${1:?$usage}
shift || true

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }
[ -n "$only" ] || { echo "--only is required: a worker pool sweeps one study" >&2; exit 2; }
[ -n "$trials" ] || { echo "--trials is required: it is split between the workers" >&2; exit 2; }
[ -n "$seed" ] || { echo "--seed is required: one seed for all is one search" >&2; exit 2; }
case "$trials$seed${workers:-0}" in
    *[!0-9]*) echo "--trials, --seed and --workers take non-negative integers" >&2; exit 2 ;;
esac
[ -n "$tag" ] || tag=$(hostname)

devices=${NVIDIA_VISIBLE_DEVICES:-}
if [ -z "$devices" ]; then
    if [ "$dry" -eq 0 ]; then
        echo "error: NVIDIA_VISIBLE_DEVICES is empty - this is not an NGT pod." >&2
        exit 1
    fi
    # A dry run off the pod still shows the shape of the launch.
    n=${workers:-1}
    if [ "$n" -lt 1 ]; then n=1; fi
    for i in $(seq 0 $((n - 1))); do devices="${devices:+$devices,}<gpu$i>"; done
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
n_devices=${#chosen[@]}
[ -n "$workers" ] || workers=$n_devices
if [ "$workers" -lt 1 ]; then echo "--workers must be at least 1" >&2; exit 2; fi
if [ "$workers" -gt "$trials" ]; then
    echo "--workers $workers exceeds --trials $trials: a worker without a trial exits at once" >&2
    exit 2
fi

# One id per line from a list of ids and a-b ranges, refusing anything else, so
# a 'cpus' that prints something unexpected stops the launch rather than pinning
# every worker to CPU 0.
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
    if [ "${#ids[@]}" -lt "$workers" ]; then
        echo "--workers $workers exceeds the ${#ids[@]} CPU id(s) in '$cpus', one per worker at least" >&2
        exit 2
    fi
else
    echo "note: no CPU ids from --cpus, 'cpus' or scripts/cluster/cpus.sh, so the workers are not pinned" >&2
fi

# Part $3 (0-based) of $1 items cut into $2 contiguous parts, as '<first> <count>'.
# The first $1 mod $2 parts carry one item more, so no two parts differ by more
# than one item.
share_of() {
    local per=$(($1 / $2)) rem=$(($1 % $2))
    if [ "$3" -lt "$rem" ]; then
        echo "$(($3 * (per + 1))) $((per + 1))"
    else
        echo "$((rem * (per + 1) + ($3 - rem) * per)) $per"
    fi
}

# Worker $1's share of --trials.
trials_of() {
    local count
    read -r _ count <<< "$(share_of "$trials" "$workers" "$1")"
    echo "$count"
}

# Worker $1's contiguous share of the CPU ids as 'taskset -c <ids> ', empty when
# nothing is pinned.
pin_of() {
    local n=${#ids[@]} first count
    [ "$n" -gt 0 ] || return 0
    read -r first count <<< "$(share_of "$n" "$workers" "$1")"
    local IFS=','
    printf 'taskset -c %s ' "${ids[*]:$first:$count}"
}

split=""
for j in $(seq 0 $((workers - 1))); do split="${split:+$split/}$(trials_of "$j")"; done

logdir="logs/searches/ngt"
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

echo "tag=$tag  workers=$workers  devices=[$(IFS=,; printf '%s' "${chosen[*]}")]" \
    "trials=$trials ($split)  seeds=$seed..$((seed + workers - 1))"
echo

for j in $(seq 0 $((workers - 1))); do
    device=${chosen[$((j % n_devices))]}
    worker_tag="${tag}_w${j}"
    log="$logdir/${worker_tag}.log"
    worker=(scripts/cluster/submit_search.sh --only "$only" --tag "$worker_tag" "$file" "$@"
        "hydra.sweeper.n_trials=$(trials_of "$j")" "hydra.sweeper.sampler.seed=$((seed + j))")
    pin=$(pin_of "$j")
    printf 'CUDA_VISIBLE_DEVICES=%s %ssetsid nohup bash %s > %s 2>&1 &\n' \
        "$device" "$pin" "${worker[*]}" "$log"
    if [ "$dry" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES="$device" bash "${worker[0]}" --dry-run "${worker[@]:1}"
        continue
    fi
    # shellcheck disable=SC2086  # $pin is 'taskset -c <ids> ' or nothing, split on purpose
    CUDA_VISIBLE_DEVICES="$device" $pin setsid nohup bash "${worker[@]}" > "$log" 2>&1 &
    sleep "$STAGGER_SECONDS"
done

echo
if [ "$dry" -eq 1 ]; then
    echo "$workers worker(s) would start; nothing was launched."
    exit 0
fi
echo "$workers worker(s) running detached (launch logs: $logdir), safe to leave the pod."
echo "Progress: python3 scripts/search/sweep_status.py"
