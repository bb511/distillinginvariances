#!/bin/bash
# Run the blocks of a scripts/{mnist,jets}/run*_{train,eval}.sh catalogue.
#
# The selected blocks are queued behind a single nohup, so one GPU runs one
# block at a time; several invocations with --gpu 0..3 and disjoint --only
# selections fill a multi-GPU host. --folds fans a block out over the folds,
# which is one queue entry each: the run directory follows from the fold through
# the config, so nothing but the fold= override changes.
#
# Usage (from the repository root):
#   bash scripts/cluster/submit.sh --list    scripts/jets/rundeepsets_train.sh
#   bash scripts/cluster/submit.sh --dry-run scripts/jets/rundeepsets_train.sh
#   bash scripts/cluster/submit.sh --only kd --gpu 1 scripts/jets/rundeepsets_train.sh
#   bash scripts/cluster/submit.sh --only 4 --folds 2-5 --gpu 1 scripts/jets/rundeepsets_train.sh
#   bash scripts/cluster/submit.sh --only 1 --device MIG-<uuid> --cpus 4-7 scripts/mnist/runcnn_train.sh
#
#   --list            list the blocks in the catalogue and exit
#   --dry-run         print the full commands without running anything; works off
#                     the training machine, so it needs neither the env nor the data
#   --only <sel>      comma-separated label substrings ('transformed', 'method=kd'
#                     is not a label, 'kd_fold1' is) or 1-based indices
#   --gpu <i>         rewrite trainer.devices to this GPU
#   --device <d>      export CUDA_VISIBLE_DEVICES=<d> (a UUID or an index) in the
#                     queue and leave trainer.devices=[0]; the alternative to --gpu
#                     on a pod, where CUDA sees one MIG slice per parent card, so
#                     trainer.devices cannot reach the others
#   --cpus <ids>      launch the queue under taskset -c <ids>
#   --share <j/k>     keep the j-th (0-based) of every k selected commands, counted
#                     over blocks then folds: how submit_ngt.sh deals the commands
#                     round-robin over its devices
#   --folds <spec>    '5' for folds 1..5, '2-5' for a range, '1,3' for a list;
#                     default is the fold the block itself carries
#   --hf-cache <p>    value for HF_HUB_CACHE; default is the one .env exports
#
# Extra hydra overrides after the file name are appended to every command; hydra
# takes the last of a duplicated override, so they win over the file's value.
set -eu

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

only=""
dry=0
list=0
gpu=""
device=""
cpus=""
share=""
folds=""
hf_cache=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --list) list=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --gpu) gpu="${2:?--gpu needs an index}"; shift 2 ;;
        --device) device="${2:?--device needs a UUID or an index}"; shift 2 ;;
        --cpus) cpus="${2:?--cpus needs a list of ids}"; shift 2 ;;
        --share) share="${2:?--share needs j/k}"; shift 2 ;;
        --folds) folds="${2:?--folds needs a spec}"; shift 2 ;;
        --hf-cache) hf_cache="${2:?--hf-cache needs a path}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit.sh [--list|--dry-run] [--only sel] [--gpu i | --device d] [--cpus ids]"
usage="$usage [--share j/k] [--folds spec] [--hf-cache p] <run*_{train,eval}.sh> [overrides...]"
file=${1:?$usage}
shift || true
extra="$*"

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }
if [ -n "$gpu" ] && [ -n "$device" ]; then
    echo "--gpu and --device are alternatives: pass one" >&2
    exit 2
fi
if [ -n "$share" ]; then
    [[ "$share" =~ ^([0-9]+)/([1-9][0-9]*)$ ]] || { echo "bad --share '$share', want j/k" >&2; exit 2; }
    [ "${BASH_REMATCH[1]}" -lt "${BASH_REMATCH[2]}" ] || { echo "--share $share: j must be below k" >&2; exit 2; }
fi

cmds=$(extract_cmds "$file")
n_total=$(printf "%s\n" "$cmds" | grep -c . || true)
if [ "$n_total" -eq 0 ]; then
    echo "No commands found in $file" >&2
    exit 1
fi

if [ "$list" -eq 1 ]; then
    printf "%-3s %-46s %s\n" "#" "block" "sweep"
    i=0
    while IFS= read -r cmd; do
        i=$((i + 1))
        printf "%-3s %-46s %s\n" "$i" "$(block_label "$cmd")" \
            "$(printf '%s' "$cmd" | grep -oE ' -m( |$)' >/dev/null && echo multirun || echo single)"
    done <<EOF
$cmds
EOF
    exit 0
fi

load_env
[ -z "$hf_cache" ] || export HF_HUB_CACHE="$hf_cache"
[ "$dry" -eq 1 ] || require_hf_cache

# PYTHON=/path/to/env/bin/python3 skips activation; --dry-run needs no env.
PYTHON=${PYTHON:-python3}
[ "$dry" -eq 1 ] || require_imports "$PYTHON" "hydra, lightning" \
    "conda activate distill_symm" \
    "(or re-run with PYTHON=/path/to/env/bin/python3)"
# A pod's cpuset is neither 0-based nor contiguous; a bad id list would otherwise
# surface only as a one-line queue log.
if [ "$dry" -eq 0 ] && [ -n "$cpus" ]; then
    taskset -c "$cpus" true || { echo "error: cannot pin to CPUs '$cpus'" >&2; exit 1; }
fi

stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/submit/$stem"
run_ts=$(date +%Y-%m-%d_%H-%M-%S)
tag=${gpu:+_gpu$gpu}
[ -z "$device" ] || tag="_dev${device//[^[:alnum:]-]/_}"
queue="$logdir/queue_${run_ts}${tag}.sh"
pin=${cpus:+taskset -c $cpus }
if [ "$dry" -eq 0 ]; then
    mkdir -p "$logdir"
    start_queue "$queue" "$file"
    [ -z "$device" ] || printf 'export CUDA_VISIBLE_DEVICES=%s\n' "$device" >> "$queue"
elif [ -n "$device$cpus" ]; then
    echo "queue: ${device:+CUDA_VISIBLE_DEVICES=$device }${pin}nohup bash $queue"
fi

# 'keep' leaves a block at the fold its own command carries.
fold_list=$(expand_folds "$folds")
[ -n "$fold_list" ] || fold_list="keep"

i=0
n_sel=0
n_sub=0
while IFS= read -r cmd; do
    i=$((i + 1))
    selected "$i" "$(block_label "$cmd")" || continue
    for fold in $fold_list; do
        n_sel=$((n_sel + 1))
        in_share "$n_sel" || continue
        run_cmd="$cmd"
        [ "$fold" = "keep" ] || run_cmd=$(substitute_fold "$cmd" "$fold")
        [ -z "$gpu" ] || run_cmd=$(pin_gpu "$run_cmd" "$gpu")
        label=$(block_label "$run_cmd")
        full="$PYTHON $run_cmd${extra:+ $extra}"
        n_sub=$((n_sub + 1))
        if [ "$dry" -eq 1 ]; then
            echo "[$i/$n_total] $full"
            continue
        fi
        printf 'echo "[%d] %s"\n%s || echo "FAILED: %s" >&2\n' \
            "$n_sub" "$label" "$full" "$label" >> "$queue"
    done
done <<EOF
$cmds
EOF

if [ "$n_sub" -eq 0 ]; then
    echo "nothing matched --only '$only' in $file" >&2
    exit 1
fi

if [ "$dry" -eq 1 ]; then
    echo
    echo "$n_sub command(s) from $n_total block(s) would run."
else
    log="${queue%.sh}.log"
    # shellcheck disable=SC2086  # $pin is 'taskset -c <ids> ' or nothing, split on purpose
    $pin nohup bash "$queue" > "$log" 2>&1 &
    echo
    echo "$n_sub command(s) from $n_total block(s) queued on ${device:-GPU ${gpu:-0}}, running detached."
    echo "Queue:  $queue"
    echo "Log:    $log"
fi
