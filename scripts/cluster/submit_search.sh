#!/bin/bash
# Launch the Optuna drivers of a scripts/{jets,mnist}/run*_search.sh catalogue.
#
# A search block already carries '-m' and its sweeper settings; this adds only
# what the catalogue does not carry: the GPU and a private sweep directory. Each
# selected block becomes one long-lived nohup'd driver, so every selected block
# starts on this host at once: select with --only and pin with --gpu, one
# invocation per GPU.
#
# Usage (from the repository root):
#   bash scripts/cluster/submit_search.sh --list    scripts/jets/runstudent_search.sh
#   bash scripts/cluster/submit_search.sh --dry-run scripts/jets/runstudent_search.sh
#   bash scripts/cluster/submit_search.sh --only pt_eta_phi --gpu 1 scripts/jets/runstudent_search.sh
#
#   --list           list the blocks in the catalogue and exit
#   --dry-run        print the full commands without running anything; it needs
#                    neither the env nor the data
#   --only <sel>     comma-separated label substrings ('pt_eta_phi') or 1-based
#                    indices; default is every block in the file
#   --gpu <i>        rewrite trainer.devices to this GPU
#   --tag <t>        suffix of this driver's log and sweep directory. One study may
#                    be swept by several drivers at once (the sweeper opens its
#                    study with load_if_exists), and on the NGT pods logs/ is one
#                    shared volume, so without a tag per driver the second worker
#                    rotates the first one's live log, and two starting in the
#                    same second share a sweep directory. See submit_search_ngt.sh.
#
# Extra hydra overrides after the file name are appended to every driver; hydra
# takes the last of a duplicated override, so they win over the file's value.
set -eu

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

only=""
dry=0
list=0
gpu=""
tag=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --list) list=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --gpu) gpu="${2:?--gpu needs an index}"; shift 2 ;;
        --tag) tag="${2:?--tag needs a value}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit_search.sh [--list|--dry-run] [--only sel] [--gpu i] [--tag t]"
usage="$usage <run*_search.sh> [overrides...]"
file=${1:?$usage}
shift || true
extra="$*"

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }
cmds=$(extract_cmds "$file")
n_total=$(printf "%s\n" "$cmds" | grep -c . || true)
[ "$n_total" -gt 0 ] || { echo "No search commands found in $file" >&2; exit 1; }

# The block's study, '<experiment>_<regime>': what --only matches on and what
# names the driver's log and sweep directory.
search_label() {
    local exp regime
    exp=$(field "$1" 'experiment')
    regime=$(field "$1" 'data\.regime')
    printf "%s_%s" "${exp##*/}" "${regime:-canonical}"
}

if [ "$list" -eq 1 ]; then
    printf "%-3s %-34s %-26s %-9s %s\n" "#" "block" "hparams_search" "n_trials" "max_epochs"
    i=0
    while IFS= read -r cmd; do
        i=$((i + 1))
        printf "%-3s %-34s %-26s %-9s %s\n" "$i" "$(search_label "$cmd")" \
            "$(field "$cmd" 'hparams_search')" \
            "$(field "$cmd" 'hydra\.sweeper\.n_trials')" \
            "$(field "$cmd" 'trainer\.max_epochs')"
    done <<< "$cmds"
    exit 0
fi

load_env
# PYTHON=/path/to/env/bin/python3 skips activation; --dry-run needs no env.
PYTHON=${PYTHON:-python3}
if [ "$dry" -eq 0 ]; then
    require_hf_cache
    require_imports "$PYTHON" "hydra, optuna, hydra_plugins.hydra_optuna_sweeper" \
        "conda activate distill_symm" "(or re-run with PYTHON=/path/to/env/bin/python3)"
fi

stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/searches/$stem"
sweep_ts=$(date +%Y-%m-%d_%H-%M-%S)
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

# The storage URL in one config file.
storage_in_file() {
    grep -oE "^[[:space:]]*storage:[[:space:]]*\S+" "$1" | head -1 | awk '{print $2}' || true
}

# The block's own storage override, else that of the hparams_search config it
# selects, else that of a config this one inherits.
storage_of() {
    local hp f s d base
    # Quoted twice in a catalogue: hydra's grammar wants the '=' of '?timeout=600'
    # inside quotes of its own, and bash -c strips the outer pair.
    s=$(field "$1" 'hydra\.sweeper\.storage')
    s=${s#\"}; s=${s%\"}; s=${s#\'}; s=${s%\'}
    if [ -n "$s" ]; then printf "%s" "$s"; return 0; fi
    hp=$(field "$1" 'hparams_search')
    f="configs/hparams_search/${hp}.yaml"
    if [ -z "$hp" ] || [ ! -f "$f" ]; then return 0; fi
    s=$(storage_in_file "$f")
    if [ -z "$s" ]; then
        # A defaults entry is group-relative or from the group root; try both.
        while IFS= read -r d; do
            for base in "$(dirname "$f")" "configs/hparams_search"; do
                d=${d#/hparams_search/}
                if [ -f "$base/$d.yaml" ]; then s=$(storage_in_file "$base/$d.yaml"); fi
                if [ -n "$s" ]; then break 2; fi
            done
        done < <(grep -oE '^[[:space:]]*-[[:space:]]*[A-Za-z0-9_/]+$' "$f" | awk '{print $2}')
    fi
    printf "%s" "$s"
}

# Each sqlite study database is created serially before any driver starts:
# drivers racing optuna's alembic bootstrap on a missing file die seconds in with
# 'UNIQUE constraint failed: alembic_version.version_num'. Idempotent.
if [ "$dry" -eq 0 ]; then
    while IFS= read -r url; do
        case "$url" in
            sqlite:///*)
                # SQLAlchemy strips the '?timeout=600' busy timeout itself; the file
                # test needs the bare path.
                path=${url#sqlite:///}
                path=${path%%\?*}
                if [ ! -f "$path" ]; then
                    mkdir -p "$(dirname "$path")"
                    echo "initialising study database: $path"
                    "$PYTHON" -c 'import optuna,sys; optuna.storages.RDBStorage(sys.argv[1])' "$url"
                fi ;;
        esac
    done < <(printf "%s\n" "$cmds" | while IFS= read -r c; do storage_of "$c"; echo; done | sort -u | grep -v '^$')
fi

n_sub=0
i=0
while IFS= read -r cmd; do
    i=$((i + 1))
    label=$(search_label "$cmd")
    selected "$i" "$label" || continue
    if [ -n "$gpu" ]; then cmd=$(pin_gpu "$cmd" "$gpu"); fi
    # A private sweep directory per driver: hydra's second-resolution timestamp
    # collides when several start together, merging their trial directories.
    sweep=" hydra.sweep.dir=logs/searches/multiruns/${sweep_ts}_${label}${tag:+_$tag}"
    full="$PYTHON $cmd$sweep${extra:+ $extra}"
    n_sub=$((n_sub + 1))
    if [ "$dry" -eq 1 ]; then
        echo "[$i/$n_total] $full"
        echo
        continue
    fi
    echo "[$i/$n_total] starting driver: $label"
    # The previous log is kept: it is the evidence of why that driver died.
    log="$logdir/${i}_${label}${tag:+_$tag}.log"
    if [ -f "$log" ]; then mv "$log" "$log.$(date +%Y%m%d-%H%M%S)"; fi
    nohup bash -c "$full" > "$log" 2>&1 &
    # Drivers sharing one sqlite file contend while creating their study rows.
    sleep 5
done <<< "$cmds"

[ "$n_sub" -gt 0 ] || { echo "nothing matched --only '$only' in $file" >&2; exit 1; }
if [ "$dry" -eq 0 ]; then
    echo
    echo "$n_sub driver(s) running detached (logs: $logdir), safe to log out."
    echo "Progress: python3 scripts/search/sweep_status.py"
fi
