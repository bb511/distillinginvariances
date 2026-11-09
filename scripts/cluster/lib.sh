#!/bin/bash
# Helpers of submit.sh and submit_ngt.sh.
#
# The block parser below is the single definition of what a "command" in a
# scripts/{mnist,jets}/run*_{train,eval}.sh catalogue is.

# A block opens at a commented 'python3 src/{train,eval,cache}.py' line and runs
# to the first line without a trailing continuation. Emits one flat command per
# block, entry point first, so the caller need not branch on which of the three
# it holds.
extract_cmds() {
    awk '
        /^# python3 src\/(train|eval|cache)\.py/ {
            line = $0
            sub(/^# python3[ \t]+/, "", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            ends = (line ~ /\\$/)
            sub(/[ \t]*\\$/, "", line)
            cmd = line
            collecting = 1
            if (!ends) { collecting = 0; print cmd }
            next
        }
        collecting {
            line = $0
            sub(/^# ?/, "", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            ends = (line ~ /\\$/)
            sub(/[ \t]*\\$/, "", line)
            if (line != "") cmd = cmd " " line
            if (!ends) { collecting = 0; print cmd }
        }
    ' "$1"
}

# The value of an override, e.g. field "$cmd" 'data\.regime'. Empty when absent.
# The key is a regex, and it has to sit at the start of the command or behind a
# space, so that a key is never read as the tail of a longer one ('fold' never
# matches the 'n_folds' of 'data.n_folds=5').
field() {
    printf "%s" "$1" |
        grep -oE "(^|[[:space:]])$2=[^ ]+" |
        head -1 |
        sed -E 's/^[[:space:]]+//' |
        cut -d= -f2- || true
}

# The block's position in the study, '<experiment>_<regime>_<method>_<run>[_s<seed>]',
# <run> being 'main' for a block without a fold override and 'fold<k>' otherwise:
# what --only matches on and what names the driver's logs.
block_label() {
    local exp regime method fold seed run=main
    exp=$(field "$1" 'experiment')
    exp=${exp##*/}
    regime=$(field "$1" 'data\.regime')
    method=$(field "$1" 'method')
    fold=$(field "$1" 'fold')
    seed=$(field "$1" 'seed')
    [ -z "$fold" ] || run="fold$fold"
    printf "%s_%s_%s_%s%s" "$exp" "${regime:-canonical}" "${method:-teacher}" "$run" \
        "${seed:+_s$seed}"
}

# $1 = 1-based index, $2 = name, matched against the caller's $only (comma-
# separated name substrings or indices; empty selects everything). Explicit ifs
# rather than 'test && action': under 'set -e' a failing && list aborts.
selected() {
    if [ -z "$only" ]; then return 0; fi
    local idx="$1" name="$2" s
    local IFS=','
    for s in $only; do
        if [ -n "$s" ]; then
            case "$s" in
                # An all-digit selector is an index only, never a substring.
                *[!0-9]*) case "$name" in *"$s"*) return 0 ;; esac ;;
                *) if [ "$s" = "$idx" ]; then return 0; fi ;;
            esac
        fi
    done
    return 1
}

# $1 = 1-based count of the selected commands, matched against the caller's
# $share, 'j/k' (empty keeps everything): command n belongs to share (n - 1) mod k.
in_share() {
    if [ -z "$share" ]; then return 0; fi
    [ $((($1 - 1) % ${share#*/})) -eq "${share%/*}" ]
}

# '5' -> 1..5, '2-5' -> 2..5, '1,3' -> 1 3. Empty prints nothing, which leaves a
# block at the fold its own command carries.
expand_folds() {
    local spec="$1" part
    local IFS=','
    for part in $spec; do
        case "$part" in
            *-*) seq "${part%%-*}" "${part##*-}" ;;
            '') ;;
            *) seq 1 "$part" ;;
        esac
    done
}

# $1 = command, $2 = fold. The run directory follows from the fold through the
# config, so only the override itself and the fold_<k> component of the upstream
# paths a block names (a manifest, a teacher cache) are rewritten. A main-run block,
# which carries no fold override, gains one; its upstream paths are written with
# ${run_label}, which follows the fold by itself.
substitute_fold() {
    if printf "%s" "$1" | grep -qE "(^|[[:space:]])fold=[0-9]+"; then
        printf "%s" "$1" | sed -E "s/(^|[[:space:]])fold=[0-9]+/\\1fold=$2/g; s#fold_[0-9]+#fold_$2#g"
    else
        printf "%s fold=%s" "$1" "$2"
    fi
}

# Every command and every log directory is relative to the repository root.
require_repo_root() {
    if [ ! -f .project-root ]; then
        echo "run this from the repository root" >&2
        exit 1
    fi
}

# Export what scripts/setup.sh wrote, so a nohup'd driver inherits PROJECT_ROOT
# (configs/paths/default.yaml will not compose without it), HF_HUB_CACHE and
# MLFLOW_ALLOW_FILE_STORE. An absent .env is not an error: the caller may have
# exported them itself.
load_env() {
    if [ -f .env ]; then
        set -a
        # shellcheck disable=SC1091  # .env is generated, not checked in
        . ./.env
        set +a
    fi
    # The loader runs inside the driver and its CPU work is small, so torch's default of one
    # thread per core would only make the drivers sharing a machine contend.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
}

# Without HF_HUB_CACHE the 5.7 GB jets snapshot is downloaded again into the
# driver's home directory, one copy per user on a shared machine.
require_hf_cache() {
    if [ -z "${HF_HUB_CACHE:-}" ]; then
        echo "error: HF_HUB_CACHE is not set - the record snapshot would be re-downloaded." >&2
        echo "       run 'bash scripts/setup.sh' and edit .env, or pass --hf-cache <path>." >&2
        exit 1
    fi
}

# $1 = command, $2 = GPU index. The catalogues all say trainer.devices=[0]; this
# is what makes several queues on one multi-GPU host independent.
pin_gpu() {
    printf "%s" "$1" | sed -E "s/trainer\.devices=\[[0-9]+\]/trainer.devices=[$2]/"
}

# $1 = python, $2 = modules, rest = hint lines. The drivers are nohup'd, so an
# inactive env shows up only as a tiny log per driver and a whole batch is lost.
require_imports() {
    local py="$1" mods="$2" line
    shift 2
    "$py" -c "import $mods" >/dev/null 2>&1 && return 0
    echo "error: '$py' cannot import $mods - the training env is not active." >&2
    for line in "$@"; do echo "       $line" >&2; done
    exit 1
}

# $1 = queue path, $2 = the catalogue it came from. A queue is a plain script so
# that it can be read, edited and re-run by hand.
start_queue() {
    printf '#!/bin/bash\n# Generated on %s from %s.\n' "$(date +%Y-%m-%d_%H-%M-%S)" "$2" > "$1"
    # No 'set -e': one block that dies must not take the rest of the queue with it.
    printf 'cd %s || exit 1\n' "$PWD" >> "$1"
}
