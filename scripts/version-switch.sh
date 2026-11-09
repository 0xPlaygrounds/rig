#!/usr/bin/env bash

# Switch one used rig home between two versions of the coding agent and back,
# and check what a user who switches keeps. The release PR runs it with the
# previous release as OLD and the release candidate as NEW. Not part of CI: it
# builds the agent five times (a few minutes each) and needs crates.io.
#
# Usage: scripts/version-switch.sh OLD NEW --work DIR [--expected FILE]
#                                  [--port N] [--timeout SECONDS]
#   OLD, NEW         a git ref of this repository, built into a launcher
#                    under DIR, or the path of an installed `rig` launcher
#   --work DIR       everything goes here: worktrees, launcher builds, the
#                    disposable home, logs and summary.txt (new or empty)
#   --expected FILE  check ids known to fail for this pair, one per line
#                    (`#` starts a comment); they report XFAIL, and one that
#                    passes reports XPASS and fails the run so the list stays
#                    current. See scripts/version-switch-expected.txt.
#   --port N         the fake model's port (default 11434: v0.46.0 talks to
#                    Ollama on localhost:11434 only)
#   --timeout S      limit for one rig command, build included (default 1800)
#
# The model is scripts/version-switch-model.py, a fake Ollama that answers
# with the user turns it was sent and can hold a turn open. Every rig command
# runs with `env -i` and a fresh HOME, XDG_*, RIG_HOME, CLAUDE_CONFIG_DIR,
# CODEX_HOME and GIT_CONFIG_GLOBAL, so no login, key or setting of the caller
# leaks in. Builds use CARGO_BUILD_JOBS (default 4). Linux, bash 4+, python3.
#
# Exits 0 when every check passes or fails as expected, 1 otherwise, 2 on
# bad arguments.

set -euo pipefail

usage() {
    sed -n '/^# Usage:/,/^# Exits/p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

die() {
    echo "version-switch: $*" >&2
    exit 1
}

old_arg="" new_arg="" work="" expected_file="" port=11434 timeout_s=1800
while (($#)); do
    case "$1" in
        --work) work="${2:?}"; shift 2 ;;
        --expected) expected_file="${2:?}"; shift 2 ;;
        --port) port="${2:?}"; shift 2 ;;
        --timeout) timeout_s="${2:?}"; shift 2 ;;
        -h | --help) usage ;;
        -*) usage ;;
        *)
            if [[ -z "$old_arg" ]]; then old_arg="$1"
            elif [[ -z "$new_arg" ]]; then new_arg="$1"
            else usage; fi
            shift ;;
    esac
done
[[ -n "$old_arg" && -n "$new_arg" && -n "$work" ]] || usage
for tool in git cargo python3 timeout du stat; do
    command -v "$tool" >/dev/null || die "$tool is required"
done

repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
model_script="$repo/scripts/version-switch-model.py"
mkdir -p "$work"
work="$(cd "$work" && pwd)"
[[ -z "$(ls -A "$work")" ]] || die "$work is not empty; pass a new directory"

declare -A expected=()
if [[ -n "$expected_file" ]]; then
    while read -r id _; do
        [[ -z "$id" || "$id" == \#* ]] || expected[$id]=1
    done <"$expected_file"
fi

# The plugins each version turns off with its own `rig plugin disable`; the
# other version must show them off too. Two, so the downgrade cannot pass on
# a file the old version wrote before the upgrade.
plugin_by_old="rig_basics::skills::SkillsPlugin"
plugin_by_new="rig_basics::agents_md::AgentsMdPlugin"
marker="# Written by hand before any switch; every version must keep this line."

logs="$work/logs" home="$work/home" project="$work/project"
rig_home="$home/.rig" hold="$work/hold" model_log="$work/logs/model.log"
mkdir -p "$logs" "$project" "$home"/{.config,.local/share,.local/state,.cache,.claude,.codex,tmp,run}
chmod 700 "$home/run"
printf '[user]\n\tname = version switch\n\temail = switch@example.invalid\n' >"$home/.gitconfig"
printf 'alpha\nbeta\n' >"$project/notes.txt"
caller_path="$PATH"
cargo_home="${CARGO_HOME:-$HOME/.cargo}"
rustup_home="${RUSTUP_HOME:-$HOME/.rustup}"
jobs="${CARGO_BUILD_JOBS:-4}"

model_pid="" held_pid="" worktrees=()
# shellcheck disable=SC2329 # run by the EXIT trap
cleanup() {
    touch "$hold" 2>/dev/null || true
    [[ -n "$held_pid" ]] && kill "$held_pid" 2>/dev/null || true
    [[ -n "$model_pid" ]] && kill "$model_pid" 2>/dev/null || true
    for tree in "${worktrees[@]}"; do
        git -C "$repo" worktree remove --force "$tree" 2>/dev/null || true
    done
}
trap cleanup EXIT

# launcher_for LABEL ARG: the launcher path for ARG, building a git ref.
declare -A launcher=()
launcher_for() {
    local label="$1" arg="$2" commit tree package
    if [[ -f "$arg" && -x "$arg" ]]; then
        launcher[$label]="$(cd "$(dirname "$arg")" && pwd)/$(basename "$arg")"
        return
    fi
    commit="$(git -C "$repo" rev-parse --verify --quiet "$arg^{commit}")" ||
        die "$arg is neither a launcher file nor a git ref"
    tree="$work/src/$label"
    git -C "$repo" worktree add --quiet --detach "$tree" "$commit"
    worktrees+=("$tree")
    package=rig
    [[ -f "$tree/crates/rig-cli/Cargo.toml" ]] && package=rig-cli
    echo "building the $label launcher from $arg ($commit) …"
    (cd "$tree" && CARGO_TARGET_DIR="$work/build/$label" CARGO_BUILD_JOBS="$jobs" \
        cargo build --release --locked -p "$package" --bin rig) >"$logs/build-$label.log" 2>&1 ||
        die "the $label launcher did not build; see $logs/build-$label.log"
    launcher[$label]="$work/build/$label/release/rig"
}

# rig LABEL NAME ARGS…: runs launcher LABEL in the project directory with the
# disposable home; output in logs/NAME.{out,err}, exit code in $rc, wall
# seconds in $wall.
rc=0 wall=0
rig() {
    local label="$1" name="$2" start
    shift 2
    start=$EPOCHREALTIME
    rc=0
    (cd "$project" && env -i \
        HOME="$home" PATH="$caller_path" LANG=C.UTF-8 TERM=dumb \
        XDG_CONFIG_HOME="$home/.config" XDG_DATA_HOME="$home/.local/share" \
        XDG_STATE_HOME="$home/.local/state" XDG_CACHE_HOME="$home/.cache" \
        XDG_RUNTIME_DIR="$home/run" CLAUDE_CONFIG_DIR="$home/.claude" \
        CODEX_HOME="$home/.codex" GIT_CONFIG_GLOBAL="$home/.gitconfig" \
        RIG_HOME="$rig_home" TMPDIR="$home/tmp" CARGO_HOME="$cargo_home" \
        RUSTUP_HOME="$rustup_home" CARGO_BUILD_JOBS="$jobs" \
        OLLAMA_API_BASE_URL="http://127.0.0.1:$port" \
        timeout "$timeout_s" "${launcher[$label]}" "$@") \
        </dev/null >"$logs/$name.out" 2>"$logs/$name.err" || rc=$?
    wall=$(awk -v a="$start" -v b="$EPOCHREALTIME" 'BEGIN { printf "%.0f", b - a }')
}

results=() failed=0
# check ID OK DETAIL: records one check (OK is 0 for a pass).
check() {
    local id="$1" ok="$2" detail="$3" state
    if ((ok == 0)); then
        state=PASS
        [[ -n "${expected[$id]:-}" ]] && state=XPASS && failed=1
    else
        state=FAIL
        if [[ -n "${expected[$id]:-}" ]]; then state=XFAIL; else failed=1; fi
    fi
    results+=("$(printf '%-6s %-26s %s' "$state" "$id" "$detail")")
    echo "  $state $id: $detail"
}

# status CMD…: 0 when CMD succeeds, 1 otherwise, without tripping set -e.
status() { if "$@"; then echo 0; else echo 1; fi; }
answered() { grep -q "FAKE-MODEL turns:.*$2" "$logs/$1.out"; }
first_error() { grep -m1 -v '^\s*$' "$logs/$1.err" | cut -c1-160 || true; }
target_bytes() { du -sb "$rig_home/target" 2>/dev/null | cut -f1 || echo 0; }
gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1e9 }'; }

# build_since NAME: when RIG_HOME/build.log changed after logs/.stamp, keeps
# it as logs/NAME.build.log and prints how many crates cargo compiled and
# how many locked versions it changed; prints nothing otherwise.
build_since() {
    local kept="$logs/$1.build.log" compiled changed
    [[ "$rig_home/build.log" -nt "$logs/.stamp" ]] || return 0
    cp "$rig_home/build.log" "$kept"
    compiled=$(grep -c -E '^ +Compiling ' "$kept" || true)
    changed=$(grep -c -E '^ +(Updating|Downgrading|Upgrading|Adding|Removing) [^ ]+ v[0-9]' "$kept" || true)
    echo "$compiled crates compiled, $changed locked versions changed"
}

switches=()
built=""
# turn LABEL NAME ARGS…: a turn that may rebuild; records its time, what
# its build compiled (in $built) and the disk RIG_HOME/target uses.
turn() {
    local label="$1" name="$2" before
    before=$(target_bytes)
    touch "$logs/.stamp"
    sleep 1
    rig "$@"
    built=$(build_since "$name")
    switches+=("$(printf '%-16s %-4s %5ss  target %s -> %s  %s' "$name" "$label" "$wall" \
        "$(gb "$before")" "$(gb "$(target_bytes)")" "${built:-no build}")")
    echo "  $name: exit $rc, ${wall}s, ${built:-no build}"
}

# plugin_off LABEL NAME PLUGIN ID: checks that launcher LABEL lists PLUGIN off,
# or refuses the settings with a message naming them.
plugin_off() {
    local label="$1" name="$2" plugin="$3" id="$4"
    rig "$label" "$name" plugin list
    if ((rc != 0)); then
        check "$id" "$(status grep -q -i 'settings\|plugins.toml' "$logs/$name.err")" \
            "refused (exit $rc): $(first_error "$name")"
    elif grep -E -q "^off +$plugin\b" "$logs/$name.out"; then
        check "$id" 0 "$plugin is still off"
    else
        check "$id" 1 "$plugin is on again, silently"
    fi
}

hand_line_kept() {
    check "$1" "$(status grep -q -F -x "$marker" "$rig_home/settings.toml")" \
        "the hand-written line in settings.toml"
}

launcher_for old "$old_arg"
launcher_for new "$new_arg"
rig old version --version && old_version="$(cat "$logs/version.out")"
rig new version --version && new_version="$(cat "$logs/version.out")"
echo "old: $old_arg (${launcher[old]}, $old_version)"
echo "new: $new_arg (${launcher[new]}, $new_version)"

if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    die "port $port is taken (a running Ollama?); stop it or pass --port"
fi
python3 -I "$model_script" "$port" "$model_log" "$hold" >"$logs/model.err" 2>&1 &
model_pid=$!
for _ in $(seq 50); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break
    sleep 0.1
done
model=(-m ollama/qwen3)

echo "== old version: a session and settings"
turn old old-first-turn "${model[@]}" -p "turn one on old"
answered old-first-turn "turn one on old" || die "the old version did not answer: $(first_error old-first-turn) (logs in $logs)"
session="$(ls "$rig_home/sessions" 2>/dev/null || true)"
[[ "$session" =~ ^[^[:space:]]+$ ]] || die "expected one session in $rig_home/sessions, found: $session"
rig old disable-old plugin disable "$plugin_by_old"
((rc == 0)) || die "rig plugin disable failed on the old version: $(first_error disable-old)"
printf '%s\n' "$marker" >>"$rig_home/settings.toml"

echo "== upgrade: new resumes the old session"
turn new upgrade "${model[@]}" -r "$session" -p "turn two on new"
check resume-upgrade "$(status answered upgrade "turn one on old | turn two on new")" \
    "exit $rc; the new version sees the old turn"
plugin_off new list-upgrade "$plugin_by_old" settings-upgrade
hand_line_kept settings-file-upgrade

echo "== new again: same version, no rebuild"
turn new restart "${model[@]}" -r "$session" -p "turn three on new"
check no-rebuild-same-version "$(status grep -q '^0 crates\|^$' <<<"$built")" \
    "${wall}s, ${built:-no build} on a second start of the same version"
rig new disable-new plugin disable "$plugin_by_new"
((rc == 0)) || die "rig plugin disable failed on the new version: $(first_error disable-new)"

echo "== downgrade: old resumes the session new wrote to"
turn old downgrade "${model[@]}" -r "$session" -p "turn four on old"
check resume-downgrade "$(status answered downgrade "turn three on new | turn four on old")" \
    "exit $rc; the old version sees the new turns"
plugin_off old list-downgrade "$plugin_by_new" settings-downgrade
hand_line_kept settings-file-downgrade

echo "== one home, both versions at once: new holds a turn while old runs"
rm -f "$hold"
touch "$logs/.stamp"
sleep 1
start=$EPOCHREALTIME
(rig new shared-new "${model[@]}" -p "[hold] new keeps running"; echo "$rc $wall" >"$logs/shared-new.rc") &
held_pid=$!
until grep -q '"held": true' "$model_log" 2>/dev/null || ! kill -0 "$held_pid" 2>/dev/null; do
    sleep 1
done
wall=$(awk -v a="$start" -v b="$EPOCHREALTIME" 'BEGIN { printf "%.0f", b - a }')
built=$(build_since shared-new)
switches+=("$(printf '%-16s %-4s %5ss  until its held turn reached the model  %s' shared-new new "$wall" "${built:-no build}")")
turn old shared-old "${model[@]}" -p "old turn while new runs"
check shared-home-old-turn "$(status answered shared-old "old turn while new runs")" \
    "exit $rc while a new agent ran on the same home"
touch "$hold"
wait "$held_pid" || true
held_pid=""
read -r rc wall <"$logs/shared-new.rc" || rc=1
check shared-home-running-new "$(status answered shared-new "new keeps running")" \
    "exit $rc; the running new agent finished its turn after old rebuilt"

{
    echo "version switch: old $old_arg ($old_version) -> new $new_arg ($new_version)"
    echo
    echo "checks:"
    printf '  %s\n' "${results[@]}"
    echo
    echo "turns (wall time includes any agent build; target is RIG_HOME/target):"
    printf '  %s\n' "${switches[@]}"
    echo
    echo "logs: $logs"
} | tee "$work/summary.txt"
exit "$failed"
