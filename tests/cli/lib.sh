# Helpers every tests/cli scenario gets (sourced by tests/cli/run).

cd "$RIG_CLI_WORK" || exit 1
rig() { "$RIG" "$@"; }

# script JSON: the scripted model's turns for what runs next.
script() {
  printf '%s\n' "$1" >"$RIG_CLI_WORK/script.json"
  export RIG_SCRIPTED_MODEL=$RIG_CLI_WORK/script.json
  # A running server's workers read the script at start; start afresh.
  "$RIG" server stop >/dev/null 2>&1
}

fail() {
  echo "failed: $*"
  exit 1
}

# expect CODE CMD...: runs CMD, which must exit CODE; its stdout is in $out,
# its stderr in $err.
expect() {
  local want=$1
  shift
  out=$("$@" 2>"$RIG_CLI_WORK/stderr")
  local got=$?
  err=$(cat "$RIG_CLI_WORK/stderr")
  if [ "$got" != "$want" ]; then
    fail "\`$*\` exited $got, not $want; stdout: $out; stderr: $err"
  fi
}

# Every failure says what to run next.
has_next() {
  case "$err" in
  *"next: "*) ;;
  *) fail "no next: line in stderr: $err" ;;
  esac
}

# The JSON field PATH of $out (one object), through python's json.
field() {
  python3 -c 'import json,sys
v=json.loads(sys.argv[1])
for k in sys.argv[2].split("."):
    v=v[k]
print(v)' "$out" "$1"
}
