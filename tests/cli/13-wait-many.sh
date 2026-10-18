# F-021: rig wait A B returns when the first of them settles, naming it;
# --all waits for every one.
script '[
  {"when": "quick", "reply": "QUICK"},
  {"when": "slow", "calls": [{"tool": "shell", "args": {"command": "sleep 600"}}]}
]'
expect 0 rig start "slow work"
slow=$out
expect 0 rig start "quick work"
quick=$out
expect 0 rig wait "$slow" "$quick" --timeout 60s
[ "$(printf '%s\n' "$out" | head -1)" = "$quick" ] || fail "the first line is not the settled session: $out"
[ "$(printf '%s\n' "$out" | tail -1)" = QUICK ] || fail "the answer: $out"
expect 0 rig wait --json "$slow" "$quick" --timeout 60s
[ "$(field sessionId)" = "$quick" ] || fail "sessionId in $out"
expect 124 rig wait --all "$slow" "$quick" --timeout 3s
has_next
# The settled one is printed; next: waits for the one still working.
case "$out" in *"$quick"*QUICK*) ;; *) fail "--all printed: $out" ;; esac
case "$err" in *"next: rig wait --all $slow"*) ;; *) fail "next for --all: $err" ;; esac
expect 2 rig wait "$slow" "$quick" --after 1
has_next
expect 0 rig stop "$slow"
