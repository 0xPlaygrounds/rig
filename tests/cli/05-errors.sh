# Every error says what to run next, with the documented exit code.
script '[]'
# No server runs yet: status says so, and starts none.
expect 7 rig server status
has_next
expect 2 rig wait --nope s
has_next
expect 2 rig send
has_next
expect 2 rig frobnicate
has_next
expect 0 rig help --allow
expect 0 rig help-agents
lines=$(printf '%s\n' "$out" | wc -l)
[ "$lines" -le 30 ] || fail "help-agents' first screen has $lines lines"
expect 2 rig wait no-such-session --timeout 5s
has_next
