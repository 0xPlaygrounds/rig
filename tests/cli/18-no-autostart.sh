# Reading starts no server (analysis §3.5): with none running, rig agents
# says so, and rig wait and rig log on a finished session read its
# journal, leaving no rigd behind.
script '[{"reply": "PONG"}]'
expect 0 rig -p --json "say pong"
out=$(printf '%s\n' "$out" | tail -1)
id=$(field sessionId)
rig server stop >/dev/null 2>&1
expect 7 rig agents
has_next
expect 0 rig wait "$id"
[ "$out" = PONG ] || fail "wait without a server printed '$out'"
expect 0 rig log "$id"
case "$out" in *PONG*) ;; *) fail "log without a server: $out" ;; esac
expect 7 rig server status
