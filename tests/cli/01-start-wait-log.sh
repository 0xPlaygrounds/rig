# start, wait, send, wait, log: the common loop, with exit 0 and a clean
# stderr on a cached build (M-7, F-006).
script '[{"reply": "PONG"}, {"reply": "PONG again"}]'
expect 0 rig start "say pong"
id=$out
[ -n "$id" ] || fail "rig start printed no id"
[ -z "$err" ] || fail "rig start wrote to stderr: $err"
expect 0 rig wait "$id" --timeout 60s
[ "$out" = "PONG" ] || fail "wait printed '$out'"
[ -z "$err" ] || fail "rig wait wrote to stderr: $err"
expect 0 rig send "$id" "once more"
case "$out" in *" started"|*" queued"|*" steered") ;; *) fail "send printed '$out'" ;; esac
expect 0 rig wait "$id" --timeout 60s
[ "$out" = "PONG again" ] || fail "second wait printed '$out'"
expect 0 rig log "$id"
case "$out" in *"PONG again"*) ;; *) fail "the log lacks the answer: $out" ;; esac
expect 0 rig agents
case "$out" in *"$id"*) ;; *) fail "rig agents lacks $id: $out" ;; esac
