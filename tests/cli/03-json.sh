# --json: one object with "rig":1, a kind, the ids, status and next.
script '[{"reply": "PONG"}]'
expect 0 rig start --json "say pong"
[ "$(field rig)" = 1 ] || fail "no rig:1 in $out"
[ "$(field kind)" = started ] || fail "kind in $out"
id=$(field sessionId)
case "$(field next)" in "rig wait $id") ;; *) fail "next in $out" ;; esac
expect 0 rig wait --json "$id" --timeout 60s
[ "$(field kind)" = wait ] || fail "kind in $out"
[ "$(field status)" = answered ] || fail "status in $out"
[ "$(field answer)" = PONG ] || fail "answer in $out"
expect 0 rig agents --json
[ "$(field kind)" = agents ] || fail "kind in $out"
expect 0 rig log --json "$id"
first=$(printf '%s\n' "$out" | head -1)
out=$first
[ "$(field rig)" = 1 ] || fail "a log line lacks rig:1: $first"
[ "$(field seq)" = 1 ] || fail "the log starts at seq $(field seq)"
