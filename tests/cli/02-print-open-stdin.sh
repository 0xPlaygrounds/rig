# M-1: `rig -p PROMPT` never reads stdin, so an open pipe that never
# closes does not hang it; stdout is the answer alone.
script '[{"reply": "PONG"}]'
started=$(date +%s)
out=$(rig -p "say pong" < <(sleep 120) 2>"$RIG_CLI_WORK/stderr")
code=$?
elapsed=$(( $(date +%s) - started ))
[ "$code" = 0 ] || fail "rig -p exited $code: $(cat "$RIG_CLI_WORK/stderr")"
[ "$out" = "PONG" ] || fail "rig -p printed '$out'"
[ "$elapsed" -le 30 ] || fail "rig -p took ${elapsed}s"
# `-` reads the prompt from stdin.
script '[{"when": "from stdin", "reply": "READ"}]'
expect 0 bash -c 'echo "from stdin" | "$RIG" -p -'
[ "$out" = "READ" ] || fail "rig -p - printed '$out'"
# -m picks the model before the prompt runs, as drivers pass it.
script '[{"reply": "PONG"}]'
expect 0 rig -m scripted/model -p "say pong"
[ "$out" = "PONG" ] || fail "rig -m ... -p printed '$out'"
expect 0 rig start -m scripted/model "say pong"
id=$out
expect 0 rig wait "$id" --timeout 60s
[ "$out" = "PONG" ] || fail "rig start -m ...; wait printed '$out'"
