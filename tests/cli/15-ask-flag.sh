# --ask: the default build asks nothing; --ask shell makes a session ask
# before a shell command (wait exits 3), and --allow answers up front.
script '[
  {"when": "run it", "calls": [{"tool": "shell", "args": {"command": "echo hi"}}]},
  {"when": "run it", "reply": "ran"}
]'
expect 0 rig start --ask shell "run it"
id=$out
expect 3 rig wait "$id" --timeout 60s
case "$err" in *"next: rig answer $id "*) ;; *) fail "no answer command: $err" ;; esac
expect 0 rig answer "$id" --allow
expect 0 rig wait "$id" --timeout 60s
[ "$out" = ran ] || fail "after the approval: '$out'"
expect 0 rig -p --ask shell --allow "shell:echo" "run it"
[ "$out" = ran ] || fail "an up-front rule should answer: '$out'"
