# F-016: the verbs build again after plugins.toml changes, instead of
# silently running the old build; the session runs the new one.
script '[{"reply": "PONG"}, {"reply": "PONG"}]'
build_of() {
  rig log "$1" --json | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["data"]["build"])'
}
expect 0 rig start "say pong"
before=$(build_of "$out")
rig plugin disable rig_agents::AgentsPlugin >/dev/null || fail "could not disable rig-agents"
expect 0 rig start "say pong"
after=$(build_of "$out")
rig plugin enable rig_agents::AgentsPlugin >/dev/null || fail "could not enable rig-agents"
[ -n "$before" ] && [ "$before" != "$after" ] || fail "the session after the edit ran build $after, as before ($before)"
# Back as it was: one more build, then the cached one again.
expect 0 rig -p "say pong"
[ "$out" = PONG ] || fail "after re-enabling, rig -p printed '$out'"
