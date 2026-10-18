# F-013: a rig -p run ends with its tree: when it exits, the helpers its
# session started are stopped, not left working unwatched.
script '[
  {"when": "lead it", "calls": [{"tool": "start_agent", "args": {"task": "helper slowpoke: work", "name": "slowpoke"}}]},
  {"when": "lead it", "reply": "started a helper"},
  {"when": "helper slowpoke", "calls": [{"tool": "shell", "args": {"command": "sleep 600"}}]}
]'
expect 0 rig -p --json "lead it"
out=$(printf '%s\n' "$out" | tail -1)
[ "$(field answer)" = "started a helper" ] || fail "rig -p answered: $out"
lead=$(field sessionId)
expect 0 rig agents --tree --all --json
python3 - "$out" "$lead" <<'PY' || fail "the helper after rig -p: $out"
import json, sys
rows = json.loads(sys.argv[1])["sessions"]
kids = [r for r in rows if r.get("parent") == sys.argv[2]]
assert kids, rows
assert all(r["state"] == "stopped" for r in kids), kids
PY
