# F-013: a rig -p run ends with its tree: when it exits, the helpers its
# session started are stopped, not left working unwatched.
script '[
  {"when": "lead it", "calls": [{"tool": "start_agent", "args": {"task": "helper slowpoke: work", "name": "slowpoke"}}]},
  {"when": "lead it", "reply": "started a helper"},
  {"when": "helper slowpoke", "calls": [{"tool": "shell", "args": {"command": "sleep 600"}}]}
]'
expect 0 rig -p "lead it"
[ "$out" = "started a helper" ] || fail "rig -p printed '$out'"
expect 0 rig agents --tree --all --json
python3 - "$out" <<'PY' || fail "the helper after rig -p: $out"
import json, sys
rows = json.loads(sys.argv[1])["sessions"]
kids = [r for r in rows if r.get("parent")]
assert kids, rows
assert all(r["state"] == "stopped" for r in kids), kids
PY
