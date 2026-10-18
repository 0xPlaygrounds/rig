# Writes a probe into $RIG_HOME/auth: never into a linked, real sign-in.
[ -L "$RIG_HOME/auth" ] && fail "RIG_HOME/auth is a link (a real sign-in?); run this in a home of its own"
# A confined helper's shell commands can't read the model sign-in or reach
# rigd, though its own folder works; the parent's up-front rule for `cat`
# carries over to it, so nothing asks.
mkdir -p inside "$RIG_HOME/auth"
echo mine >inside/own.txt
echo sign-in >"$RIG_HOME/auth/probe"
script "[
  {\"when\": \"lead the work\", \"calls\": [{\"tool\": \"start_agent\", \"args\": {\"task\": \"helper delta: run commands\", \"name\": \"delta\", \"cwd\": \"inside\", \"dirs\": [\"inside\"]}}]},
  {\"when\": \"lead the work\", \"calls\": [{\"tool\": \"wait_agent\", \"args\": {\"ids\": [\"delta\"]}}]},
  {\"when\": \"lead the work\", \"reply\": \"done\"},
  {\"when\": \"helper delta\", \"calls\": [
    {\"tool\": \"shell\", \"args\": {\"command\": \"cat own.txt\"}},
    {\"tool\": \"shell\", \"args\": {\"command\": \"cat $RIG_HOME/auth/probe\"}}]},
  {\"when\": \"helper delta\", \"reply\": \"delta done\"}
]"
cp "$RIG_SCRIPTED_MODEL" inside/script.json
export RIG_SCRIPTED_MODEL=$PWD/inside/script.json
expect 0 rig start --name lead --allow "shell:cat" "lead the work"
parent=$out
expect 0 rig wait "$parent" --timeout 120s
expect 0 rig agents --tree --json
child=$(python3 -c 'import json,sys
rows=json.loads(sys.argv[1]); rows=rows if isinstance(rows,list) else rows.get("sessions",rows)
print([r for r in rows if r.get("parent")==sys.argv[2]][0]["sessionId"])' "$out" "$parent") || fail "no child: $out"
expect 0 rig log "$child" --json
python3 - "$out" <<'PY' || fail "the helper's commands: $out"
import json, sys
results = []
for line in sys.argv[1].splitlines():
    r = json.loads(line)
    if r.get("kind") != "conversation.change":
        continue
    for entry in r["data"]["record"].get("entries", []):
        for part in entry.get("message", {}).get("content", []) or []:
            if isinstance(part, dict) and part.get("name") == "shell":
                results.append(json.dumps(part))
assert len(results) == 2, results
own, auth = results
assert "mine" in own, own
assert "sign-in" not in auth and "Permission denied" in auth, auth
PY
expect 0 rig stop "$parent"
