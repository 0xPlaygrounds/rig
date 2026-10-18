# A helper granted only one folder can't read beyond it: not a file beside
# it through a symlink inside, not rig's run/token. Its parent gets the
# refusals in its report, and its own folder still works.
mkdir -p inside
echo secret >secret.txt
echo mine >inside/own.txt
ln -s "$PWD/secret.txt" inside/link
script "[
  {\"when\": \"lead the work\", \"calls\": [{\"tool\": \"start_agent\", \"args\": {\"task\": \"helper gamma: read files\", \"name\": \"gamma\", \"cwd\": \"inside\", \"dirs\": [\"inside\"]}}]},
  {\"when\": \"lead the work\", \"calls\": [{\"tool\": \"wait_agent\", \"args\": {\"ids\": [\"gamma\"]}}]},
  {\"when\": \"lead the work\", \"reply\": \"done\"},
  {\"when\": \"helper gamma\", \"calls\": [
    {\"tool\": \"read\", \"args\": {\"path\": \"own.txt\"}},
    {\"tool\": \"read\", \"args\": {\"path\": \"link\"}},
    {\"tool\": \"read\", \"args\": {\"path\": \"$RIG_HOME/run/token\"}}]},
  {\"when\": \"helper gamma\", \"reply\": \"gamma done\"}
]"
# The helper reads the script too, and may read only its folder.
cp "$RIG_SCRIPTED_MODEL" inside/script.json
export RIG_SCRIPTED_MODEL=$PWD/inside/script.json
expect 0 rig start --name lead "lead the work"
parent=$out
expect 0 rig wait "$parent" --timeout 120s
expect 0 rig agents --tree --json
child=$(python3 -c 'import json,sys
rows=json.loads(sys.argv[1]); rows=rows if isinstance(rows,list) else rows.get("sessions",rows)
print([r for r in rows if r.get("parent")==sys.argv[2]][0]["sessionId"])' "$out" "$parent") || fail "no child: $out"
expect 0 rig log "$child" --json
python3 - "$out" <<'PY' || fail "the helper's reads: $out"
import json, sys
results = []
for line in sys.argv[1].splitlines():
    r = json.loads(line)
    if r.get("kind") != "conversation.change":
        continue
    for entry in r["data"]["record"].get("entries", []):
        for part in entry.get("message", {}).get("content", []) or []:
            if isinstance(part, dict) and part.get("name") == "read":
                results.append(json.dumps(part))
assert len(results) == 3, results  # one result per read call
own, link, token = results
assert "mine" in own, own
# The refusals name the file, never its contents.
for refused in (link, token):
    assert "Permission denied" in refused and '"is_error": true' in refused, refused
assert "secret\\n" not in link, link
print("\n".join(results))
PY
echo "$out" >/dev/null
expect 0 rig stop "$parent"
