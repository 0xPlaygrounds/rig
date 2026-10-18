# F-025: rig -p -r ID unloads the resumed session after its turn, as a
# fresh rig -p does, rather than leave its worker (and so rigd) running.
script '[{"reply": "ONE"}, {"reply": "TWO"}]'
expect 0 rig -p --json "first"
out=$(printf '%s\n' "$out" | tail -1)
id=$(field sessionId)
expect 0 rig -p -r "$id" "second"
[ "$out" = TWO ] || fail "rig -p -r printed '$out'"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  expect 0 rig agents --json
  loaded=$(python3 -c 'import json,sys
rows=json.loads(sys.argv[1])["sessions"]
print([r["worker"] for r in rows if r["sessionId"]==sys.argv[2]][0])' "$out" "$id")
  [ "$loaded" = False ] && exit 0
  sleep 1
done
fail "the resumed session's worker still runs 10 s after rig -p -r"
