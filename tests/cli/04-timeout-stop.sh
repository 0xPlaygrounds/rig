# A turn still working: wait exits 124 and says to wait again; stop stops
# it (exit 0), and a wait afterwards exits 5.
script '[{"calls": [{"tool": "shell", "args": {"command": "sleep 600"}}]}]'
expect 0 rig start "work for a long time"
id=$out
expect 124 rig wait "$id" --timeout 3s
has_next
case "$err" in *"next: rig wait $id"*) ;; *) fail "next after a timeout: $err" ;; esac
expect 0 rig stop "$id" --turn
expect 5 rig wait "$id" --timeout 30s
has_next
expect 0 rig stop "$id"
# A stopped session that gets a message again runs again, and
# `rig log --follow` follows it past the old stop rather than ending there.
script '[{"reply": "BACK"}]'
expect 0 rig send "$id" "again"
expect 0 rig wait "$id" --timeout 60s
timeout 4 "$RIG" log "$id" --follow >/dev/null 2>&1
code=$?
[ "$code" = 124 ] || fail "rig log --follow ended (exit $code) at the old stop"
expect 0 rig stop "$id"
