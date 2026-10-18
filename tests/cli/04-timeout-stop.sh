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
