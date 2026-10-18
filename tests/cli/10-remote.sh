# --host: the verbs reach a host's rig server through its [remote.hosts]
# transport, which runs `rig connect --stdio` there. Here the "remote" is
# this machine through `sh -c`, so the bridge, its token and the remote
# client path all run for real.
mkdir -p bin
ln -sf "$RIG" bin/rig
export PATH="$PWD/bin:$PATH"
grep -q '^loop = ' "$RIG_HOME/settings.toml" 2>/dev/null ||
  printf '\n[remote.hosts]\nloop = ["sh", "-c"]\n' >>"$RIG_HOME/settings.toml"
script '[{"reply": "REMOTE"}]'
expect 0 rig start --host loop --cwd "$PWD" "say it"
id=$out
expect 0 rig wait --host loop "$id" --timeout 60s
[ "$out" = REMOTE ] || fail "the remote answer: '$out'"
expect 0 rig agents --host loop --json
case "$out" in *"$id"*) ;; *) fail "the remote's sessions lack $id: $out" ;; esac
expect 7 rig agents --host nowhere
has_next
