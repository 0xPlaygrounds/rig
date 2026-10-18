# Asks: an approval makes rig wait exit 3 and a question 4, each printing
# the command that answers it; rig answer answers, and the turn goes on.
# The default plugins ask nothing, so the session runs on a plugin list
# (rig start --plugins) with two corpus plugins that ask: the git guard,
# set to ask before a `git commit --no-verify`, and ask_user.
printf '[rules]\nno-verify = "ask"\n' >"$RIG_HOME/git-guard.toml"
asks=$RIG_CLI_WORK/asks.toml
cp "$RIG_HOME/plugins.toml" "$asks"
cat >>"$asks" <<TOML

[[plugin]]
crate = "corpus-ct"
path = "$RIG_SOURCE/corpus/ct"
plugin = "corpus_ct::git_guardrails::GitGuardPlugin"

[[plugin]]
crate = "corpus-base"
path = "$RIG_SOURCE/corpus/base"
plugin = "corpus_base::ask_user::AskUserPlugin"
TOML
git init -q repo
script '[
  {"when": "commit it", "calls": [{"tool": "shell", "args": {"command": "git -c user.name=t -c user.email=t@t commit --no-verify --allow-empty -m x"}}]},
  {"when": "commit it", "reply": "committed"},
  {"when": "ask me", "calls": [{"tool": "ask_user", "args": {"question": "which colour?"}}]},
  {"when": "ask me", "reply": "thanks"}
]'
expect 0 rig start --plugins "$asks" --cwd repo "commit it"
id=$out
expect 3 rig wait "$id" --timeout 120s
case "$out" in *"next: rig answer $id "*"--allow"*) ;; *) fail "no answer command in: $out" ;; esac
expect 0 rig answer "$id" --allow
expect 6 rig answer "$id" --allow
has_next
expect 0 rig wait "$id" --timeout 60s
[ "$out" = committed ] || fail "after the approval, wait printed '$out'"

expect 0 rig start --plugins "$asks" "ask me"
id=$out
expect 4 rig wait "$id" --timeout 60s
case "$out" in *"which colour?"*"next: rig answer $id "*"--text"*) ;; *) fail "the question: $out" ;; esac
expect 0 rig answer "$id" --text blue
expect 0 rig wait "$id" --timeout 60s
[ "$out" = thanks ] || fail "after the answer, wait printed '$out'"
