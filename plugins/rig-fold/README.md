# rig-fold

`rig_fold::FoldPlugin` folds finished tool calls in the rig agent's terminal
view. It is off by default: without it every call is drawn whole. Turn it on
with `rig plugin enable rig_fold::FoldPlugin`, in `/plugins`, or in
`$RIG_HOME/settings.toml`:

```toml
[plugins]
enable = ["rig_fold::FoldPlugin"]
```

A finished call shows its header, the first lines after it, a line that
counts the lines left out and names the key, and the last lines:

```
 $ seq 400
   1
   2
   3
   4
   … 388 lines · ctrl+o shows
   393
   394
   395
   396
   397
   398
   399
   400
```

It works on what each tool's look drew, so it folds every tool the same way,
an MCP server's included. A call that failed or was stopped, and one still
running, is drawn whole, and so is a result too short to save two rows. Only
transcript panes with their own keys fold, such as the main one. A picker's
preview, where the key can't act, is drawn whole.

The key shows the call selected with Alt+Up whole, and folds it again. With
no call selected it shows every call of the agent shown whole, and folds them
all again. Both are per pane and per agent, so a helper's or a fork's calls
fold on their own. Calls of an archived entry can't be selected, so only the every-call
toggle shows them whole.

Only the terminal view's rows change. The model, the session log, `--json`,
ACP and exports get every line, as without the plugin.

## Settings

```toml
[fold]
head = 4        # lines kept after the header
tail = 8        # last lines kept
key = "ctrl+o"  # written as rig-tui's key bindings are, such as "alt+o"
```

The key is bound in every scope, so it works while typing, with a call
selected and in a panel. In a transcript pane that has the keys, such as a
helper's, it acts on that pane's selection. If a binding added earlier holds
the key, or the input already uses it (such as Enter or Ctrl+E), the fold key
is refused and nothing is folded, so folded lines always have a key that shows
them. The terminal view says so when it starts. An input binding of the same key added later is shadowed
by the fold key.
