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
running, is drawn whole, and so is a result too short to save two rows.

The key shows the call selected with Alt+Up whole, and folds it again. With
no call selected it shows every call whole, and folds them all again. The
state is per pane and per agent, so a helper's or a fork's calls fold on their
own. Calls of an archived entry can't be selected, so only the every-call
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
selected and in a panel. If a binding added earlier holds the key, the fold key
is refused with a warning and nothing is folded, so folded lines always have a
key that shows them. The same holds when a `keys.toml` entry with
`override = true` disables the fold key. A binding of the same key that is only
for the input is shadowed by the fold key.
