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
no call selected it shows every call whole, and folds them all again.

Only the terminal view's rows change. The model, the session log, `--json`,
ACP and exports get every line, as without the plugin.

## Settings

```toml
[fold]
head = 4        # lines kept after the header
tail = 8        # last lines kept
key = "ctrl+o"  # written as rig-tui's key bindings are, such as "alt+o"
```

A key another binding takes is refused at startup with a warning, as for
any plugin's key binding.
