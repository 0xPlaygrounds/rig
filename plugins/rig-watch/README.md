# rig-watch

`rig_watch::WatchPlugin` wakes the rig agent when something it waits on
happens. It is off by default. Turn it on with `rig plugin enable
rig_watch::WatchPlugin`, in `/plugins`, or in `$RIG_HOME/settings.toml`:

```toml
[plugins]
enable = ["rig_watch::WatchPlugin"]
```

An agent that directs other processes (workers in terminal panes, reviewers,
a build) ends its turn and then hears nothing until the user types. With the
`watch` tool it says what it waits on, each a fact of a file:

| kind | holds when |
|---|---|
| `exists` | the file or folder is there |
| `gone` | it is not |
| `has` | the file's first 4 KiB hold a text |
| `lacks` | they do not; a file that is not there lacks every text |

Every five seconds the plugin looks. A watch whose fact holds fires once: the
agent gets a message that names it, which starts a turn if the agent is idle.
An agent in a turn is looked at when the turn is over, so no wake waits
behind its work. While nothing fires, a message comes every `every_minutes`
(10 unless the call says otherwise), so that a worker that hangs is noticed.
A wake starts with the local time it was sent at:

```
[watch] 2026-11-23 17:19:14 -08:00
Fired:
  worker report: exists /work/reports/07.md
Still watched:
  worker idle: has /run/panes/worker/title "idle"
```

The agent is woken until it calls the tool with `done: true`. `/watch` lists
what is watched and `/watch off` ends it. An arming also ends by itself after
60 wakes. What is watched is saved with the session, so it goes on after
`/reload`, a restart and `/resume`.

A wake is a turn, and costs what a turn costs. The plugin reads no file
beyond its first 4 KiB and runs no command: what has no file, such as a CI
run, is left to the wake that comes when nothing fired.
