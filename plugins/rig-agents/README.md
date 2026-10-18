# rig-agents

Helper agents for the model. Each helper is a **child session** of the rig agent
server (`rigd`), with its own worker process. The parent drives it through the same
channel and methods (`rig/session/*`) that `rig start`/`send`/`wait` and any
outside driver use. On by default.

| Tool | What it does |
|---|---|
| `start_agent {task, name?, cwd?, host?, tools?, dirs?, facts?, view?}` | Starts a helper on `task` and returns its id at once. The helper works in the parent's folder unless `cwd` or `host` says otherwise, with the parent's tools and folders or fewer. |
| `message_agent {id, text, interrupt?}` | Sends a correction or more work. The helper reads it before its next model call, or at once with `interrupt`. |
| `wait_agent {ids, timeout_s?}` | Waits until one helper answers, needs an approval or an answer, fails or stops, and returns that. |
| `answer_agent {id, ask, allow? \| option? \| text?}` | Answers a helper's approval or question. |
| `stop_agent {id}` | Stops a helper and every helper it started. |
| `list_agents {}` | Lists your helpers, and theirs, with their state. |

## How it behaves

- **A message is a message, and the final answer is the report.** This holds by
  construction:
  - `wait_agent` settles only on a `turn.ended` whose `through` covers the last
    message the helper was sent, on an open ask, or on the helper's end;
  - a helper's progress messages never settle a wait;
  - each report is given once. The record of delivered reports is saved, so a
    restart doesn't repeat it.
- **Context hand-over.** A helper gets:
  - its `cwd` (by default the parent's, never the process's);
  - its host;
  - its narrowed grant;
  - `facts`, which carries what the parent learned.

  The project's AGENTS.md comes from the harness, as for any session.
- **Least privilege.** The server narrows the requested grant with
  `Grant::narrow`, and a request for more is refused (`-32015`):
  - tools must be a subset of the parent's;
  - folders must be under the parent's (writable only under writable ones);
  - network only if the parent has it;
  - up-front rules only if the parent's own rules already answer them.

  A helper with narrower folders runs sandboxed: Landlock on Linux,
  `sandbox-exec` on macOS. Its file tools and its shell are both confined, so it
  can't read rig's `run/token`, other sessions or the rest of the home folder, even
  through a symlink. A grant the platform can't enforce is refused (`-32017`),
  never widened.
- **Approvals.**
  - Every ask names the session and agent asking.
  - A helper's question goes to its parent, which answers with `answer_agent`.
  - A helper's approval goes to the nearest ancestor whose own up-front rules
    allow that call. Anything wider escalates to the user (`rig answer`, the TUI).
  - The user may answer any helper's ask.
  - An ask nobody answers is denied after the policy's timeout (600 s).
- **Limits.**
  - At most 8 helpers of a tree run at once, or the root grant's `children`; a
    parent's own `children` caps its direct helpers. Starts over the limit are
    accepted and queued.
  - A tree is at most 4 deep.
  - A token and cost budget (`rig start --budget-tokens`) is shared by the whole
    tree. When it runs out, every running turn in the tree stops.
- **Stopping.** `stop_agent` (and `rig stop` of any ancestor) stops a tree leaves
  first. Sessions keep their history.
- **Live view.** Inside fux, `view: true` opens the helper as a live TUI in a fux
  pane, through `rig attach`, so the user can watch it or take it over. Only an
  unrestricted parent can do this.

Without an agent server (an old-style session), the tools refuse and say so.

## Compared with `SubagentsPlugin`

`rig_basics::subagents::SubagentsPlugin` (in-process subagents in the parent's
World) is an opt-in proof of concept, off by default. This crate supersedes it and
doesn't build on it.
