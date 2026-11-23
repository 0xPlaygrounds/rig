# rig-linear

`rig_linear::LinearPlugin` shows a rig run on a Linear issue as an
[agent session](https://linear.app/developers/agents), so the people watching
the issue see the agent's plan, notes, tool calls and result.

The launcher lists it off. Turn it on with:

```sh
rig plugin enable rig_linear::LinearPlugin
```

On, it does nothing and sends nothing unless it has both a token and a
target.

## Setup

Agent sessions need an OAuth application that Linear treats as an agent:
in the application's settings, turn webhooks on with the "Agent session
events" category, as Linear's agent documentation describes. Until then
Linear answers "agent sessions disabled". The token needs the `write`
scope, which the default `[linear] scope` asks for; without it every agent
session request fails with "Invalid scope: `write` required". When Linear
refuses the session, the plugin warns once with Linear's message and sends
nothing more for the rest of the run.

## Token

- The secret `linear`: an OAuth access token, an app token, or a personal API
  key (`lin_api_…`, sent without `Bearer`).
- Or the secrets `linear-client-id` and `linear-client-secret`: an OAuth
  application's client credentials, exchanged once at startup for an app
  token, with the scopes `[linear] scope` names.

Give them in the environment as `RIG_SECRET_LINEAR`,
`RIG_SECRET_LINEAR_CLIENT_ID` and `RIG_SECRET_LINEAR_CLIENT_SECRET`. The
plugin asks for them only in a build that is given a target, so enabling
it makes rig prompt nobody who does not use Linear.

## Target

- `--linear-issue <ID>` (or `RIG_LINEAR_ISSUE`): a new agent session on this
  issue, an identifier such as `ENG-123` or the issue's id.
- `--linear-session <ID>` (or `RIG_LINEAR_SESSION`): the existing agent
  session with this id, such as one a webhook created.

An option wins over a variable, and a session over an issue.

## Settings

```toml
[linear]
api_url = "https://api.linear.app/graphql" # the OAuth token endpoint is /oauth/token on its origin
heartbeat_minutes = 10
scope = "read,write,app:assignable,app:mentionable"

[[linear.plan]]
content = "Reproduce the bug"
skill = "reproduce"          # starts when the agent reads <dir>/reproduce/SKILL.md

[[linear.plan]]
content = "Open a pull request"
command = "gh pr create"     # starts when a shell command contains this
```

A launcher that does not own `settings.toml` names a plan file in
`RIG_LINEAR_PLAN` instead: TOML with the same `[[plan]]` tables, which
replaces `[linear] plan`.

```toml
[[plan]]
content = "Reproduce the bug"
skill = "reproduce"
```

## What it sends

At startup it creates the session (or reads the existing one), prints its
link on stderr, sends the plan with every step pending, and publishes the
link as a `rig_harness::links::SessionLink`. The agent's system prompt gets a
`linear` section: the run is shown on the issue, the harness posts its
result, so it posts no comment of its own, and it ends the run with
`report_outcome`.

`report_outcome { status, summary }`, the plugin's tool, posts at once:
`done` as a response, `gave_up` as an error, `needs_input` as an
elicitation, with the summary as the body. Without it, the run's end is the
main agent's last reply as a response, its failed last turn as an error, or
"The run was stopped before it finished." when a signal or an exit stopped a
turn. A reload or a session switch posts no end.

From the main agent's effect records (a subagent's own are skipped):

- each reply's text: a thought, cut at about 2,000 characters;
- its reasoning, when the provider returned any: a passing (ephemeral)
  thought;
- each settled tool call: a passing action, `Ran`, `Read`, `Edited`,
  `Searched`, `Delegated`, `Called <tool>` for an MCP tool, else the tool's
  name, with the command, path or arguments (cut at about 300 characters),
  and on failure the exit code or error kind and the end of the output.

A plan step starts when its trigger fires and it comes after the current
step; the steps before it complete. `done` completes every step; `gave_up`
or an error cancels the current one and those after it; `needs_input`
leaves the plan as it is. Each change sends the plan whole.

Every `SessionLink` a plugin spawns becomes one of the session's external
links, and so does the pull request a shell command with `gh pr create`
printed (`https://github.com/<owner>/<repo>/pull/<n>`, as `Pull request
#<n>`).

Requests go one at a time from a task on rig's tokio runtime, at least a
second apart, each within 10 s. Passing activities queued back to back are
coalesced to the last. A 429 is sent again after its `Retry-After`. When
nothing was sent for `heartbeat_minutes`, a passing "Still working…" thought
keeps the session from going stale. A failure is a warning and never stops
or slows the run. On exit the plugin waits at most 10 s for the queue.
The plugin's own token and client secret are scrubbed from everything it
sends. Other secrets are `rig_basics::redact::RedactPlugin`'s: with it on,
they reach the records, and so Linear, as `‹SECRET_n›` placeholders.
Without it nothing else is redacted.

## Smoke test

Against a real workspace, with an issue you can write to:

```sh
RIG_SECRET_LINEAR=lin_api_… rig --linear-issue ENG-123 -p "Say hello, then report done."
```

The stderr line links to the session; the issue shows the thought, the
response and, with a plan, its steps.

## Not covered

Receiving Linear's webhooks (`created`, `prompted`, `stop`), which a future
`rig linear` front would serve as `rig acp` does for editors, and asking a
provider for reasoning summaries.
