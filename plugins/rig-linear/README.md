# rig-linear

`rig_linear::LinearPlugin` shows a rig run on a Linear issue as an
[agent session](https://linear.app/developers/agents), so the people watching
the issue see the agent's plan, notes, tool calls and result, and can write to
the agent or stop it from the session: their prompts reach the run as Linear's
webhooks, which whatever receives them forwards to a port the run opens.

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

The secret `linear-webhook-secret` is the signing secret of the webhook whose
requests the webhook port takes (see [What it hears](#what-it-hears)).

Give them in the environment as `RIG_SECRET_LINEAR`,
`RIG_SECRET_LINEAR_CLIENT_ID`, `RIG_SECRET_LINEAR_CLIENT_SECRET` and
`RIG_SECRET_LINEAR_WEBHOOK_SECRET`. The
plugin reads them from the environment and does not prompt interactively,
so enabling it never prompts runs that do not use Linear.

## Target

- `--linear-issue <ID>` (or `RIG_LINEAR_ISSUE`): a new agent session on this
  issue, an identifier such as `ENG-123` or the issue's id.
- `--linear-session <ID>` (or `RIG_LINEAR_SESSION`): the existing agent
  session with this id, such as one a webhook created. The prompts on it
  that came after its last answer (a response, an error or a question), such
  as the request a person wrote as they mentioned or delegated to the app,
  are the run's to read: they reach the agent as the run attaches (see
  below).

An option wins over a variable, and a session over an issue.

## Claiming the issue

`--linear-claim` (or `RIG_LINEAR_CLAIM` set to `1` or `true`), off by
default, has the run claim its issue, on a new session or on an existing
one that has an issue. Once the session is there the plugin reads the app
(`viewer`), the issue's delegate and state, and its team's `started` states.
In one `issueUpdate` it then sets the app as the issue's delegate, unless
the issue has one, and moves the issue to the `started` state with the
lowest position, unless it is started, completed or canceled already. A
failure is a warning and the run goes on. Leave it off for runs an
automation started: Linear asks agents to keep those issues in triage and
leave their assignment to a person. Claiming suits an app token, whose
viewer is the app.

## Settings

```toml
[linear]
api_url = "https://api.linear.app/graphql" # the OAuth token endpoint is /oauth/token on its origin
heartbeat_minutes = 10
listen = "0.0.0.0:8787"              # the webhook port; none by default
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
replaces `[linear] plan`. Likewise `RIG_LINEAR_LISTEN` replaces
`[linear] listen`.

```toml
[[plan]]
content = "Reproduce the bug"
skill = "reproduce"
```

## What it sends

At startup it creates the session (or reads the existing one), prints its
link on stderr, posts the thought "Working on it." at once, so the session
never sits pending, sends the plan with every step pending, claims the issue
when asked, and publishes the link as a `rig_linear::SessionLink`. The agent's system prompt gets a
`linear` section: the run is shown on the issue, the harness posts its
result, so it posts no comment of its own, and it ends the run with
`report_outcome`.

`report_outcome { status, summary, options? }`, the plugin's tool, posts at
once: `done` as a response, `gave_up` as an error, `needs_input` as an
elicitation, with the summary as the body. With `needs_input`, `options`
(strings) makes the elicitation a `select`: each option, trimmed, without
blanks or repeats, at most 100 of 256 characters, goes in
`signalMetadata.options` as its own label and value. People may still answer
otherwise, and a pick comes back as a prompt. Other statuses ignore it. Without it, the run's end is the
main agent's last reply as a response, its failed last turn as an error, or
"The run was stopped before it finished." when a signal or an exit stopped a
turn. On reload the Linear session id is saved with the rig session so the
next process reattaches to it instead of opening a second session; a
session switch ends the outgoing session.

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
nothing was added to the session for `heartbeat_minutes`, a passing "Still
working…" thought
keeps the session from going stale. A failure is a warning and never stops
or slows the run. On exit the plugin waits at most 10 s for the queue.
The plugin's own supplied token and client secret (and any exchanged app
token, and the webhook secret) are scrubbed from everything it sends before
any truncation. Other
secrets are `rig_basics::redact::RedactPlugin`'s: with it on, effect records
are scrubbed before `rig-linear` reads them, so they reach Linear as
`‹SECRET_n›` placeholders. Without it nothing else is redacted.

## What it hears

The plugin reads the session's `prompt` activities once, as it creates or
attaches to the session. On a session the run creates, the prompts there
were then, such as the one Linear adds as it creates a session, are left
alone. On a session given with `--linear-session`, so are those up to its
last answer; the later ones, which no run answered yet, are told at once. A
run that reattaches to its own session after a reload tells none of those
already there. Prompts from an app, such as Linear's synthetic "Work on …",
are never told.

After that it learns of new prompts only through its webhook port, which
it opens with `[linear] listen` (or `RIG_LINEAR_LISTEN`) and the secret
`linear-webhook-secret`; with only one of them it opens no port and warns.
It prints `Linear webhooks: http://<address>/linear/webhook` on stderr.
Whatever receives Linear's webhooks (with the "Agent session events"
category), such as a function that finds the run working on the session,
forwards each `AgentSessionEvent` there as a `POST`, its body and
`Linear-Signature` header unchanged. The port answers:

- `401` unless `Linear-Signature` is the hex HMAC-SHA256 of the body with
  the webhook's signing secret, and the body's `webhookTimestamp` is within
  a minute of now;
- `400` when the body is not JSON;
- `202` for a `prompted` event, whose prompt (`agentActivity.content.body`,
  or `agentActivity.body`) is told unless it is another session's, an
  app's, blank, already told (Linear delivers a webhook again when it got no
  answer in time) or after a stop;
- `204` for any other event, which it ignores.

What a prompt told does:

- A prompt with the `stop` signal ends the run: the plan's open steps are
  canceled, the error "Stopped at the request of <name>." is posted as the
  run's end, and nothing is sent after it. The app then exits as a SIGTERM
  makes it, which stops the running turn and its tool calls, and the
  process exits with the signal code. The plugin tells nothing more.
- Any other prompt goes to the main agent as a message from the plugin
  `linear`, titled `<name> on Linear`, with the running turn's next model
  call (an idle agent starts a turn with it), and the thought "Got your
  message: <its first line, cut short>" acknowledges it. A message that
  comes after the turn's last model call waits in the agent's inbox for its
  next turn: in print mode, which exits once the turn ends, that is the next
  run on the same rig session.

## Smoke test

Against a real workspace, with an issue you can write to:

```sh
RIG_SECRET_LINEAR=lin_api_… rig --linear-issue ENG-123 -p "Say hello, then report done."
```

The stderr line links to the session; the issue shows the thought, the
response and, with a plan, its steps.

## Not covered

Receiving Linear's webhooks itself, or starting a run for a `created`
event, which is left to whatever receives them, and asking a provider for
reasoning summaries.
