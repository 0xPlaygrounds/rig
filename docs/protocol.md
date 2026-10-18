# The rig wire protocol, v1

The wire every rig front speaks with the rig agent server (`rigd`). Its Rust types are in `crates/rig-protocol`.

## Framing, ids and versions

- **Framing:** one JSON object per line (LF; CRLF accepted on input), UTF-8, at most 1 MiB. There are no JSON-RPC batches. Stdout on stdio transports carries only frames; diagnostics go to stderr.
- **Every frame carries `"rig": 1`,** in both directions and in errors. A frame without it, or with a version outside `SUPPORTED`, gets `-32010`:
  - `data.update` names the side to update: `"server"` if the client's version is higher, `"client"` if lower.
  - `data.supported` lists the versions the refusing side speaks.
  - `data.next` says the command, for example `rig server restart` after `rig update`, or "update rig_client.py".
- **Ids:**
  - JSON-RPC `id`s are per connection.
  - `seq` is per session, monotonic from 1, durable and never reused.
  - `CommandSeq` numbers accepted commands per session.
  - `key` is the client's idempotency key, required on every mutating method. The CLI generates a UUID, and `--key` overrides it.
  - Integers are JSON numbers, and the spec bounds them below 2^53.
- **Versions, per direction:**
  - **Client to server:** a server accepts any frame version in `SUPPORTED`. When v2 exists, the server keeps `upgrade::v1_to_v2_request(Value) -> Result<Value, RpcError>` for one major back.
  - **Server to client:** the server writes the version negotiated in `rig/hello` (the highest both speak), downgrading through `upgrade::v2_to_v1_*`. When no common version exists, `hello` refuses and names the side.
  - **Worker and server:** the same check, plus an exact `BuildId`-independent `PROTOCOL` match. A stale worker build is refused with `data.update = "worker"` and `next: rig build`.
  - **Fixtures:** `crates/rig-protocol/tests/fixtures/v1/{requests,responses,events}.jsonl` round-trip byte-stable. `fixtures/refuse/{v0,v2}.jsonl` must produce `-32010` with the right side.
  - **Plugin events and methods** carry their own `version` in `rig/hello`'s catalog. An unknown event kind is passed through opaque, never an error.

## Methods

Each method is one `impl Method` in `rig_protocol::methods`. A row marked M is mutating and requires `key`.

| Method | Params | Result | Notes |
|---|---|---|---|
| `rig/hello` | `{token, client:{name, version}, rig:{min,max}}` | `{rig, server:{version, pid}, clientId, methods:[{name,version}], events:[{kind,version}], limits:{frame}}` | Must come first. A wrong token or uid gives `-32011`. |
| `rig/session/start` M | `{cwd, prompt?, name?, host?, build?, grant?, policy?, budget?:{tokens?,costUsd?}, context?:{facts:[String]}, unload?: "idle"\|"turn", view?:bool, key}` | `{sessionId, seq, commandSeq?}` | `cwd` is required: no process-global default (R3). `parent` is implicit when the caller is a worker. `build` is honoured for Owner callers only, and children use the parent's build. |
| `rig/session/send` M | `{sessionId, text? \| content?:[ContentBlock], mode?: "steer"\|"queue"\|"interrupt", key}` | `{commandSeq, disposition: "started"\|"steered"\|"queued"}` | `steer` lands at the next model call. `interrupt` stops the running step (`StopTurn`), then delivers. A `/cmd` line runs a slash command (`Incoming::command`). |
| `rig/session/wait` | `{sessionId, afterCommand?: CommandSeq, timeoutMs?}` | `WaitResult {status, answer?, ask?:AskInfo, error?, seq, commandSeq}` | Defaults to `afterCommand` = the latest accepted command when the call arrives, so it never returns an older idle. It returns at once on an open ask. Status is one of `answered`, `needs_approval`, `needs_answer`, `failed`, `stopped`, `timed_out`. |
| `rig/session/answer` M | `{sessionId, askId, optionId? \| text?, key}` | `{resolved:true}` | Policy-checked. The first valid answer wins, and later ones get `-32014`. |
| `rig/session/stop` M | `{sessionId, scope:"turn"\|"session", key}` | `{stopping:[SessionId]}` | `session` stops the whole tree, leaves first. Completion is the `session.stopped` event per session. |
| `rig/session/list` | `{tree?:bool, all?:bool}` | `[SessionInfo {sessionId, name, cwd, host, state, parent, children:[..], lastSeq, usage, worker:bool}]` | Owner principals see everything; a worker sees itself and its descendants. |
| `rig/session/read` | `{sessionId}` | `{info:SessionInfo, lastAnswer?, openAsks:[AskInfo], lastSeq}` | Comes from `rigd` state and never wakes a worker. |
| `rig/events/subscribe` | `{sessionId, afterSeq, follow:bool}` | `{subscription, nextSeq}` | Notifications `rig/event` (an `EventRecord`) follow, in order: replay from the journal, then live. The response is sent before the first notification. |
| `rig/events/unsubscribe` | `{subscription}` | `{}` | |
| `rig/events/read` | `{sessionId, afterSeq, limit?}` | `{events:[EventRecord], nextSeq}` | Pull form, for `rig log` without `--follow`. |
| `rig/session/attach` | `{sessionId, role:"viewer"\|"driver", takeover?:bool}` | `{driver: ClientId?}` | Takeover moves the lease and emits `driver.changed`. Detach is `rig/session/detach` or a disconnect, and it never stops the session. |
| `rig/session/detach` | `{sessionId}` | `{}` | |
| `rig/session/reload` M | `{sessionId, build, key}` | `{commandSeq}` | `reload.*` events follow. |
| `x/<plugin>/<verb>` | `{sessionId, …}` | plugin's | Routed to that session's worker, started if needed. |

**Server-to-client notifications** are `rig/event` (an `EventRecord`) and `rig/lagged` (`{sessionId, nextSeq}`, sent only if a subscription is cancelled for exceeding its queue *and* the transport closes). The native protocol has **no server-to-client requests**: an approval is an event, answered with `rig/session/answer`. That keeps a 20-line Python client possible. `rig acp` turns those events into ACP's `session/request_permission`.

**Worker channel methods** are private, and only a `Principal::Session` connection holding the worker token may call them:
- `worker/hello {rig, sessionId, build, applied: CommandSeq, logs:[{log, len}]}` → `{knownThrough:{log: index}, replay:[Command]}`
- `worker/events {events:[{kind, data, source?:SourceKey}]}` → `{}`, one batch per frame, so deltas are coalesced per 16 ms frame
- `worker/applied {through}` and `worker/quiescent {quiescent:bool}`
- `rigd` sends the notification `worker/command {commandSeq, command}`, where `command` is one of `Send`, `Stop`, `Answer`, `Plugin{method,params,reply}` or `Reload`
- `worker/reply {reply, result}` answers a `Plugin` call

Workers start and drive children with the normal `rig/session/*` methods. Their principal limits them to their own descendants.

## Event kinds

`data` shapes are fixed in `rig_protocol::event`. A durable kind comes from the worker's saved log, with a `source` key. A live kind goes straight to the journal.

| kind | data | class |
|---|---|---|
| `session.started` | `{cwd, name, host, parent, build, grant}` | rigd |
| `command.accepted` | `{commandSeq, key, method, by: ClientId, text?}` | rigd |
| `worker.ready` / `worker.exited` | `{pid, build}` / `{code, signal, restarting, log}` | rigd |
| `turn.started` | `{agentId, turnId}` | durable |
| `message.delta` | `{agentId, messageId, text?, reasoning?}` | live |
| `message` | `{agentId, messageId, role, text}` (the authoritative final text) | durable |
| `conversation.change` | `{agentId, log, index, record}` (`record` = `rig_core::conversation::ChangeRecord` JSON, for mirrors) | durable |
| `call.started` / `call.progress` / `call.ended` | `{agentId, callId, tool, args}` / `{callId, text}` / `{callId, ok, result, error?}` | durable / live / durable |
| `ask.opened` | `{askId, agentId, agentName, kind:"approval"\|"question", title, body, options:[{optionId, label, kind}], deadlineMs, answerers}` | durable |
| `ask.resolved` | `{askId, by:"client:<id>"\|"rule"\|"timeout"\|"parent", optionId?, text?}` | rigd |
| `turn.ended` | `{agentId, turnId, status:"answered"\|"failed"\|"stopped", answer?, error?, through: CommandSeq}` | durable |
| `usage` | `{tokens, costUsd?, tree:{tokens, costUsd}}` | rigd (summed) |
| `notice` | `{level, text}` | live |
| `status` | `{items:[{id, text}]}` | live |
| `child.started` / `child.ended` | `{sessionId, name}` / `{sessionId, status}` | rigd |
| `driver.changed` | `{driver: ClientId?}` | rigd |
| `budget.exhausted` | `{tokens, costUsd}` | rigd |
| `reload.started` / `reload.done` / `reload.failed` | `{build}` / `{build}` / `{build, error, kept}` | rigd |
| `session.idle` / `session.stopped` / `session.faulted` | `{}` / `{}` / `{reason, log}` | rigd |
| `x/<plugin>/<name>` | plugin-defined `{version, …}` | durable or live, as published |
