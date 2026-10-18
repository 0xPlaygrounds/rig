# The rig wire protocol, v1

The wire every rig front speaks with the rig agent server, `rigd`: the TUI, `rig -p`, the CLI verbs, child-agent tools, `rig acp`, Python and remote drivers. It is JSON-RPC 2.0, one JSON object per line, with rig's own guarantees on top: a per-session event `seq`, idempotency `key`s, resume after a lost connection, and backpressure.

- Rust types: `crates/rig-protocol` (`rig_protocol::methods`, `rig_protocol::event`). A Rust client: `crates/rig-client`.
- A Python client (standard library only): `clients/python/rig_client.py`, with `clients/python/examples/drive.py`.
- Version fixtures: `crates/rig-protocol/tests/fixtures/`. The conformance suite any client can run: `tests/conformance/`.

## A session in six lines

```
→ {"jsonrpc":"2.0","rig":1,"id":1,"method":"rig/hello","params":{"token":"…","client":{"name":"me","version":"1"},"rig":{"min":1,"max":1}}}
← {"jsonrpc":"2.0","rig":1,"id":1,"result":{"rig":1,"server":{"version":"0.44.0","pid":4242},"clientId":"c-1","methods":[…],"events":[…],"limits":{"frame":1048576}}}
→ {"jsonrpc":"2.0","rig":1,"id":2,"method":"rig/session/start","params":{"cwd":"/work/repo","prompt":"fix the failing test","key":"6f1c…"}}
← {"jsonrpc":"2.0","rig":1,"id":2,"result":{"sessionId":"20261018-120000-a1b2","seq":2,"commandSeq":1}}
→ {"jsonrpc":"2.0","rig":1,"id":3,"method":"rig/session/wait","params":{"sessionId":"20261018-120000-a1b2"}}
← {"jsonrpc":"2.0","rig":1,"id":3,"result":{"status":"answered","answer":"Fixed: the test now passes.","seq":42,"commandSeq":1}}
```

## Transports and connecting

| Transport | Where | Who may connect |
|---|---|---|
| Unix socket | `$RIG_HOME/run/rig.sock` | The socket is 0600 in a 0700 `run/` directory, both checked for owner and type before binding. The server checks the peer's uid (`SO_PEERCRED`, `getpeereid`). |
| Windows named pipe | `\\.\pipe\rig-<16 hex digits>`: a stable FNV-1a hash of the user's SID and the socket path above | The pipe's DACL grants only the user's SID. It is created with `FILE_FLAG_FIRST_PIPE_INSTANCE` (a second server can't take the name) and `PIPE_REJECT_REMOTE_CLIENTS`. |
| stdio | `rig connect --stdio` bridges its stdin/stdout to the local server | Whoever can run it: a remote driver reaches it through `[remote.hosts].NAME`'s transport (ssh, `koh exec`), which authenticates. The bridge fills in its own host's token. |

On every local transport the first request must be `rig/hello` carrying `run/token` (32 random bytes, 0600). A sandboxed child's shell can't read the token, so it can't reach the owner socket even though it runs as the same user. A worker authenticates with its own per-worker token instead, which grants only its session and that session's descendants.

**Starting the server.** The first client that finds no live server starts one:
1. Connecting fails because the socket (or pipe) is missing or refuses.
2. If `run/server.lock` is free, the client starts the rig binary with `RIG_ROLE=server`, `RIG_HOME` set, stdin and stdout closed, stderr appended to `run/server.log`, in its own process group. If the lock is held, a server is already starting.
3. The server takes an exclusive lock on `run/server.lock` for its whole life and exits quietly when it can't. It writes `run/token` before it binds, so a socket that accepts means the token is there.
4. The client retries the connection for up to 10 s.

`rigd` exits after `server.idle_exit` (10 s) with no clients and no workers. Clients that keep a connection open keep it alive.

## Framing

- One JSON object per line: UTF-8, LF-terminated (CRLF is accepted on input), at most 1 MiB (`MAX_FRAME`) without the line ending. Blank lines are skipped.
- No JSON-RPC batches: a line starting with `[` is refused with `-32600`.
- A line over 1 MiB is refused with `-32020` and skipped to its end. A line that isn't UTF-8 or JSON is refused with `-32700`. In both cases the connection stays open and the next line reads normally.
- On stdio transports stdout carries only frames; diagnostics go to stderr.
- Integers (ids, `seq`, `commandSeq`) are JSON numbers below 2^53.
- Field names are camelCase, except `EventRecord.at_ms` and `Policy`'s `timeout_s`/`on_timeout`.

## Versions

**Every frame carries `"rig": 1`**, in both directions, errors included. A frame without it, or with a version this side doesn't speak, gets `-32010` with:
- `data.update`: the side to update. `"server"` when the peer's version is newer than anything the server speaks, `"client"` when older or missing, `"worker"` for a stale worker build.
- `data.supported`: the versions the refusing side speaks.
- `data.next`: what to do, such as "install a newer rig where the server runs, then `rig server stop` so the new one starts", or `rig build` for a worker.

The refusal answers the request it refuses (it carries that request's `id` when one could be read).

**Per direction:**
- **Client to server:** the server accepts any version in its `SUPPORTED` range. Each request's params are brought up to the server's shape by `rig_protocol::upgrade::request_up(version, params)`. In v1 that is the identity for v1 and a refusal otherwise. When v2 exists, the server keeps the v1-to-v2 step for one major back.
- **Server to client:** `rig/hello` carries the client's range `{min, max}`. The server answers with the highest version both speak (`upgrade::negotiate`) and writes everything on that connection at that version, brought down by `upgrade::down`. With no common version, `hello` is refused, naming the side to update.
- **Worker and server:** the same check. A worker of an older build is refused with `update: "worker"` and `next: "rig build"`.
- **Plugins:** each plugin method and event kind has its own `version` in `rig/hello`'s `methods` and `events` catalogs. An event kind a client doesn't know is passed through opaque, never an error.

**Fixtures** (checked in, run by `cargo nextest run -p rig-protocol fixtures`):
- `fixtures/v1/requests.jsonl`, `responses.jsonl`, `events.jsonl`: every method's params and result and every built-in event kind. Each line reads into its Rust type and writes back byte for byte.
- `fixtures/refuse/v0.jsonl` (versions 0 and missing) must be refused naming the client, and `fixtures/refuse/v2.jsonl` naming the server.

## Ids

| Id | Scope | Notes |
|---|---|---|
| JSON-RPC `id` | per connection | A number or a string. A reconnecting client may resend a request with the same id on the new connection. |
| `sessionId` | per host | Today's session id format, `YYYYMMDD-HHMMSS-xxxx`. |
| `seq` | per session | The event's position in the session's journal: from 1, contiguous, durable, never reused. |
| `commandSeq` | per session | The position of an accepted command (`start` with a prompt, `send`, `answer`, `stop`, `reload`). |
| `key` | per session | The client's idempotency key, required on every mutating method. The CLI generates a UUID; `--key` overrides it. |
| `clientId` | per connection | Given by `rig/hello`. A reconnect says hello again and gets a new one. |
| `subscription` | per connection | Given by `rig/events/subscribe`. |

## Methods

Each method is one `impl Method` in `rig_protocol::methods`: the type is the params, `Method::Result` the answer. A row marked **M** is mutating and requires `key`.

| Method | Params | Result | Notes |
|---|---|---|---|
| `rig/hello` | `{token, client:{name, version}, rig:{min, max}}` | `{rig, server:{version, pid}, clientId, methods:[{name, version}], events:[{kind, version}], limits:{frame}}` | Must come first. A wrong token or uid gives `-32011`. `rig` in the result is the negotiated version. |
| `rig/session/start` **M** | `{cwd, prompt?, name?, host?, build?, grant?, policy?, budget?:{tokens?, costUsd?}, context?:{facts:[String]}, unload?:"idle"\|"turn", view?, key}` | `{sessionId, seq, commandSeq?}` | `cwd` is required. A worker caller's session is the implicit parent. `build` is honoured for owner callers only; children use the parent's build. `commandSeq` is set when a `prompt` was accepted. |
| `rig/session/send` **M** | `{sessionId, text? \| content?:[ContentBlock], mode?:"steer"\|"queue"\|"interrupt", key}` | `{commandSeq, disposition:"started"\|"steered"\|"queued"}` | Returns at admission. `steer` (the default while a turn runs) lands at the next model call, `queue` after the turn, `interrupt` stops the running step and then delivers. A `/cmd` line runs a slash command. `content` holds ACP content blocks. |
| `rig/session/wait` | `{sessionId, afterCommand?, timeoutMs?}` | `{status, answer?, ask?, error?, seq, commandSeq}` | Waits for the outcome of command `afterCommand`, by default the latest accepted when the call arrives, so it never returns an older idle. It returns at once while an ask is open. `status` is `answered`, `needs_approval`, `needs_answer`, `failed`, `stopped` or `timed_out`. |
| `rig/session/answer` **M** | `{sessionId, askId, optionId? \| text?, key}` | `{resolved:true}` | Checked against the session's policy. The first valid answer wins; later ones get `-32014`. |
| `rig/session/stop` **M** | `{sessionId, scope:"turn"\|"session", key}` | `{stopping:[sessionId]}` | `session` stops the whole tree, leaves first. Each session's `session.stopped` event marks it done. |
| `rig/session/list` | `{tree?, all?}` | `[{sessionId, name, cwd, host, state, parent, children, lastSeq, usage:{tokens, costUsd?}, worker}]` | Owner principals see every session; a worker sees itself and its descendants. `state` is `idle`, `running`, `needs_approval`, `needs_answer`, `stopping`, `stopped` or `faulted`. |
| `rig/session/read` | `{sessionId}` | `{info, lastAnswer?, openAsks:[AskInfo], lastSeq}` | From `rigd`'s state; never wakes a worker. |
| `rig/events/subscribe` | `{sessionId, afterSeq, follow}` | `{subscription, nextSeq}` | `rig/event` notifications follow the response: the journal after `afterSeq`, then, when `follow`, every new event. `nextSeq` is the seq the next new event will get. |
| `rig/events/unsubscribe` | `{subscription}` | `{}` | |
| `rig/events/read` | `{sessionId, afterSeq, limit?}` | `{events:[EventRecord], nextSeq}` | The pull form, for `rig log` without `--follow`. |
| `rig/session/attach` | `{sessionId, role:"viewer"\|"driver", takeover?}` | `{driver: clientId?}` | `takeover` moves the driver lease and journals `driver.changed`. |
| `rig/session/detach` | `{sessionId}` | `{}` | Detaching, or disconnecting, never stops the session. |
| `rig/session/reload` **M** | `{sessionId, build, key}` | `{commandSeq}` | `reload.started`, then `reload.done` or `reload.failed` follow. |
| `x/<plugin>/<verb>` | `{sessionId, …}` | the plugin's | Routed to that session's worker, which is started if needed. |

`AskInfo` is `{sessionId, askId, agentId, agentName, kind, title, body, options, deadlineMs, answerers}`: an `ask.opened` event's data plus the session.

### Notifications

The server sends two notifications to clients, and no requests:
- `rig/event`: `{subscription, session, seq, at_ms, kind, data, source?}`, one journaled event (an `EventRecord`) and the subscription it belongs to. One connection may hold several subscriptions on one session, each with its own replay, so a client routes by `subscription`.
- `rig/lagged`: `{sessionId, nextSeq}`, only when a subscription is cancelled because its queue overflowed and the transport is then closed. The client reconnects and resubscribes from its last seq.

An approval or a question is an event (`ask.opened`), answered with `rig/session/answer`. With no server-to-client requests, a client is a read loop and a write call: the Python client is under 200 lines. `rig acp` turns these events into ACP's `session/request_permission`.

### The worker channel

Private methods, callable only by a worker's own connection (its worker token):
- `worker/hello {rig, sessionId, build, applied, logs:[{log, len}]}` → `{knownThrough:{<log>: index}, replay:[{commandSeq, command}]}`. `knownThrough` is the last record of each saved log already journaled; the worker re-exports every record after it. `replay` is every accepted command after `applied`.
- `worker/events {events:[{kind, data, source?:{log, index}}]}` → `{}`: one batch per worker frame, so deltas are coalesced per frame.
- `worker/applied {through}`, `worker/quiescent {quiescent}`, `worker/reply {reply, result | error}` → `{}`.
- The notification `worker/command {commandSeq, command}`, from `rigd`, where `command` is one of `{"send": Send}`, `{"stop": Stop}`, `{"answer": Answer}`, `{"plugin": {method, params, reply}}` or `{"reload": Reload}`.

Workers start and drive children with the ordinary `rig/session/*` methods; their principal limits them to their own descendants.

## Errors

Every error is `{code, message, data?}`. `data.next`, when present, is the exact command or action that fixes it.

| Code | Name | When | `data` |
|---|---|---|---|
| -32700 | parse | the line isn't UTF-8 or JSON | |
| -32600 | invalid request | a batch, or not a request | |
| -32601 | method not found | an unknown method | |
| -32602 | invalid params | params that don't fit the method | |
| -32603 | internal | the server's bug | |
| -32010 | version | no shared wire version | `update`, `supported`, `next` |
| -32011 | unauthorized | a bad or missing token, a wrong uid, or a principal that may not call this | |
| -32012 | not found | an unknown session, ask or agent id | `next` |
| -32013 | key conflict | a `key` reused with a different command | `next` |
| -32014 | already resolved | the ask was answered or expired | `by`, `choice` |
| -32015 | grant wider | a child asked for more than its parent has | `next` |
| -32016 | overloaded | more than 64 requests in flight on the connection | retry with the same `key` |
| -32017 | sandbox unavailable | a narrower grant this platform can't enforce | `next` |
| -32018 | worker failed | the worker couldn't start, or crashed repeatedly | `log` |
| -32019 | limit | the depth limit (4) | `next` says what to stop |
| -32020 | frame too large | a line over 1 MiB | |

## Event kinds

`data` shapes are fixed in `rig_protocol::event` (`Event::decode`, `Event::encode`). The class says where an event comes from: **rigd** events are written by the server, **durable** ones are derived from the worker's saved log and carry a `source`, and **live** ones go straight from the worker to the journal.

| kind | data | class |
|---|---|---|
| `session.started` | `{cwd, name, host, parent, build, grant}` | rigd |
| `command.accepted` | `{commandSeq, key, method, by, text?, command?}` | rigd |
| `worker.ready` / `worker.exited` | `{pid, build}` / `{code, signal, restarting, log}` | rigd |
| `turn.started` | `{agentId, turnId}` | durable |
| `message.delta` | `{agentId, messageId, text?, reasoning?}` | live |
| `message` | `{agentId, messageId, role, text}`: the authoritative final text | durable |
| `conversation.change` | `{agentId, log, index, record}`: `record` is a `rig_core::conversation::ChangeRecord`, for mirrors | durable |
| `call.started` / `call.progress` / `call.ended` | `{agentId, callId, tool, args}` / `{callId, text}` / `{callId, ok, result, error?}` | durable / live / durable |
| `ask.opened` | `{askId, agentId, agentName, kind:"approval"\|"question", title, body, options:[{optionId, label, kind}], deadlineMs, answerers}` | durable |
| `ask.resolved` | `{askId, by:"client:<id>"\|"rule"\|"timeout"\|"parent", optionId?, text?}` | rigd |
| `turn.ended` | `{agentId, turnId, status:"answered"\|"failed"\|"stopped", answer?, error?, through}`: `through` is the last command the turn answered | durable |
| `usage` | `{tokens, costUsd?, tree:{tokens, costUsd?}}` | rigd |
| `notice` | `{level, text}` | live |
| `status` | `{items:[{id, text}]}` | live |
| `child.started` / `child.ended` | `{sessionId, name}` / `{sessionId, status}` | rigd |
| `driver.changed` | `{driver}` | rigd |
| `budget.exhausted` | `{tokens, costUsd?}` | rigd |
| `reload.started` / `reload.done` / `reload.failed` | `{build}` / `{build}` / `{build, error, kept}` | rigd |
| `session.idle` / `session.stopped` / `session.faulted` | `{}` / `{}` / `{reason, log}` | rigd |
| `x/<plugin>/<name>` | the plugin's, with its `version` in the hello catalog | durable or live, as published |

An option's `kind` is ACP's: `allow_once`, `allow_always`, `reject_once` or `reject_always`.

## Delivery guarantees

### What "exactly once" means here

**Commands are applied exactly once per `key`.** The boundary is the server's journal:
1. `rigd` journals `command.accepted` (with the command) and fsyncs it **before** it answers. Once a client has an answer, the command survives a crash of `rigd` or the worker.
2. A request repeated with the same `key` and the same command gets the original answer and is not applied again. The same `key` with a different command gets `-32013`. So a client that lost its connection before the answer resends with the same key, and the command is applied once whether or not the first attempt reached the journal.
3. The worker applies the command and saves `Applied(commandSeq)` in the same agent-log append as the message or effect it produced. That append is atomic. On restart, `worker/hello.applied` tells `rigd` which commands to replay, and a command at or below it is never applied twice.

What is *not* covered: a request whose answer never arrived **and** that the client never resends, with no key. The server can't tell that from a new request. Every mutating method therefore requires a key.

**Events are delivered exactly once and in order at the reader.** The boundary is the reader's last seen `seq`:
1. Every event is journaled with its `seq` before it is sent to anyone. The journal keeps every event, deltas included, for the session's lifetime, so a replay from any `afterSeq` has no gap.
2. The server sends a subscription's events in `seq` order. A client drops any event with `seq` at or below the last one it delivered, so a replay that overlaps (after a reconnect) yields nothing twice.
3. Durable worker events are exported only after the worker's own log save and carry a `source` (`{log, index}`). After a worker crash `rigd` sends `knownThrough`, the worker re-exports every saved record after it, and a record whose `source` was already journaled is dropped. A crash between the save and the export loses nothing and duplicates nothing.
4. Live deltas journaled before a crash stay in the journal. `worker.exited` follows them, which tells a reader to drop the partial text of an unfinished message; the authoritative `message` comes later.

What is *not* covered: a delta that was produced but not yet sent when the worker crashed is lost. It was never journaled and never had a `seq`; the final `message` carries the whole text.

### Resume

A client that loses its connection reconnects with backoff (20 ms doubling to 2 s), says hello again, resends every request still waiting for its answer (keys make that safe) and resubscribes every stream with `afterSeq` = the last `seq` it received. `rig-client` and `rig_client.py` both do this; a call or stream gives up after 30 s without a server (`ClientError::Unavailable`, exit 7). A local client also restarts `rigd` if it is gone. A crashed `rigd` reloads its sessions from their journals and restarts the workers of sessions that were running.

### Backpressure

- Each connection has a bounded out-queue (1,024 frames or 4 MiB). Fan-out to it never blocks the server, a session or other clients.
- When a subscription's queue is full it pauses with its cursor held, and refills from the journal file once the client drains. A slow client costs only its own disk reads.
- Inbound, each connection has a bounded queue of 256 frames: only the flooding client waits. More than 64 requests in flight gets `-32016`; retry with the same key.

## Exit codes

Every waiting command (`rig wait`, `rig -p`, the CLI verbs) ends with one of `rig_protocol::Exit`:

| Code | Meaning |
|---|---|
| 0 | answered |
| 1 | failed |
| 2 | usage: a bad argument, an unknown id or method |
| 3 | needs approval |
| 4 | needs an answer |
| 5 | stopped |
| 6 | not allowed: unauthorized, already resolved, a wider grant, no sandbox |
| 7 | the server is unavailable, or no shared version |
| 124 | timed out |
| 130 | interrupted |
