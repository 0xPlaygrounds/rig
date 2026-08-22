# agent_no_tokio

Runs a rig agent on Bevy's `AsyncComputeTaskPool` with no tokio dependency in
this crate's manifest.

`Agent::run_channel` splits a run into a runtime-agnostic future and a bounded
`RunEvents` feed. The future is spawned on the pool; the main thread acts as a
frame loop, draining events with the non-blocking `RunEvents::try_next` and
checking the task with `bevy_tasks::futures::check_ready` — the shape a Bevy
system or any other tick-driven host would use.

```sh
OPENAI_API_KEY=… cargo run -p agent_no_tokio
```

Tokio appears only transitively, inside `rig-reqwest`, which drives the HTTP
wire on a private runtime:

```sh
cargo tree -p agent_no_tokio -e normal -i tokio
```
