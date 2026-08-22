# agent_no_tokio

Runs a rig agent on Bevy's `AsyncComputeTaskPool` with no tokio dependency in
this crate's manifest.

`Agent::run_channel` splits a run into a `RunHandle` (its `RunId` and a cancel
handle), a runtime-agnostic `RunFuture`, and a bounded `RunEvents` feed whose
every `RunEvent` is stamped with that id. The future is spawned on the pool;
the main thread acts as a frame loop, draining events once per tick with the
non-blocking `RunEvents::try_drain` and checking the task with
`bevy_tasks::futures::check_ready` — the shape a Bevy system or any other
tick-driven host would use.

The mapping to an ECS is direct: the `ActiveRun` struct (handle + task + feed)
is the component a spawned run would carry; the loop body is the system that
queries every such component, drains its feed, and routes each event by
`event.run`; `RunHandle::abort` is what that system calls when the entity goes
away.

```sh
OPENAI_API_KEY=… cargo run -p agent_no_tokio
# abort through the handle after 5 events and see PromptCancelled:
OPENAI_API_KEY=… RIG_EXAMPLE_ABORT_AFTER=5 cargo run -p agent_no_tokio
```

Tokio appears only transitively, inside `rig-reqwest`, which drives the HTTP
wire on a private runtime:

```sh
cargo tree -p agent_no_tokio -e normal -i tokio
```
