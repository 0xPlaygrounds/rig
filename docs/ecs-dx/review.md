| Reviewer workflow | Improvement visible in the diff | Evidence |
| --- | --- | --- |
| Start an application | [Agent/tool example](../../crates/rig-ecs/examples/agent_with_tools.rs) shows installation, registration and schedule driving; [provider construction](../../crates/rig-ecs/examples/provider_construction.rs) uses a real Gemini provider with an explicit in-memory transport. | Both examples executed offline; provider example asserts its exact answer within 5,000 bounded host ticks. Crate startup doctest compiled. |
| Configure and submit | [Agent/Prompt](../../crates/rig-ecs/src/commands.rs) replace the setup tuple, manual grants and positional options. | [commands tests](../../crates/rig-ecs/tests/commands.rs) compare captured requests with an independently constructed graph and test both parameter-buffer orders. |
| Extract a helper | Ordinary `Commands`, `Handlers` and `Query<RunView>` remain usable without handwritten lifetimes or future boxing. | `one_system_registers_and_submits_with_commands_before_handlers` calls the extracted helper twice, adds application components and records IDs in a resource. Typed registration inference is covered in [bus_scene](../../crates/rig-ecs/tests/bus_scene.rs). |
| Read and stream results | [RunView/RunInfo](../../crates/rig-ecs/src/inspect.rs), per-effect [StreamText](../../crates/rig-ecs/src/stream.rs), and cohort-scoped [best_of_n](../../crates/rig-ecs/examples/best_of_n.rs). | Lifecycle/stream tests exercise missing entities, unknown phases and interleaved Unicode; examples executed offline. |
| Approve an operation | [Typed approval](../../crates/rig-ecs/src/approval.rs) and [host input](../../crates/rig-ecs/examples/human_in_the_loop.rs) replace blocking stdin in a schedule. | Approval tests verify other-run progress, exact ticket/proposal identity and competing holds; input tests cover absent/partial input, EOF and shutdown. Scripted and stdin paths executed. |
| Cancel, steer and fork | [Lifecycle operations](../../crates/rig-ecs/src/lifecycle.rs) check timing/liveness and expose structured failures. | Commands, lifecycle and [steering](../../crates/rig-ecs/tests/steer_hooks.rs) tests cover retry feedback, patches, repeated/stale operations, invalid fork cuts and observer removal. |
| Customize the runtime | Named private query data and internal worker bookkeeping; ordinary graph components, sets and witness access remain. | [External-style bus tests](../../crates/rig-ecs/tests/bus_api.rs), retained architecture guards, native delivery/worker tests and executed WASM tests. |
| Restore or replay | [WorldScene::save/load](../../crates/rig-ecs/src/agent/scene.rs), consistent with RunScene and bus Scene; [Replay::register_in](../../crates/rig-ecs/src/bus/replay.rs) removes a nested host callback/result. | Migrated scene, memory, reflection, steering, batch and replay tests retain their original assertions; current verification status is recorded in [progress](progress.md). |

This implementation review guide is accompanied by the API/example inventories.
The PR description records completed verification, the tested implementation
revision and downstream limitations; [progress](progress.md) retains the work
chronology. Parent reference:
[`ad8e9b69f857992f7078d66555bce5c4db751df5`](https://github.com/0xPlaygrounds/rig/commit/ad8e9b69f857992f7078d66555bce5c4db751df5).
Relative code links refer to the DX tree. The PR body records the tested commit
and execution-time parent.

## Installation and registration

[Parent example support](https://github.com/0xPlaygrounds/rig/blob/ad8e9b69f857992f7078d66555bce5c4db751df5/crates/rig-ecs/examples/support/mod.rs)
hid App construction, both World installers, ticking and a failure observer in
`support::app()`. `Handlers::with(world, |handlers| handlers.register(...))`
required two error layers outside a system.

Current examples show these obligations directly:

```rust
let mut app = App::new();
install(app.world_mut(), ServingPolicy::default())?;
app.add_systems(Update, run_to_quiescence);
let model = Handlers::register_in(app.world_mut(), "model", model)?;
```

See the [complete provider example](../../crates/rig-ecs/examples/provider_construction.rs)
for actual imports, client/adapter construction, error handling and a bounded host
loop. Its transport is synthetic; it is not a genuine provider recording. The
bus initializes shared task pools if the host has not already initialized them.
App remains a dev/optional dependency, not a requirement of the base library.

## Agents, tools and named submission

The parent support function assembled `Owner`, `Preamble`, `Temperature`,
`MaxTokens`, `AdditionalParams`, `ToolChoiceSpec`, `Output`, `DefaultMaxTurns`,
`MaxTurns`, `InvalidCalls` and `UsesModel`. Callers separately spawned ordered
`Grant` children. The [current calculator](../../crates/rig-ecs/examples/agent_with_tools.rs)
uses:

```rust
let agent = Agent::new(model)
    .owner("calculator")
    .preamble(PREAMBLE)
    .tools([add, subtract])
    .max_turns(2)
    .spawn(world)?;
Prompt::new(agent, "Calculate 2 - 5.").spawn(world)?;
```

[Parent streaming submission](https://github.com/0xPlaygrounds/rig/blob/ad8e9b69f857992f7078d66555bce5c4db751df5/crates/rig-ecs/examples/streaming_ui.rs)
queued a World closure around `spawn_run(world, agent, &history, text, true, None)`.
The current example submits `Prompt::new(agent, text).history(history).streaming()`.
The old helper remains an explicit advanced graph primitive; normal callers use
Prompt for validation and structured errors.

These owned values are consumed into the graph. They are not another runtime or
configuration store. Deferred operations reserve IDs; bindings and components
become visible on application. Registration inspection deliberately reports
applied state, including during replacement/removal. A registration ID can still
be used by construction commands in the same system before its own buffer runs.

Construction observes Bevy's synchronous observer flushing. Checked initialization
revalidates targets and removes its constructor-owned graph on failure. It does
not roll back unrelated application actions. Children are attached before their
payload is inserted, so relationship observers may see an attached child without
its Grant/Utterance yet; payload observers see an established relationship.

## Results, streaming and cohorts

The parent's streaming display used one `Local<usize>` for every queried stream
and indexed `stream.text[shown..]`. The [current display](../../crates/rig-ecs/examples/streaming_ui.rs)
uses `text.read(effect, stream)` with `stream: &Streamed` and resets a removed stream's cursor.
It preserves the public stream events/errors/usage. Arbitrary same-length text
replacement is not automatically detected; callers explicitly forget replaced
streams. The per-effect cursor handles interleaving and UTF-8 boundaries.

[best_of_n](../../crates/rig-ecs/examples/best_of_n.rs) records its submitted cohort
and reads `Query<RunView>` for those IDs. Unrelated runs cannot complete the demo,
failed/missing candidates are handled, and the selection rule remains an explicit
demo policy. Views derive status from current components; contradictory markers
report Unknown rather than inventing a stall diagnosis.

## Approval, cancellation, steering and fork

The [parent approval example](https://github.com/0xPlaygrounds/rig/blob/ad8e9b69f857992f7078d66555bce5c4db751df5/crates/rig-ecs/examples/human_in_the_loop.rs)
read stdin in a schedule system. The current host polls input independently and
applies `decide(world, ticket, decision)`. Tickets bind world, effect generation,
revision, run, owner and exact proposed operation. Mutation makes old proposals
stale; decisions release only their named hold. No expiry or transactional
external-write guarantee is claimed. The input bridge owns no permanently blocked
reader thread and its Unix interactive limitation is explicit.

`commands.cancel_run`, `commands.retry_turn`, `commands.patch_turn` and
`commands.grant_tool` expose expected errors through CommandFailures. Immediate
operations return Result. Retry accepts complete, tool-free output before
materialization; patches merge only while Fresh. Grants affect future snapshots.
Cancellation leaves issued work to its handler. `fork(world, run)` refuses active
workers, unsupported cuts and remembered conversations; it preserves history
without copying prior effect subtrees. [Lifecycle tests](../../crates/rig-ecs/tests/lifecycle.rs)
exercise actual resulting requests and tool counts.

## Advanced scheduling and internal organization

Bevy QueryData names DispatchCandidate, CollectingStream, LandedEffect,
AssemblingRun, FreshTurn and AwaitingRun internally. It preserves scheduler access
rather than replacing parallel systems with exclusive World callbacks. Settings,
worker storage, collector bookkeeping, registry helpers and witness internals are
restricted to their implementation modules. The [API disposition table](api.md)
records details and API family dispositions.

Ordinary components, relationships, RigSet/BusSet and RigSchedule remain deliberate
extension points. SubjectWalk remains public because a mutating gate needs to
read subject identity without conflicting with its mutable PendingEffect query.
The [public consumer test](../../crates/rig-ecs/tests/bus_api.rs) demonstrates that
actual access pattern. Similar names in the two schedules refer to different
layers: RigSet::Judge reads a turn's folded outputs; BusSet::Judge handles effect
outcomes. Runtime query fields and delivery flush boundaries remain explicit.

## Save, restore and replay

Previously the combined graph/effect scene used `save_world(world)` and
`load_world(&scene, world)`, while its constituent scene types used methods.
Current call sites use `WorldScene::save(world)?` and `scene.load(world)?`.
RunScene::take is private; callers that need the advanced graph-only scene still
have RunScene::save/load and its persisted fields. No duplicate free-function
aliases remain.

The [run scene consumer](../../crates/rig-ecs/tests/run_scene.rs) now registers
replayers directly:

```rust
Replay::default().register_in(app.world_mut(), &log)?;
let loaded = saved.load(app.world_mut())?;
```

This uses the same binding, replay-header/delivery and scene validation paths.
Supported application components still require explicit SceneExtensions
registration. Loading is not fully transactional; application insertion observers
belong after restoration. Persisted field names, handler identities, policy
compatibility and unfinished-stream refusals remain unchanged.

## Consumer coverage and advanced exceptions

All original crate examples are migrated; hello_model intentionally teaches the
bus-only effect layer. Shared example support contains mocks/printing, not hidden
installation or agent builders. The [example inventory](examples.json) records
features and execution modes, including the new provider example.

[Provider test support](../../tests/common/ecs_agent.rs) uses public installation,
registration, Agent, checked grants and Prompt. It retains explicit legacy golden
identity metadata and the runtime-entry adapter required by provider IO; these are
fixture obligations, not required ordinary application boilerplate. The OpenAI
tool-lifecycle consumer uses named submission options and its original wire/tool
assertions. Twelve OpenAI recorded cases and two Anthropic golden-log comparisons
passed without recapture.

Direct graph/policy, scene, WASM and corpus tests remain independent oracles where
their lesson is component relationships, persistence, runtime ordering or exact
request behavior. They are not all converted to construction-wrapper tests.
Root facade docs and workspace example packages contained no routine ECS setup
snippets to migrate; the root facade continues to describe its separate classic
agent bus. Full provider/feature verification and exact-commit rigcoder validation
are still required before publication.

## Additional before/after call sites

These excerpts come from the linked source; they are fragments of complete
examples/tests, not standalone programs. Advanced graph tests intentionally keep
raw component operations when those operations are the contract under test.

### Approval input

[Parent schedule system](https://github.com/0xPlaygrounds/rig/blob/ad8e9b69f857992f7078d66555bce5c4db751df5/crates/rig-ecs/examples/human_in_the_loop.rs#L78)
blocked the schedule on terminal input:

```rust
let mut line = String::new();
let decision = match std::io::stdin().read_line(&mut line) {
    Ok(0) | Err(_) => None,
    Ok(_) => Some(line.trim().to_ascii_lowercase()),
};
```

The [current host loop](../../crates/rig-ecs/examples/human_in_the_loop.rs)
polls input between ticks and applies the exact displayed ticket:

```rust
match decide(app.world_mut(), ticket, decision) {
    Ok(_) => println!("decision applied"),
    Err(error) => eprintln!("decision rejected: {error}"),
}
```

### Cancellation and forks

[Parent cancellation test](https://github.com/0xPlaygrounds/rig/blob/ad8e9b69f857992f7078d66555bce5c4db751df5/crates/rig-ecs/tests/steer_hooks.rs#L83):

```rust
app.world_mut()
    .entity_mut(run)
    .insert(Cancelled("stopped at run start".to_owned()));
```

[Current lifecycle consumer](../../crates/rig-ecs/tests/lifecycle.rs) checks
liveness/phase through the named operation (this test deliberately expects success):

```rust
cancel(app.world_mut(), run, "stop").unwrap();
```

The [parent best-of-N loop](https://github.com/0xPlaygrounds/rig/blob/ad8e9b69f857992f7078d66555bce5c4db751df5/crates/rig-ecs/examples/best_of_n.rs#L56)
discarded branch identities and performed unchecked cloning:

```rust
for _ in 1..N {
    fork(world, run);
}
```

The [current example](../../crates/rig-ecs/examples/best_of_n.rs) checks the cut,
retains each identity and later judges that cohort:

```rust
let second = fork(world, first)?;
let third = fork(world, first)?;
```

### Scheduled internal access

[Parent assembly](https://github.com/0xPlaygrounds/rig/blob/ad8e9b69f857992f7078d66555bce5c4db751df5/crates/rig-ecs/src/systems/mod.rs#L87)
exposed positional runtime access publicly:

```rust
pub type AssemblingView = (
    &'static RunOf,
    &'static RunSeq,
    &'static Streamed,
    Option<&'static UsesModel>,
    &'static OutputToolName,
);
```

[Current assembly](../../crates/rig-ecs/src/systems/mod.rs) uses a private named
query while retaining ordinary Bevy scheduling and component access:

```rust
#[derive(bevy_ecs::query::QueryData)]
struct AssemblingRun {
    agent: &'static RunOf,
    sequence: &'static RunSeq,
    streaming: &'static RunStreaming,
    model: Option<&'static UsesModel>,
    output_tool: &'static OutputToolName,
}
```

The corresponding system parameter changes from `Query<AssemblingView, ...>` to
`Query<AssemblingRun, ...>`. These internal lifetime annotations belong to Bevy's
borrowed query definition, not to normal application call sites. Public
`RigSet`/`BusSet` ordering remains available; callers no longer need the internal
query alias to customize the runtime.

### Replay registration and scene loading

[Parent consumer](https://github.com/0xPlaygrounds/rig/blob/ad8e9b69f857992f7078d66555bce5c4db751df5/crates/rig-ecs/tests/run_scene.rs#L294):

```rust
Handlers::with(app.world_mut(), |handlers| {
    Replay::default()
        .register(handlers, &log)
        .expect("the golden's replayers")
})
.expect("a bus");
```

[Current consumer](../../crates/rig-ecs/tests/run_scene.rs):

```rust
Replay::default()
    .register_in(app.world_mut(), &log)
    .expect("the golden's replayers");
```

The adjacent load changes from
`load_world(&saved, app.world_mut()).expect("the model is bound")` to
`saved.load(app.world_mut()).expect("the model is bound")`, retaining the same
validation and restoration assertions.
