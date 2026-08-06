# Refactor PR #2252 to make runtime model selection entirely hook-driven

Repository: <https://github.com/0xPlaygrounds/rig>

Existing draft PR: <https://github.com/0xPlaygrounds/rig/pull/2252>

Branch: `refactor/runtime-model-swapping`

This is a follow-up refactor inside PR #2252. Do not open a replacement PR and
do not reimplement PR #2228's provider/configuration architecture.

## Objective

Remove the public `.select_model(...)` API and all dedicated selector storage
from PR #2252. Runtime model selection must instead be a first-class
`AgentHook` lifecycle event composed exclusively through the existing
`HookStack`.

The final architecture must have:

- one default `ModelHandle` on `AgentRunner`;
- no `model_selector` field or equivalent routing-policy field on the runner;
- no private selector callback stored outside the hook stack;
- no `.select_model(...)` convenience method on the runner, prompt requests,
  typed requests, streaming requests, or any other surface;
- dynamic selection performed only by hooks implementing a dedicated model
  selection event;
- the exact selected handle bound to `PreparedModelAttempt` before any
  capability-sensitive request preparation;
- identical behavior on blocking and streaming runs through the shared
  `drive_agent` path.

Keep the rest of PR #2252's architecture: public opaque `ModelHandle`, concrete
high-level agent types, private typed-model erasure, provider-neutral prepared
requests, attempt binding, normalized high-level streaming finals, direct
low-level typed provider APIs, and open `CompletionModel` authoring.

Backward compatibility and breaking semver are not concerns. PR #2252 is still
draft, so rewrite its documentation and changelogs as though hook-driven model
selection had been the design from the beginning. Do not document removal of
an interim `.select_model(...)` API that has never shipped.

## Read and preserve the current state

Before editing:

1. Read the repository's `AGENTS.md` and `CONTRIBUTING.md` completely.
2. Fetch the latest `origin/main` and the current PR branch.
3. Record the merge base and inspect the complete PR diff.
4. Inspect staged, unstaged, and untracked files. Preserve all user-owned
   changes and prompt drafts.
5. Read the current implementations of:
   - `crates/rig-agent/src/agent/model.rs`;
   - `crates/rig-agent/src/agent/hook.rs`;
   - `crates/rig-agent/src/agent/runner.rs`;
   - `crates/rig-agent/src/agent/prompt_request/streaming.rs`;
   - `crates/rig-agent/src/agent/prompt_request/mod.rs`;
   - `crates/rig-agent/src/agent/completion.rs`;
   - the runtime-model-swapping integration tests and example.
6. Rebase or merge the latest `origin/main` only if required by the repository
   and PR state. Do not discard existing PR work or unrelated user files.

## Public hook API

Add a dedicated borrowed hook event in `agent::hook`, following the naming and
documentation style of `CompletionCall`, `ToolCall`, and the other lifecycle
events. A suitable shape is:

```rust,ignore
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct ModelSelection<'a> {
    /// Prompt for the pending model call.
    pub prompt: &'a Message,
    /// Canonical history visible to the pending model call.
    pub history: &'a [Message],
    /// Model selected for the preceding model attempt in this run.
    pub previous_model: Option<&'a ModelHandle>,
    /// Runner default used as the initial candidate for this call.
    pub default_model: &'a ModelHandle,
    /// Candidate after all earlier model-selection hooks.
    pub selected_model: &'a ModelHandle,
}
```

`HookContext` remains the authoritative source for:

- run id;
- one-based turn/model-call index;
- streaming versus blocking mode;
- agent name;
- the run-scoped scratchpad.

Do not duplicate those values on `ModelSelection`. A routing hook should use
`ctx.turn()` and `ctx.is_streaming()`.

Add an event-specific action, following existing hook action conventions:

```rust,ignore
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ModelSelectionAction {
    /// Keep the candidate supplied to this hook.
    Continue,
    /// Replace the candidate and pass it to later hooks.
    Select(ModelHandle),
    /// Stop the run before request preparation or model execution.
    Stop(String),
}
```

Provide constructors consistent with the other action types, such as
`continue_run`, `select`, and `stop`.

Add the hook method:

```rust,ignore
fn on_model_select(
    &self,
    _ctx: &HookContext,
    _event: ModelSelection<'_>,
) -> ModelSelectionAction {
    ModelSelectionAction::Continue
}
```

This event is deliberately synchronous, even though the other hook methods
return futures. Selecting an already-constructed `ModelHandle` must not add
network I/O, asynchronous routing, health checks, or a second async policy
lifecycle to this PR. Stateful routing hooks may use caller-owned explicit
synchronization, and all hook types remain subject to Rig's native/browser-WASM
compatibility bounds.

Add synchronous object-erased dispatch to the private `DynAgentHook` adapter.
Do not box a ready future merely to disguise the synchronous contract.

Export `ModelSelection` and `ModelSelectionAction` anywhere the other public
hook events/actions are exported, including the root `rig` facade and relevant
prelude. Do not move the advanced `CompletionModel` authoring trait into the
ordinary prelude.

## HookStack composition

Implement model selection through the existing ordered `HookStack`. Do not add
a separate hook lane or selector collection.

Composition rules are:

1. Begin each `CallModel` boundary with a clone of the runner's default
   `ModelHandle` as the current candidate.
2. Invoke every hook once in registration order.
3. Before each invocation, set `event.selected_model` to the candidate produced
   by all earlier hooks.
4. `Continue` preserves the candidate.
5. `Select(handle)` replaces the candidate and continues to later hooks.
6. If several hooks select, the last `Select` wins.
7. `Stop(reason)` is terminal and prevents later model-selection hooks,
   completion-call hooks, request preparation, and provider execution.
8. If every hook continues, use the runner default.
9. Nested `HookStack`s must preserve the same candidate threading and terminal
   stop semantics. A nested stack returns `Select(final_candidate)` only when
   at least one nested hook selected; it otherwise returns `Continue`.

Do not compare handles to infer whether a selection occurred. `ModelHandle`
has identity-free value semantics and should not gain `PartialEq` merely for
HookStack bookkeeping. Track whether any hook returned `Select` explicitly.

Update the hook module overview and `HookStack` documentation with these
composition rules. Preserve all existing rules for completion-call patch
merging, tool-call/result rewrite chaining, invalid-call recovery, model-turn
steering, observations, and stop short-circuiting.

## Runner and driver integration

`AgentRunner` should contain only its default:

```rust,ignore
pub(crate) model: ModelHandle,
```

Remove:

- `AgentRunner::model_selector`;
- `ModelSelector`;
- `SelectorCallback`;
- `AgentRunner::select_model`;
- every forwarding `.select_model` method and macro expansion;
- `ModelSelectionContext` if it exists solely for the removed closure API;
- every stale import, re-export, test, example, and doc reference.

At each `AgentRunStep::CallModel` in shared `drive_agent`:

1. Set the turn on `HookContext`.
2. Start with the runner's default model as the candidate.
3. Resolve the model-selection hook event exactly once through `HookStack`.
4. If a hook stops, cancel through the existing `AgentRun::cancel_error` path
   and surface the correct blocking or streaming error.
5. Record the final handle as `previous_model` for the next model-call
   boundary.
6. Resolve completion-call hooks.
7. Build the provider-neutral request using the final handle's capabilities.
8. Clone that exact handle into `PreparedModelAttempt`.
9. Execute the handle retained by the prepared attempt.

Do not update `previous_model` when model selection stops the run.

The model-selection phase occurs once per actual `CallModel`, including:

- the initial call;
- a call after tool results;
- invalid-tool-call retries that re-enter `CallModel`;
- accepted-turn hook retries;
- structured-output repair attempts.

It does not run for tool execution, individual stream polls, deltas, response
observations, or request-patch merging.

The selected handle cannot change while a unary future or provider stream is
in flight. Dropping the attempt must still drop the retained future/stream and
handle without detached work.

## Meaning of fixed/default overrides

Keep:

- `Agent::set_model_handle` / `set_model` / `with_model_handle`;
- `AgentRunner::using_model` / `using_model_value`;
- the corresponding prompt, streaming, typed, and extraction overrides.

With hook-driven routing, these APIs set the default or initial candidate; they
do not suppress registered model-selection hooks. This should be explicit in
their documentation:

- agent replacement changes the default captured by later runners;
- a runner created earlier retains its prior default snapshot;
- `.using_model(...)` changes the default for that run only;
- model-selection hooks may replace that candidate at each `CallModel`;
- hook registration order determines precedence between routing hooks;
- a caller who needs an unconditional fixed run should not attach a routing
  hook, or should append a final routing hook that always selects the fixed
  handle.

Do not add a hidden fixed-model hook, selector tag, routing enum, priority
number, `Any` downcast, or another runner field to emulate the removed
`.select_model` precedence rules. The hook stack is the only dynamic routing
mechanism.

## User-facing usage

The intended API should look like:

```rust,ignore
#[derive(Clone)]
struct RouteModels {
    fast: ModelHandle,
    strong: ModelHandle,
}

impl AgentHook for RouteModels {
    fn on_model_select(
        &self,
        ctx: &HookContext,
        _event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        if ctx.turn() == 1 {
            ModelSelectionAction::select(self.fast.clone())
        } else {
            ModelSelectionAction::select(self.strong.clone())
        }
    }
}

let answer = agent
    .prompt("Research this, use tools, then synthesize")
    .max_turns(3)
    .add_hook(RouteModels {
        fast: fast.clone(),
        strong: strong.clone(),
    })
    .await?;
```

Agent-builder hooks route every run created from that agent. Request/runner
hooks route only that run. Because `HookStack` is cloned into a runner, each run
gets an independent stack snapshot while explicitly synchronized state inside
a hook remains shared according to that hook's own clone semantics.

## Serialization and extensibility boundaries

`ModelHandle`, `ModelSelectionAction`, routing hooks, and hook stacks are live
runtime behavior. Do not make them `Serialize` or `Deserialize`, do not put
them inside serde `AgentRun`, and do not hide them behind skipped serde fields.

Do not place handles in `RequestPatch`, `additional_params`, `Scratchpad`, tool
context, or message history. Do not add provider registries, provider enums,
stable serialized model identities, `Any`, downcasting, unsafe vtables, or
process-global routing state.

Preserve:

- external `CompletionModel` implementations;
- typed direct unary and streaming provider responses;
- custom hooks and nested HookStacks;
- tool, memory, vector-store, transport, and provider extension traits;
- native and browser-WASM behavior;
- provider request payloads and cassette fixtures.

## Tests

Rewrite the existing deterministic runtime-model-swapping tests to use routing
hooks instead of selector closures. Add focused hook-level tests as well.

Cover at least:

1. With no selecting hook, every attempt uses the runner default.
2. One model-selection hook routes to a heterogeneous `ModelHandle`.
3. An agent-builder routing hook applies to later runners.
4. A request-local routing hook applies only to that run.
5. Two selecting hooks run in registration order and the last selection wins.
6. A later hook observes the candidate chosen by an earlier hook.
7. `Continue` preserves the current candidate.
8. `Stop` prevents later hooks and provider execution and returns the existing
   cancellation error shape.
9. Nested HookStacks preserve candidate threading and stop semantics.
10. Each registered routing hook runs exactly once per actual `CallModel` and
    never for tool execution or stream polling.
11. A tool-producing first model and a synthesizing second model work through
    both blocking and streaming surfaces.
12. Model-turn, invalid-tool, and structured-output retries re-enter routing
    exactly once and receive the correct `previous_model`.
13. Capability-sensitive preparation uses the final hook-selected model.
14. `PreparedModelAttempt` executes the same handle used for preparation.
15. Dropping pending unary and streaming attempts retains cancellation
    behavior.
16. Concurrent runs select independently without cross-run leakage.
17. `.using_model(...)` changes the event's default and initial candidate but a
    routing hook can override it.
18. Agent replacement and clone value semantics remain correct.
19. Existing completion, tool, observation, scratchpad, memory, output-mode,
    and hook-composition regression tests remain green.
20. A downstream-style custom `CompletionModel` can be selected by a custom
    hook without knowing any private erasure types.
21. Browser-WASM compile checks pass with routing hooks and actions.

Use labels and recorded requests to assert handle selection; do not add handle
identity or downcast APIs solely for tests.

After the refactor, this search must return no source or documentation matches
other than historical commit/PR text deliberately outside the branch diff:

```bash
rg -n "select_model|model_selector|ModelSelector|SelectorCallback|ModelSelectionContext"
```

## Documentation and PR presentation

Update:

- `crates/rig-agent` module and public item docs;
- the credential-free runtime-routing example;
- `crates/rig-agent/README.md`;
- root and crate changelogs;
- `MIGRATING.md`;
- the draft PR #2252 title/body if necessary.

Explain:

- why routing is a hook lifecycle event;
- selection composition and registration order;
- default versus selected models;
- the model-call boundary and immutable attempt binding;
- blocking/streaming parity;
- clone, concurrency, retry, and cancellation semantics;
- live hooks and handles being intentionally non-serializable;
- direct provider APIs retaining typed raw responses.

Do not mention `.select_model(...)` as a supported or removed public API. The
PR should present hook-driven selection as its original design.

## Verification

Run the narrow checks first, then the complete required suite. At minimum:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --workspace --all-features --doc
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
git diff --check
```

Also run:

- the focused `rig-agent` hook unit tests;
- the runtime-model-swapping integration test target;
- the credential-free routing example;
- current WASM checks for `rig-core`, `rig-agent`, `rig`, `rig-candle`, and the
  WASM example packages affected by the PR;
- a complete search proving all selector APIs/storage were removed;
- an explicit cassette diff and safety review proving request fixtures did not
  change.

If full workspace rustdoc fails only on the existing unrelated `rig-neo4j`
private-link warning, confirm the same failure on `origin/main`, run the scoped
`rig-core`/`rig-agent`/`rig` rustdoc command, and report the limitation rather
than modifying unrelated code.

## PR completion gate

Before updating the remote PR:

1. Identify the latest intended `origin/main` and inspect the full diff against
   its merge base, including staged, unstaged, and untracked files.
2. Preserve unrelated user-owned files and exclude them from commits.
3. Run all required formatting, linting, tests, docs, examples, and WASM checks.
4. Have a fresh independent reviewer or subagent that did not implement this
   refactor review the complete diff for correctness, public API coherence,
   hook composition, concurrency, cancellation, WASM behavior, security, and
   missing tests.
5. Validate every finding against current code. Fix every confirmed P0/P1 and
   address in-scope lower-severity findings or document why they remain.
6. Rerun affected checks and obtain a final independent review after fixes.
7. Create an intentional follow-up commit, push the existing branch, and update
   draft PR #2252. Do not open a new PR.
8. Monitor required CI and unresolved actionable review threads. Do not claim
   completion while checks are pending/failing or confirmed P0/P1 findings or
   requested review work remain.

The final handoff must report the reviewed diff scope, findings and resolutions,
verification commands and results, cassette state, CI/review state, and any
remaining risks or blockers.

## Definition of done

This refactor is complete only when:

- `.select_model(...)` and all selector-specific types/storage are absent;
- `AgentRunner` contains no dynamic routing field beyond its default
  `ModelHandle` and existing `HookStack`;
- model selection is a dedicated synchronous `AgentHook` event;
- HookStack threads candidate handles in registration order with last-selection
  wins and terminal stop semantics, including nested stacks;
- selection runs once per `CallModel` before capability-sensitive preparation;
- the final selected handle is immutably bound to each prepared unary or
  streaming attempt;
- blocking and streaming routing behavior is equivalent;
- fixed/default replacement, retries, tools, hooks, memory, output modes,
  cancellation, concurrency, external models, and low-level typed APIs retain
  their intended behavior;
- documentation and the example teach only the hook-driven API;
- native, WASM, formatting, lint, test, doctest, and documentation checks pass;
- the complete revised diff passes fresh independent review;
- draft PR #2252 is updated, required CI is green, and no actionable required
  review thread remains unresolved.
