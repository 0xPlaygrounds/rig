# ECS DX implementation progress

Parent: `ad8e9b69f857992f7078d66555bce5c4db751df5` (`origin/feat/effect-bus`).
Task branch: `feat/ecs-dx`. No PR published yet.

The complete requirements remain those in the workspace prompt and companion audit.
The parent file inventory is `parent-files.tsv`; it records scope, not a correctness verdict.
`examples.json` starts the migration inventory; root snippets and test dispositions remain to be added.

## Remaining work

- Public/API inventory, including feature exports and supported advanced consumers.
- Agent construction, named run options, immediate/deferred entry points, registration and errors.
- Checked lifecycle/fork, borrowed inspection, per-stream progress.
- Nonblocking approval with exact identity, owned holds, and shutdown coverage.
- Internal named query data, visibility boundary and architecture guard changes.
- All example/snippet migrations, representative consumer tests, before/after review guide.
- Targeted and complete PR verification, local rigcoder integration.
- Independent full-diff review/fix loop, publication, required CI and review inspection.

No completion claim is made by this progress document.

## First implementation pass

Replaced the awaiting-model positional query with private named `AwaitingRun`
query data, internalized its sole `materialise` system, and corrected
`RequestGraph` ownership documentation. All-feature library check passed.
Targeted run graph/output/steering tests are in progress; inspect the running
Cargo session before restarting checks.

Independent read-only boundary reviewer identified these safe candidates:
collection/dispatch query aliases and worker systems; HandlerTable, Served,
WorldServe (also restrict WorldHandler::served); record observer internals;
SubjectWalk/DispatchWitness; Intake and WorldOutcomeCounter. Public modules
mean removing re-exports alone is insufficient.

Keep Progress (host extension), Subjects (agent witness consumer), and identity
counters (scene/reflect contracts). Observed, Executions, and Despawning need
internal test relocation before hiding: preserve cancellation-boundary and
executed WASM worker-drop coverage. Settings must retain per-component fallback
rather than a single query excluding partial overrides. Replace the lexical
visibility ban with a real public consumer test and retain other guards.

## Construction and boundary implementation pass

Added `commands::{Agent, Prompt, RigCommands, OperationError, CommandFailures,
install}`. These are owned temporary construction values and native Commands
extensions, sharing the original graph initializer. Added `Handlers::register_in`
for an immediate single-Result registration path. Prelude exports the common
construction vocabulary. `Agent::memory` covers same-system memory setup.

Deferred failures retain reserved IDs for host reporting/cleanup. New integration
tests compile ordinary systems and reusable helpers without named lifetimes,
exercise commands before handlers, memory in both parameter orders, logical
removal, stale targets, duplicate installation, and independently built request
parity. Eight tests pass. Additional history/streaming/lifecycle coverage and
error-draining example integration remain required.

Independent candidate review found a Bevy relationship-target race and a stale
binding race. Fixed through targeted application of pending initial handler
bindings: pending configuration is consumed once into Bound, and a removed
binding is never recreated. Construction no longer scans/materializes unrelated
handlers. The queued registration stores an initial Bound only until applied;
it does not maintain a second persistent configuration store. Review this new
path again in the full-diff review, including reentrant observers and replacements.

Internalized HandlerTable/Served/WorldServe, their observer helpers, and dispatch.
Removed unused table size accessors. Replaced the blanket restricted-visibility
ban with `tests/bus_api.rs`, an executable consumer of supported public components,
registration and Gate/Judge scheduling. Other architecture guards remain.

`agent_with_tools` now visibly installs, registers, builds and submits through
public APIs, without hidden setup helpers or manual grants/World closures.

Verification passed:
- `cargo check -p rig-ecs --all-features --lib`
- `cargo check -p rig-ecs --all-features --example agent_with_tools`
- `cargo test -p rig-ecs --all-features --test commands --test run_graph --test run_output_tool_config --test steer_hooks --test memory_resume` (36 tests at that intermediate revision)
- latest `cargo test -p rig-ecs --all-features --test bus_api --test bus_effects --test bus_world --test commands` (30 tests, no warnings)
- `cargo fmt --all`, `git diff --check`

No commits or publication yet. Full prompt scope remains, especially approvals,
inspection/lifecycle, remaining examples/docs/inventory, further internal cleanup,
full verification, downstream validation, independent final review and CI.

The migrated `agent_with_tools` executed offline within a 90-second guard,
printed exactly `-3`, and exited successfully. No verification session is left
running from this pass. Candidate independent review follow-up may still be active;
inspect `/root/dx_boundary_review` before assigning another task.

A further candidate review found that binding observers can remove the reserved
agent destination during materialization. The constructor now rechecks destination
liveness afterward; a dedicated observer-removal regression was added. Atomicity
documentation now promises validation before agent creation and explicitly allows
registration observer effects. Removed redundant Prompt materialization plumbing.
Latest consumer suite contains nine tests; confirm session 96518 result if this
handoff is read before the command finishes. Next substantial work should address
lifecycle/inspection and the remaining examples, retaining the complete checklist.

## Lifecycle, inspection, streaming, and inference pass

Added checked cancellation/fork operations, derived borrowed RunView/RunInfo
inspection, and StreamText offsets per effect. Fork validation rejects active
workers and unsupported cuts, omits completed effect subtrees, and cleans up
allocated clones when application observers remove the branch. A private
installation marker avoids falsely reporting missing installation while Bevy
has temporarily removed the executing schedule from its schedule resource.
Renamed agent Streamed selection to RunStreaming; persisted streamed keys stay
unchanged. Fixed the reflection roundtrip consumer import after that rename.

Migrated best_of_n and streaming_ui with explicit public setup, cohort/selected
run completion, checked forks, per-effect cursors and removal cleanup. Both
executed offline with exact output assertions and successful termination.
Their migration inventory entries now reflect those results.

Typed handler registration infers Key<H::Family> from the concrete handler.
Explicit register_erased_typed retains runtime selection for dynamic handlers.
The typed-key consumer no longer annotates its result or supplies a turbofish;
a regression checks inconsistent runtime descriptors are rejected before binding.

Verified lifecycle (8), run_fork (1), stream_text (2); scene/reflection/identity/
steering/memory suites (26); bus_scene plus commands (29 before the additional
metadata regression). Poll the final bus_scene session for the new regression.

Still incomplete: approval, remaining example/snippet migrations, lifecycle
steering/grants ergonomics, full internal/public boundary cleanup and inventories,
before/after document, full required verification, rigcoder integration,
independent complete review, publication and CI. No PR or commits published.

The inferred-registration metadata regression passed (bus_scene: 21 tests).
Independent bounded review then identified a second descriptor read during
binding; registration now binds the validated snapshot directly and a stateful
descriptor regression covers that edge. Custom kind/payload compatibility is
still the handler contract, as documented; coarse Family validation cannot prove
Rust payload compatibility. This was not a final full-diff review.

prompt_from_assets now visibly installs/registers/builds, submits through Prompt,
tracks submission on the application agent, and drains deferred CommandFailures.
All-feature compile and bounded offline execution passed with exact stdout:
`granted 1 tool(s) from agent.tools.json` followed by `-3`, exit 0.

## Nonblocking approval and complete crate example migration

Added approval::{ApprovalRequired, ApprovalRequest, ApprovalTicket,
ApprovalChoice, DecisionOutcome, ApprovalError, decide}. The agent schedule
captures required tool proposals after Gate and before Dispatch using the
existing named hold mechanism. Tickets bind World, effect generation, revision,
key/name/arguments/run/owner snapshot. Changes require new input; expected stale,
late, conflicting, lost-hold and removed-target errors are explicit. Denial and
cancellation preserve existing execution semantics. Failed preparation denies
dispatch and retains ApprovalError. Read-only effect inspection exposes the
approval snapshot without inventing a stall diagnosis or mutable status cache.

The library owns no input runtime. human_in_the_loop now owns a manual App tick
loop and bounded Unix readiness-polled input (dev-only nix dependency), with
scripted decisions on native platforms. No thread remains blocked on stdin;
dropping the input source needs no join. The main loop retains the displayed
ticket while input arrives. The README documents scheduling, repeated/stale
input, external source-state validation, and transient/non-persisted approval.

Independent bounded reviewer found decision insertion observer exposure, repeat
validation ordering, and cloned-request identity holes. Fixed by mutating the
existing decision field, checking current proposal before repeat acknowledgement,
enforcing World and effect identity, and revalidating after request publication.
Observer-supplied decisions and copied-state cases have regressions. The final
bounded source recheck found no additional confirmed findings; this was not the
required complete-diff review and the reviewer did not independently run tests.

Verification passed:
- approval: 11 tests, including actual dispatch counts and unrelated progress
- approval_input: 3 tests, including absent/partial input and idle shutdown
- lifecycle: 8 tests after approval integration
- cargo check/build -p rig-ecs --all-features --examples
- all six examples executed offline with bounded completion; five assert exact
  stdout, approval asserts the displayed review, applied input, and final answer
- approval CLI stdin approve/deny/cancel/EOF, all scripted decisions, and delayed
  partial input produced expected success/error exits

hello_model retains the deliberate bus-only lesson, with explicit fallible
registration and the existing CompletionRequestBuilder. Removed misleading
line-count claims and error-as-success handling. All six crate examples now
show setup directly; support::app/register/agent are deleted. The asset example
also handles an empty Children set when counting grants. Root/docs snippets,
representative test migration/dispositions, and the complete review inventory
still need work; crate examples alone do not complete that requirement.

A full cargo test -p rig-ecs --all-features run is now session 7487; poll its
actual result before restarting. No commits or PR published. Full internal API
cleanup, remaining lifecycle/grant conveniences, full inventory/review guide,
repository verifier, rigcoder validation, final independent full-diff review,
publication and CI remain outstanding.

Full crate all-features test run (session 7487) completed successfully, exit 0,
with no failures. This includes native unit/integration coverage; WASM-gated
suites contain zero native tests and still require executed WASM verification.
There are currently zero crate doctests, so this is not evidence that the
required new documentation snippets compile. The final bounded approval review
confirmed all reported identity/observer findings addressed with no additional
confirmed issue. Repository-wide verification and full-diff review are pending.

## Internal boundary and query cleanup pass

Internalized bus collection/dispatch/record/witness helpers and worker state:
CollectionBudget/CollectedOutcome/Intake/WorldOutcomeCounter, record observers,
DeliveryBatch/ReplacedBy/WorldObserver, witness Refused/SeenOutcome/DispatchWitness,
and query aliases. Runtime systems and Settings are private; next_order_in is
crate-only for assets. Public host sets/components and per-component override
resolution remain intact. DispatchCandidate, CollectingStream, LandedEffect,
AssemblingRun and FreshTurn replace positional query tuples with named fields.

Retained SubjectWalk deliberately: a public consumer now mutates PendingEffect
while naming its subject; Subjects would conflict with that mutable access.
The new docs/ecs-dx/api.md records reviewed dispositions and explicitly identifies
remaining inventory work rather than claiming a complete inventory.

Observed/ObservedState are bus-private. The complete original-answer cancellation
and policy/exchange replay test moved to bus/delivery/visibility_tests.rs with its
exact timing predicate and assertions; shared fixtures are reused through a
test-only self-crate alias. Cleanup assertions moved to private record/witness
unit tests. Despawning is private; public cancellation trace assertions remain.
An initial cleanup test missed installing Witnessing; fixed its setup and reran.

Executions and Streaming::spawn are bus-private. The WASM worker drop test moved
unchanged (imports only) to bus/effect/wasm_tests.rs. Added wasm-rig-ecs-lib to the
verifier, required PR selection test, and CI matrix so private tests still EXECUTE
on WASM. Existing public bus/run integration suites remain executed separately.
The bounded independent reviewer compared relocated bodies and confirmed no
weakened assertions, access/order/budget change, or check-selection gap. This is
not the final full-diff gate and the reviewer did not rerun tests independently.

Verification passed:
- named-query pass: native lib + bus_api/bus_effects/bus_delivery/run_graph/
  run_output_tool_config (79 tests)
- relocation pass: native lib (26), bus_delivery (23), bus_world (7), bus_witness (20)
- xtask verification planner tests: 25 passed
- executed WASM private worker test: 1 passed
- executed WASM public bus: 5 passed; run: 2 passed
- exact new cargo xtask verify --check wasm-rig-ecs-lib: passed

The global wasm-bindgen runner is 0.2.126, incompatible with the locked 0.2.118.
Used the existing matching binary at:
/Users/kisaczka/Desktop/code/many_rigs/experiments/pull-handler-performance/tools/wasm-bindgen/bin
Prepend that directory to PATH for verifier WASM checks; do not update repository
WASM dependencies or overwrite the global tool to work around the mismatch.

Strict rustdoc checking found three old HandlerTable links after internalization;
updated public docs to describe the private registry without linking hidden names.
Poll session 99394 for final strict rustdoc result if needed. No other verification
session is live from this pass. Full goal remains incomplete: lifecycle/grant
conveniences, complete inventory/docs/representative consumer migrations, complete
repository verification, rigcoder testing, full independent review, publication,
and required CI/review state still need completion. No commits or PR published.

Strict rustdoc completed successfully after the link fixes (session 99394, exit 0), with both broken and private intra-doc links denied.


## Checked lifecycle conveniences and primary documentation

Added direct and deferred grant/revoke/retry/patch operations. Static grants
validate handler family and pending bindings, preserve ordering on repeats, and
apply to future requests; revocation leaves current snapshots and issued effects
alone. Retry accepts complete tool-free unmaterialized output in the judging
window, preserves identical pending decisions, rejects conflicting feedback, and
does not revive ended runs. Patch composes existing RequestPatch merge rules
while Fresh. Migrated representative steering tests to RigCommands and Agent /
Prompt while retaining direct component and scene oracles.

Independent bounded review found static grants incorrectly reused Retrievable
links and CommandFailure documentation implied all targets were safe to clean
up. Fixed both; actual captured-request and still-usable-agent regressions pass.
The reviewer rechecked and found no additional confirmed issue. No new failure
kind enum was added: clear target semantics suffice for current consumers.
Added retry/patch phase, repeat, conflict, removal and cancellation tests. The
first fixture incorrectly treated Folded as a marker; fixed it to Native output.
A test also assumed cancellation preserved RequestPatch; cancellation already
clears it, so the assertion now checks rejection preserves post-cancel state.

Documented and tested Handlers inspection as applied-state visibility; pending
IDs remain valid for command construction. Added compiled crate-level setup and
lifetime-free system snippets, and made README lead with Agent/Prompt and link
to the runnable example. These are documentation migrations, not evidence that
the remaining workspace/provider/snippet inventory is complete.

Verification: cargo fmt --all and git diff --check passed; cargo test -p rig-ecs
--all-features --test lifecycle --test commands --test steer_hooks passed (33
tests); strict cargo doc with broken/private links denied passed; cargo test
-p rig-ecs --all-features --doc passed both new snippets. Full changed verifier
started with matching WASM runner; output at /tmp/rig-ecs-dx-verify-changed.log.
Poll the live tool session before restarting it. No commit or PR published.


Full changed verification attempt: first run refused a reusable fmt result
because documentation edits overlapped its fingerprint check. Restarted after
freezing inputs. The second run passed fmt/source guards/tooling, then failed
clippy on this branch's approval/lifecycle helpers (nested conditions, query
type complexity, unchecked fork mapping indexing). Fixed those without changing
policy/order/cleanup behavior; bounded reviewer confirmed. Crate all-target
clippy then exposed example-support expect/indexing and example/test query
complexity; replaced with checked accesses and named private filters. Poisoned
mock script mutexes now recover contents; this change is confined to example
support and was explicitly noted by the reviewer.

Added replacement metadata visibility assertions to the registration test and
corrected stale plugin/every-component-reflect claims in lib docs and manifest
comments. Targeted commands/lifecycle/approval/steer tests passed (44). Reexecuted
agent_with_tools (-3) and prompt_from_assets (one grant, -3), both exit 0.
The most recent crate all-target clippy session must be polled (62717); full
verifier has stopped and needs rerunning after current fixes. No claim of a
passed full gate; no commit/PR publication.

Read-only inventory observations for next pass: root README/src and workspace
examples contain no ECS construction snippets to migrate. Provider tests still
have direct construction consumers, including OpenAI ecs_chat_tool_lifecycle
and Anthropic shared EcsAgent helpers; retain independent graph oracles while
migrating representative provider consumers. SceneExtensions, WorldScene,
RunScene, Loaded, reflection and assets public surfaces have been read but still
need explicit disposition entries. systems::witness::install/ending_of remain
public implementation helpers requiring consumer lookup before internalization.

Crate all-target clippy passed (62717 exit 0). The attempted combined --test and
--doc invocation was rejected by Cargo; separate bus_api and doctest commands
were used instead. Bus_api passed both tests. Poll session16473 for doctests and
scripted approval example, then rerun the full changed verifier on frozen inputs.

Session16473 completed: bus_api 2/2, doctests 2/2, scripted approval example
printed decision applied / Done and exited 0. All sessions above are terminal.
Restarting full changed verification with matching WASM runner and frozen source.


## Observer-safe construction, provider consumers and scene entry points

Previous turn classified progress. Full changed verifier session17979 was polled
live. It passed workspace clippy and default-check, then began default-tests.
Stopped only that verified process tree to fix a confirmed construction race;
no claim of complete verification, and no unrelated processes were stopped.

Independent bounded review confirmed synchronous observer removal can interrupt
Agent/Prompt initialization. Added checked entity/child/order helpers, fallible
initialization and structured deferred errors. Bevy debug output confirmed a
Children Entry command could run after a payload observer removed its parent.
Establishing ChildOf before payload insertion resolves that specific ordering
race. Relationship observers now see an attached child before payload exists;
this tradeoff is documented. A subsequent reviewer finding showed child-only
removal could leave a rejected run active; constructor-owned graphs are now
removed on initialization failure. Immediate and deferred tests cover agent
memory/tools, initial run/limit/phase/history removal, child self/parent removal,
no orphan graphs, and zero dispatch after rejected submission. Unrelated observer
actions are not rolled back. Reviewer rechecked cleanup and found no new issue.

Added provider_construction: real Gemini client and CompletionAdapter with an
explicit in-memory HTTP transport. Executed offline, exact Hello from Gemini.,
exit 0 within the five-second bound. Migrated shared provider EcsAgent setup,
registration, grants and submission plus OpenAI tool-lifecycle call sites to the
new public API. Preserved explicit None legacy default metadata used by golden
identity. Twelve OpenAI recorded tool-lifecycle cases and two Anthropic golden-log
comparisons passed in replay mode. No cassette paths changed.

Consolidated combined scene entry points as WorldScene::save and scene.load,
removed free save_world/load_world, and made RunScene::take private. All in-tree
callers migrated; UFCS remains where it lets serde result types infer. Added
Replay::register_in for immediate World registration, used by real scene/golden
replay tests. Existing validation/reconstruction bodies and persisted streamed
key remain intact. Independent bounded review found no semantic change. Downstream
rigcoder has old free-function callers and will need a LOCAL migration patch in
its isolated validation checkout; do not alter/push rigcoder main.

Internalized remaining replay collector/idle-diagnosis functions, collector
constants, and systems witness install/ending conversion after caller searches.
Added docs/ecs-dx/review.md opening scorecard and parent/current excerpts;
public-members.json enumerates 222 locally defined all-feature rustdoc items and
public fields/variants/inherent methods; public-exports.json enumerates 41
module/re-export declarations. These are mechanical inventories, not substitutes
for final source/disposition reconciliation. api.md adds feature/member rationale.
README now names supported reflection rather than falsely claiming every transient
component reflects.

Verification on this pass:
- commands/lifecycle/run_graph/run_scene/extensions/steer: 55 tests passed
- scene/memory/reflect/steer/batch/tool-access/bus_scene: 70 tests passed
- corpus_checkpoint + corpus_resume: 126 tests passed
- OpenAI tool-lifecycle recorded consumer: 12 passed
- Anthropic golden-log consumer: 2 passed
- provider_construction: executed successfully, default feature build
- strict rustdoc broken/private links denied: passed
- rig-ecs all-feature all-target clippy: passed
- cargo fmt --all and git diff --check: passed
All tool sessions in this pass are terminal except any new parent-head query.
Full verifier remains stopped; rerun required final checks after remaining work.
No commit, push, or PR publication yet. Final full-diff independent review,
exact-commit rigcoder check, parent refresh/rebase as needed, full PR verifier,
publication and required CI/review state remain incomplete.

## Full-review fixes and actual downstream boundary

Fresh full-diff review found no P0/P1 and three confirmed P2 issues, all fixed:
- registration now retains the validated Bound descriptor snapshot for applied
  family validation and same-borrow replacement; redundant WorldServe family
  storage removed;
- approval rejects missing RunOf and missing installed cancellation state before
  recording a decision; repeat failed Cancel stays an error rather than a receipt;
- request equivalence now builds the entire reference run/history graph directly,
  independently of Prompt/spawn_run, for streaming and unary submission.
Corrected the review guide's StreamText::read argument. Follow-up review confirmed
these fixes, then caught an agent-dependent test in the bus-only suite. Moved that
regression to commands.rs and retained bus-only descriptor replacement coverage.
Updated the architecture suite's explicit inventory for the five new public DX
contract files, preserving its bus-independence checks. Also corrected Begin docs
to preserve the shared runner collection budget (not reset on every pass).

An isolated rigcoder checkout at 6eaba2ed77b4b4c611b680769ecf0b36d06340b3 revealed
two real advanced consumers of newly private internals: scheduled producer ready()
inspects initial/stream task liveness, and lifecycle input runs before delivery
batch advancement. Added read-only execution_status/ExecutionStatus and
BusSet::Begin for those exact needs. No task handles exposed, no mirrored mutable
status; finished workers can still have uncollected answers. The local migration
uses the same readiness booleans and ordering with all fixture assertions intact.
Public bus_api test covers absent runtime/entity, pending initial reply, active
stream, finished-but-uncollected reply/items, collection and removal.

Provisional downstream validation uses local path overrides for all five Rig
dependencies, with scene method and scheduling/status call-site migration only.
Initial all-target check identified begin_delivery_pass access and was corrected;
workspace tests are running. No downstream publication or main changes. Exact Rig
commit validation is still pending; dirty path validation is not substituted for it.

Focused commands/approval/bus_scene passed after the three P2 fixes; bus_api and
bus_scene passed after readiness additions. Strict broken/private rustdoc-link
check passed. Latest moved-test and architecture checks are running. Full final
repository gate, final inventories/review, task commits, exact-commit downstream
verification and stacked publication/CI remain pending.

Latest evidence from this pass:
- Six retained root architecture tests passed. The provider example now uses a
  finite 5,000-tick host guard instead of a clock forbidden by the existing source
  guard; re-execution returned exactly Hello from Gemini. (exit 0).
- The final reviewer found no remaining code findings; corrected two editorial
  leftovers and canonical rustdoc paths. All 224 member inventory entries have
  generated pages; 42 exports and snippet inventory reconciled with source.
- Downstream dedicated consumer target: 58 passed, 0 failed. Offline verify:
  all 42 cases passed. These are provisional local path results, not yet an
  exact committed Rig revision.
- Downstream broader workspace run stopped at gemini_observe: 10 passed, 61
  failed. Every reported first difference is a changed policy hash. Product
  session.rs hashes include_str! source including session/checkpoint files; the
  required scene API migration changes that identity. This does not prove the
  remainder of each packet identical. Existing fixture assertions and artifacts
  were not bypassed, normalized differently, rewritten or recaptured.
- Temporary seven-file local migration saved outside repositories at
  /tmp/rigcoder-ecs-dx-provisional.patch, SHA256
  38b8d548d0e9c7817e010b5e73afc1b87ae6e162cd2ec7ff7fd163ed136fe995.
- Direct parent-blob comparison of 2,482 tracked Rig cassette/corpus fixtures:
  no changes. Path/content SHA256 inventory:
  7d963463951277bdd48416c6a8919a34d21d8009c8f8b7e157a78e0b446cc950.
- Refetched parent branch and queried #2443: OPEN, unchanged ad8e9b69.
- Final changed gate passed fmt, source guards, tooling, all-workspace Clippy,
  and default check. Default tests are running. A fixture-only clippy::panic
  allowance was needed on bus_api's shared test support; no production lint
  suppression was added. Remaining checks, commits and publication are pending.

The changed gate stopped after 6,033 default-test passes on a new test's allocator
count assertion (32 to 64). Locked Bevy 0.19.1 documents Entities::len as allocated
metadata length, not live entities. Replaced these assertions in commands and
lifecycle tests with count_spawned, preserving target/grant/dispatch/sequence
assertions. Independent reviewer confirmed the correction; no runtime change.
Targeted nextest commands+lifecycle with zero retries: 29 passed. Proceeding to
the complete PR-mode gate against origin/feat/effect-bus (80 paths, 32 checks),
which re-executes required checks after this correction. The earlier changed run
is failed evidence, not a successful full gate.

Independent completion-scope audit found two acceptance-evidence gaps, now fixed:
the public consumer combines model/tool registration, twice-called helper,
Commands/resource composition, Query<RunView> result collection and captured tool
advertisement in one test; review.md adds source-matched before/after excerpts
and parent links for approval, cancellation/fork, scheduled internal query access
and replay/restore. Reviewer rechecked both and reports no remaining scope gap.
Targeted commands nextest after the extension: 19 passed, no retries. Capturing's
Clone is test-fixture-only and shares its existing request collector.

PR-mode progress before the scope additions: default tests 7,051 passed (203
skipped), core-all 2,661 passed (2 skipped), bus verification 815 passed,
macro-hygiene passed, conformance 50 passed, derive passed. Doctests themselves
passed, but the verifier correctly refused their result because the reviewed
scope fixes changed inputs while they ran. This is not a completed PR gate.
Restarting PR mode with explicit --reuse; only fingerprint-matching successes can
be reused, and full-tests/dependency-floors still execute by repository policy.
No additional planned source/artifact edits remain before this gate finishes.
