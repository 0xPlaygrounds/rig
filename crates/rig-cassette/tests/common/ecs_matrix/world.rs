//! The world cell: a cell's program as an agent graph in a Bevy `World`,
//! its hooks the corpus's systems (`corpus::world_hooks`), served by the
//! real adapters over the same cassette as the producer, and asserted in
//! this order: (1) the log the world wrote has the kinds, outcomes,
//! events and parent chain the cell states, (2) the run's graph after
//! settle has the shape the section states, (3) where the cell names a cut, a scene
//! saved there loads in a fresh world served by replayers over the log's
//! tail and finishes to the same answer. Every settled run is then
//! despawned (`despawn_run`, CONTRACT: the world keeps nothing of a run by
//! itself).
//!
//! A failure-row cell (`super::faults`) takes the same path: the fault it
//! names decides what the driver does mid-run (parks a tool, saves a scene
//! at a cut the happy paths have no name for) and what it asserts beside
//! the record (the failure's facts, the stream's fold, the history that
//! was never committed, `despawn_run` refusing until the fault drained).

use rig_ecs::agent::checkpoint::{
    ToolTurnCommit, ToolTurnHolds, TurnAssistant, TurnResults, hold_after_tool_turn,
    release_tool_turn_hold,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy_app::App;
use bevy_ecs::prelude::*;
use futures::StreamExt;

use rig_agent::completion::CompletionModel;

use rig_agent::completion::CompletionRequest;

use rig_agent::completion::CompletionResponse;

use rig_agent::completion::ProviderCapabilities;

use rig_core::effect::EffectFamily;
use rig_core::error::ProviderError;

use rig_core::effect::EffectKind;

use rig_core::effect::HandlerKey;

use rig_core::effect::Outcome;

use rig_core::error::ErrorKind;

use rig_core::serve::ErasedHandler;

use rig_core::serve::Serve;

use rig_core::serve::adapters::CompletionAdapter;

use rig_core::serve::adapters::MemoryAdapter;

use rig_core::serve::adapters::ToolAdapter;

use rig_core::streaming::Delta;

use rig_core::streaming::StreamEvent;

use rig_core::streaming::StreamEvents;

use rig_core::streaming::StreamingCompletionResponse;

use rig_core::tool::Tool;

use rig_core::tool::ToolContext;

use rig_cassette::ecs::EffectLogResource;
use rig_cassette::ecs::identity::stamp_run;
use rig_cassette::effect_log::{Checkpoint, EffectLog, EffectLogRecorder, RequestCheck};
use rig_ecs::{
    agent::{
        AdditionalParams, Cancelled, Conversation, Cursor, DefaultMaxTurns, Failed, Failure, Grant,
        InvalidCalls, MaxTokens, MaxTurns, MessageParts, Output, OutputKind, Owner, PolicyVersion,
        Preamble, ProviderRetried, ProviderRetries, Remembers, Role, Route, Run, RunOf, RunResult,
        Runs, Settled, Temperature, ToolChoiceSpec, ToolPolicy, Turn, Unhandled as WorldUnhandled,
        UsesModel, Utterance,
    },
    bus::{
        BusSet, EffectOutcome, Handlers, IdCounter, InFlight, PendingEffect, Policy, RigSchedule,
        Streamed,
    },
    checkpoint::{RestoreMode, load_world, save_world},
    systems::{Fresh, RigSet, RunBusy, RunCommands},
};
use tokio::sync::Semaphore;

use super::cells::{Cell, Memory, ToolKind};
use super::corpus::{self, CANCEL_ADD_OUTCOME, Ending, Hook, LayerAt, Program, Unhandled};
use super::faults::{BROKEN_ORCHARD, FailingOrchard, Fault, Scene};
use super::{DEFAULT_LABEL, WireBinding};
use super::{OWNER, Wire};
use crate::ecs_agent::{RuntimeHandler, io_runtime};
use crate::goldens::{
    Adder, BROKEN_ADD, FailingAdd, FailingMemory, NoteTaker, WriteNote, families,
};
use crate::stream_faults::{sole_stream, utterance_roles};
use crate::support::{ALPHA_SIGNAL_OUTPUT, AlphaSignal, BetaSignal};

const GUARD: Duration = Duration::from_secs(180);

/// Pin the bus's task pool to one thread, as rig-cassette's world
/// interpreter does: a replayer answers same-key dispatches by position,
/// which holds when handler tasks are first polled in spawn order.
/// Process-wide: nextest runs each cell in its own process.
fn one_thread_pool() {
    bevy_tasks::IoTaskPool::get_or_init(|| {
        bevy_tasks::TaskPoolBuilder::new()
            .num_threads(1)
            .thread_name("ecs-matrix-io".to_owned())
            .build()
    });
}

/// The gates a driver holds over a cell's handlers: a parked tool answers
/// once `tool` has a permit; a gated stream goes on past its first delta
/// once `stream` has one. A cell that names neither never touches them.
pub(crate) struct Gates {
    pub(crate) tool: Arc<Semaphore>,
    pub(crate) stream: Arc<Semaphore>,
    /// The witness's log, installed for every failure-row cell: the ending
    /// it names is asserted beside the run's.
    pub(crate) witness: Option<Arc<rig_core::observe::ObservationLog>>,
}

/// A model whose stream parks after the first delta of the given kind:
/// the run's stop must land on that delta, before transport scheduling
/// can publish more of the stream (the anthropic `FirstDelta` gate,
/// for both delta hooks); a driver that saves a scene mid-stream releases
/// the gate afterwards.
pub(crate) struct FirstDelta<M> {
    pub(crate) inner: M,
    pub(crate) tool: bool,
    pub(crate) release: Arc<Semaphore>,
}

impl<M: CompletionModel> CompletionModel for FirstDelta<M> {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        self.inner.completion(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, ProviderError> {
        let stream = self.inner.stream(request).await?;
        let provider = stream.provider().to_owned();
        let message_id = stream.message_id.clone();
        let tool = self.tool;
        let mut gated = StreamingCompletionResponse::from_events(
            provider,
            gate_events(Box::pin(stream), tool, self.release.clone()),
        );
        gated.message_id = message_id;
        Ok(gated)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }
}

fn gate_events(mut events: StreamEvents, tool: bool, release: Arc<Semaphore>) -> StreamEvents {
    Box::pin(async_stream::stream! {
        let mut crossed = false;
        while let Some(item) = events.next().await {
            let boundary = !crossed
                && item.as_ref().is_ok_and(|event| match event {
                    StreamEvent::BlockDelta {
                        delta: Delta::Text { text },
                        ..
                    } => !tool && !text.is_empty(),
                    event => tool && is_tool_call_progress(event),
                });
            yield item;
            if boundary {
                crossed = true;
                // Keep the provider stream while the run observes the
                // published delta and despawns its dispatch, or saves its
                // scene.
                release.acquire().await.expect("delivery gate open").forget();
            }
        }
    })
}

/// Whether a stream event is a tool call arriving: a name or arguments
/// delta on the wires that stream calls piecewise (the first of them is the
/// delta the hooks stop on), the call's close on a wire that streams it
/// whole (Gemini: a block start, then its end, no delta between).
pub(crate) fn is_tool_call_progress(event: &StreamEvent) -> bool {
    matches!(
        event,
        StreamEvent::BlockDelta {
            delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
            ..
        } | StreamEvent::BlockEnd {
            end: rig_core::streaming::BlockClose::ToolCall(_),
            ..
        }
    )
}

/// A tool that answers only once the driver's gate has a permit: the
/// window in which a run is stopped, or saved, with the tool in flight.
struct Parked<T> {
    inner: T,
    gate: Arc<Semaphore>,
}

impl<T: Tool> Tool for Parked<T> {
    const NAME: &'static str = T::NAME;
    type Error = T::Error;
    type Args = T::Args;
    type Output = T::Output;

    fn description(&self) -> String {
        self.inner.description()
    }

    fn parameters(&self) -> serde_json::Value {
        self.inner.parameters()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.gate.acquire().await.expect("tool gate open").forget();
        self.inner.call(context, args).await
    }
}

/// The delta-stop hooks despawn the stopped stream's effect: `Cancelled`
/// alone leaves issued work to its handler (CONTRACT §9.1), and the
/// producer's record is the cancel the engine made by dropping the stream.
fn stop_stream(world: &mut World, reason: &str, predicate: impl Fn(&StreamEvent) -> bool) {
    let mut query = world.query_filtered::<(Entity, &ChildOf, &Streamed), Without<EffectOutcome>>();
    let stops: Vec<_> = query
        .iter(world)
        .filter(|(_, _, stream)| stream.events.iter().any(&predicate))
        .map(|(entity, parent, _)| (entity, parent.parent()))
        .collect();
    for (effect, turn) in stops {
        let run = world
            .get::<ChildOf>(turn)
            .expect("stream turn belongs to run")
            .parent();
        world.entity_mut(run).insert(Cancelled(reason.into()));
        world.flush();
        assert!(
            matches!(&world.get::<Failed>(run).expect("native cancellation observer").0,Failure::Cancelled(report) if report.message==reason)
        );
        world.despawn(effect);
    }
}

fn stop_text_delta(world: &mut World) {
    stop_stream(world, corpus::STOP_ON_TEXT_DELTA, |event| {
        matches!(
            event,
            StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            } if !text.is_empty()
        )
    });
}

fn stop_tool_delta(world: &mut World) {
    stop_stream(world, corpus::STOP_ON_TOOL_CALL_DELTA, |event| {
        matches!(
            event,
            StreamEvent::BlockDelta {
                delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
                ..
            }
        )
    });
}

/// The run advances only once every effect a hook dispatched directly
/// under it (a startup note or lookup, a completion-call note, an outcome
/// note) has landed: the producer's hooks await their dispatches before
/// the run goes on, and so must the graph.
type ActiveRuns<'w, 's> = Query<'w, 's, Entity, (With<Run>, Without<Settled>, Without<Failed>)>;

fn direct_dispatches_landed(
    runs: ActiveRuns,
    effects: Query<(&ChildOf, Option<&EffectOutcome>), With<PendingEffect>>,
) -> bool {
    for run in &runs {
        if effects
            .iter()
            .any(|(parent, outcome)| parent.parent() == run && outcome.is_none())
        {
            return false;
        }
    }
    true
}

/// One pass of the schedule: what one `update` runs. A cell that saves a
/// checkpoint at a cut drives the schedule pass by pass.
pub(crate) fn one_pass(world: &mut World) {
    world.run_schedule(RigSchedule);
}

#[derive(Resource, Default)]
struct CheckpointCut;

/// Select the completed tool batch through the public durable commit and its
/// actual utterance links. Scheduling conditions remain independent checks:
/// this cut has no next fresh turn or answered effect still open.
fn at_cut(world: &mut World, run: Entity, tool_turns: usize) -> bool {
    let commits: Vec<_> = world
        .query::<(&ChildOf, &ToolTurnCommit, &TurnAssistant, &TurnResults)>()
        .iter(world)
        .filter(|(parent, _, _, _)| parent.parent() == run)
        .map(|(_, commit, assistant, results)| (commit.turn, assistant.0, results.0))
        .collect();
    if commits.len() != tool_turns
        || world.get::<rig_ecs::agent::RunPhase>(run) != Some(&rig_ecs::agent::RunPhase::Assembling)
    {
        return false;
    }
    if world
        .query_filtered::<&ChildOf, With<Fresh>>()
        .iter(world)
        .any(|parent| parent.parent() == run)
        || world
            .query_filtered::<(), (With<EffectOutcome>, With<InFlight>)>()
            .iter(world)
            .next()
            .is_some()
    {
        return false;
    }
    let latest = commits
        .iter()
        .map(|(turn, _, _)| *turn)
        .max()
        .expect("a tool-result cut has a committed turn");
    assert_eq!(
        world.get::<Cursor>(run).expect("run cursor").turn,
        latest,
        "the cursor agrees with the selected durable commit"
    );
    for (_, assistant, results) in commits {
        for (utterance, role) in [
            (assistant, rig_ecs::agent::Role::Assistant),
            (results, rig_ecs::agent::Role::User),
        ] {
            assert!(
                world.get::<Utterance>(utterance).is_some(),
                "commit links an actual utterance"
            );
            assert_eq!(
                world
                    .get::<ChildOf>(utterance)
                    .expect("utterance owner")
                    .parent(),
                run,
                "committed utterance belongs to this run"
            );
            assert_eq!(
                world.get::<rig_ecs::agent::Role>(utterance),
                Some(&role),
                "commit link role"
            );
        }
        assert!(
            crate::ecs_agent::sibling_index(world, assistant).expect("assistant order")
                < crate::ecs_agent::sibling_index(world, results).expect("results order"),
            "tool results follow their actual assistant utterance"
        );
    }
    true
}

/// The families whose cut is the #2514 hold after a committed tool turn
/// (`hold_after_tool_turn`, owner `matrix/checkpoint`): the checkpoint
/// rows and the long tool loop.
fn holds_tool_turn(cell: &Cell) -> bool {
    cell.name.starts_with("checkpoint_")
        || super::long_loop::is_long_loop(cell)
        || super::long_tasks::applicable(cell)
}

/// A tool adapter as an erased handler over the harness IO runtime.
fn tool_handler<S>(adapter: S, runtime: &tokio::runtime::Handle) -> ErasedHandler
where
    S: rig_core::serve::Serve<Family = rig_core::effect::family::Tool> + Send + Sync + 'static,
{
    ErasedHandler::new(RuntimeHandler {
        inner: Arc::new(adapter),
        runtime: runtime.clone(),
    })
}

/// Whether the cell's adder answers only on the driver's word.
fn parks_tool(cell: &Cell) -> bool {
    matches!(cell.fault, Some(Fault::StopWhileToolRuns)) || cell.scene == Scene::WithToolInFlight
}

/// The world over `wire`, with the cell's handlers registered in the
/// producer's order (memory, model, route, the host's note taker, tools,
/// a late route) and the program's agent graph spawned.
pub(crate) fn open<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    program: &Program,
) -> (App, Entity, EffectLogRecorder, Gates) {
    let gate = program
        .hooks
        .iter()
        .find_map(|hook| match hook {
            Hook::StopOnTextDelta => Some(false),
            Hook::StopOnToolCallDelta => Some(true),
            _ => None,
        })
        .or((cell.scene == Scene::MidStream).then_some(false));
    open_gated(wire, cell, program, gate)
}

/// [`open`] with the model's stream gated at its first delta (`Some(false)`
/// text, `Some(true)` tool call) or not (`None`), whatever the cell says:
/// for a driver that needs the stream parked at a cut the cell has no
/// hook for.
pub(crate) fn open_gated<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    program: &Program,
    gate: Option<bool>,
) -> (App, Entity, EffectLogRecorder, Gates) {
    open_inner(wire, cell, program, gate, None, None)
}

/// Bind fresh live handlers, optionally restoring a graph before installing
/// its hooks. The observation sink belongs to the host, not the saved world.
fn open_inner<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    program: &Program,
    gate: Option<bool>,
    scene: Option<&rig_ecs::checkpoint::Checkpoint>,
    witness: Option<Arc<rig_core::observe::ObservationLog>>,
) -> (App, Entity, EffectLogRecorder, Gates) {
    one_thread_pool();
    let policy = cell.bus.policy();
    let mut app = App::new();
    app.add_plugins(rig_ecs::RigPlugin::with_policy(policy));
    app.finish();
    app.cleanup();
    super::long_tasks::install_budget(&mut app, cell);
    let gates = Gates {
        tool: Arc::new(Semaphore::new(0)),
        stream: Arc::new(Semaphore::new(0)),
        witness: (cell.fault.is_some() || cell.reasoning.is_some()).then(|| match witness {
            Some(trace) => {
                rig_ecs::bus::Witnessing::install(app.world_mut(), trace.clone());
                trace
            }
            None => crate::stream_faults::witnessed(&mut app),
        }),
    };
    if cell.reasoning.is_some() {
        super::reasoning::install_delivery_observer(&mut app);
        app.add_systems(
            RigSchedule,
            super::reasoning::witness_deltas
                .after(BusSet::Collect)
                .before(RigSet::Fold),
        );
    }
    let recorder = if cell.events {
        EffectLogRecorder::keeping_stream_events()
    } else {
        EffectLogRecorder::new()
    };
    EffectLogResource::install(app.world_mut(), recorder.clone());
    if cell.name == "checkpoint_parallel_batch" {
        super::checkpoint_world::install_parallel(&mut app);
    }
    if super::stream_delivery::applicable(cell) {
        super::stream_delivery::install(&mut app);
    }
    let world = app.world_mut();
    let runtime = io_runtime();
    let layered =
        |handler: ErasedHandler, at: LayerAt| corpus::layered(handler, program, at, &None);
    let register_memory = |world: &mut World| -> Option<Entity> {
        if cell.memory == Memory::None {
            return None;
        }
        let adapter: ErasedHandler = match cell.memory {
            Memory::InMemory => ErasedHandler::new(RuntimeHandler {
                inner: Arc::new(MemoryAdapter::new(
                    rig_core::memory::InMemoryConversationMemory::new(),
                )),
                runtime: runtime.clone(),
            }),
            Memory::FailingAppend => ErasedHandler::new(RuntimeHandler {
                inner: Arc::new(MemoryAdapter::new(FailingMemory::append_fails())),
                runtime: runtime.clone(),
            }),
            Memory::FailingLoad => ErasedHandler::new(RuntimeHandler {
                inner: Arc::new(MemoryAdapter::new(FailingMemory::load_fails())),
                runtime: runtime.clone(),
            }),
            Memory::None => unreachable!(),
        };
        let adapter = layered(adapter, LayerAt::Memory);
        Some(
            Handlers::with(world, |handlers| {
                handlers.register_erased(format!("{OWNER}/memory"), adapter)
            })
            .expect("bus installed")
            .expect("fresh memory key"),
        )
    };
    // The producer's registration order: the builder registers the memory
    // before the model on its own bus; over a host's bus the host registers
    // the model first and the agent's memory at build.
    let mut memory = None;
    if cell.bus.declared() {
        memory = register_memory(world);
    }
    let model_handler = |model: M, label: &str| {
        let adapter = CompletionAdapter::new(label, model);
        ErasedHandler::new(RuntimeHandler {
            inner: Arc::new(adapter),
            runtime: runtime.clone(),
        })
    };
    // The default model as configuration data (CONTRACT §12, §13): a world
    // whose wire the harness can describe, and whose stream is not gated,
    // serves `golden/model:default` with a client the *host* builds from
    // that configuration (`WireBinding::handler`) rather than with the
    // wire's own adapter. Nothing of the recipe reaches the world, so the
    // scene a head saves carries only the execution descriptor; the host
    // rebuilds the same client for the restored world and hands it to
    // `load_world`, which installs it as part of the load. A gated model
    // (`FirstDelta`) and a wire the harness cannot describe stay
    // hand-registered on both sides.
    let default_key = HandlerKey::from(format!("{OWNER}/model:default"));
    let data_bound: Option<WireBinding> = gate.is_none().then(|| wire.binding()).flatten();
    // Built before the load and supplied to it, never installed behind its back.
    let mut supplied: Vec<(HandlerKey, ErasedHandler)> = Vec::new();
    let model: Option<Entity> = match &data_bound {
        Some(data) => {
            let handler = data.handler(&runtime);
            // The configuration captures the model id, base URL, credential
            // and transport; the host-built handler must describe itself
            // exactly as the hand-registered adapter over the wire's own
            // model would, so a wire with model-level settings the
            // configuration does not carry cannot diverge silently.
            let by_hand = CompletionAdapter::new(DEFAULT_LABEL, wire.model.clone()).descriptor();
            let built = handler.descriptor();
            assert_eq!(
                (&built.family, &built.layers),
                (&by_hand.family, &by_hand.layers),
                "the host-built descriptor is the hand-registered adapter's"
            );
            if scene.is_some() {
                // Installed by the load, atomically with the state it serves.
                supplied.push((default_key.clone(), handler));
                None
            } else {
                Some(
                    Handlers::with(world, |handlers| {
                        handlers.register_erased(default_key.clone(), handler)
                    })
                    .expect("bus installed")
                    .expect("fresh model key"),
                )
            }
        }
        None => {
            let model = match gate {
                Some(tool) => ErasedHandler::new(RuntimeHandler {
                    inner: Arc::new(CompletionAdapter::new(
                        DEFAULT_LABEL,
                        FirstDelta {
                            inner: wire.model.clone(),
                            tool,
                            release: gates.stream.clone(),
                        },
                    )),
                    runtime: runtime.clone(),
                }),
                None => model_handler(wire.model.clone(), DEFAULT_LABEL),
            };
            Some(
                Handlers::with(world, |handlers| {
                    handlers.register_erased(default_key.clone(), model)
                })
                .expect("bus installed")
                .expect("fresh model key"),
            )
        }
    };
    if !cell.bus.declared() {
        memory = register_memory(world);
    }
    let mut route = None;
    if let Some(label) = program.route {
        let handler = model_handler(wire.route(), label);
        route = Some(
            Handlers::with(world, |handlers| {
                handlers.register_erased(format!("{OWNER}/model:{label}"), handler)
            })
            .expect("bus installed")
            .expect("fresh route key"),
        );
    }
    if cell.notes {
        Handlers::with(world, |handlers| {
            handlers.register(
                corpus::NOTE_KEY,
                RuntimeHandler {
                    inner: Arc::new(NoteTaker),
                    runtime: runtime.clone(),
                },
            )
        })
        .expect("bus installed")
        .expect("fresh note key");
    }
    let mut tools = Vec::new();
    for (order, tool) in cell.tools.iter().enumerate() {
        let (name, adapter): (&str, ErasedHandler) = match tool {
            ToolKind::Adder if parks_tool(cell) => (
                "add",
                tool_handler(
                    ToolAdapter::new(Parked {
                        inner: Adder,
                        gate: gates.tool.clone(),
                    }),
                    &runtime,
                ),
            ),
            ToolKind::CheckpointStep => (
                "checkpoint_step",
                tool_handler(
                    ToolAdapter::new(super::checkpoint::CheckpointStep),
                    &runtime,
                ),
            ),
            ToolKind::CheckpointBatch => (
                "checkpoint_batch",
                tool_handler(super::checkpoint_world::parallel_adapter(world), &runtime),
            ),
            ToolKind::CheckpointLarge => (
                "checkpoint_large",
                tool_handler(
                    ToolAdapter::new(super::checkpoint::CheckpointLarge),
                    &runtime,
                ),
            ),
            // The repository tree is host state, leased per cell and rebound
            // unchanged to a restored world (`super::long_loop`).
            ToolKind::LongTask => (
                "task_operation",
                tool_handler(ToolAdapter::new(super::long_tasks::tool(cell)), &runtime),
            ),
            ToolKind::RepoListFiles => (
                "list_files",
                tool_handler(
                    ToolAdapter::new(super::long_loop::ListFiles(super::long_loop::repo(cell))),
                    &runtime,
                ),
            ),
            ToolKind::RepoReadFile => (
                "read_file",
                tool_handler(
                    ToolAdapter::new(super::long_loop::ReadFile(super::long_loop::repo(cell))),
                    &runtime,
                ),
            ),
            ToolKind::RepoWriteFile => (
                "write_file",
                tool_handler(
                    ToolAdapter::new(super::long_loop::WriteFile(super::long_loop::repo(cell))),
                    &runtime,
                ),
            ),
            ToolKind::RepoRunTests => (
                "run_tests",
                tool_handler(
                    ToolAdapter::new(super::long_loop::RunTests(super::long_loop::repo(cell))),
                    &runtime,
                ),
            ),
            ToolKind::Adder => ("add", tool_handler(ToolAdapter::new(Adder), &runtime)),
            ToolKind::BrokenAdder => ("add", tool_handler(ToolAdapter::new(FailingAdd), &runtime)),
            ToolKind::Alpha => (
                "lookup_harbor_label",
                tool_handler(ToolAdapter::new(AlphaSignal), &runtime),
            ),
            ToolKind::Beta => (
                "lookup_orchard_label",
                tool_handler(ToolAdapter::new(BetaSignal), &runtime),
            ),
            ToolKind::BrokenBeta => (
                "lookup_orchard_label",
                tool_handler(ToolAdapter::new(FailingOrchard), &runtime),
            ),
            ToolKind::WriteNote => (
                "write_note",
                tool_handler(ToolAdapter::new(WriteNote), &runtime),
            ),
            ToolKind::Lookup => {
                // The nesting tool is a key the world serves itself
                // (`corpus::world_nesting`): only its descriptor is shared.
                let nesting = program.nesting.expect("the nesting program");
                let descriptor = rig_core::serve::Serve::descriptor(&corpus::Lookup {
                    nesting,
                    model_key: HandlerKey::from(format!("{OWNER}/model:default")),
                });
                let entity = Handlers::with(world, |handlers| {
                    handlers.register_open(corpus::NESTING_TOOL_KEY, descriptor.family)
                })
                .expect("bus installed")
                .expect("fresh lookup key");
                corpus::world_nesting::install(world, nesting, OWNER);
                tools.push((order as u64, entity));
                continue;
            }
        };
        let key = format!("{OWNER}/tool:{name}#{order}");
        let adapter = if *tool == ToolKind::Adder {
            layered(adapter, LayerAt::Tool)
        } else {
            adapter
        };
        let entity = Handlers::with(world, |handlers| handlers.register_erased(key, adapter))
            .expect("bus installed")
            .expect("fresh tool key");
        tools.push((order as u64, entity));
    }
    if let Some(label) = program.late_route {
        let handler = model_handler(wire.route(), label);
        Handlers::with(world, |handlers| {
            handlers.register_erased(format!("{OWNER}/model:{label}"), handler)
        })
        .expect("bus installed")
        .expect("fresh late route key");
    }

    let agent = if let Some(scene) = scene {
        // The checkpoint's requirements name the saved execution descriptors
        // and nothing about how to build them; the host has already built
        // what it owns. Strict: a resumed cell serves exactly what it saved.
        let expected = supplied
            .iter()
            .map(|(key, handler)| (key.clone(), handler.descriptor()))
            .collect::<Vec<_>>();
        let loaded = load_world(scene, world, RestoreMode::Strict, supplied)
            .unwrap_or_else(|error| panic!("{}: the scene loads: {error}", cell.name));
        for (key, descriptor) in expected {
            // The host-built handler is the one serving the restored key,
            // under the key the checkpoint saved rather than the one its
            // adapter names itself.
            let bound = world
                .resource::<rig_ecs::bus::HandlerIndex>()
                .entity(&key)
                .and_then(|entity| world.get::<rig_ecs::bus::Bound>(entity))
                .unwrap_or_else(|| panic!("{}: `{key}` is bound after the load", cell.name));
            assert_eq!(
                (&bound.key, &bound.descriptor.key),
                (&key, &key),
                "{}: the restored model answers the saved key",
                cell.name
            );
            assert_eq!(
                (&bound.descriptor.family, &bound.descriptor.layers),
                (&descriptor.family, &descriptor.layers),
                "{}: the restored model is the host's",
                cell.name
            );
        }
        if super::stream_delivery::applicable(cell) {
            super::stream_delivery::assert_hydration(world, cell);
        }
        let run = loaded
            .with::<Run>(world)
            .first()
            .copied()
            .expect("the saved run");
        world.get::<RunOf>(run).expect("the loaded run's agent").0
    } else {
        let mode = match program.output_mode {
            None => OutputKind::Auto,
            Some(corpus::Output::Native) => OutputKind::Native,
            Some(corpus::Output::Tool) => OutputKind::Tool,
            Some(corpus::Output::Prompted) => OutputKind::Prompted,
        };
        let agent = world
            .spawn((
                Owner(OWNER.to_owned()),
                Preamble(program.preamble.map(str::to_owned)),
                Temperature(program.temperature),
                MaxTokens(program.max_tokens),
                AdditionalParams(program.additional_params.map(|params| params())),
                ToolChoiceSpec(program.tool_choice.map(corpus::Choice::tool_choice)),
                Output {
                    mode,
                    schema: program.output_schema.map(|schema| schema()),
                },
                DefaultMaxTurns(program.default_max_turns),
                MaxTurns(program.max_turns.or(program.default_max_turns).unwrap_or(1)),
                InvalidCalls {
                    retries: program.invalid_retries,
                    unhandled: match program.unhandled {
                        Unhandled::Fail => WorldUnhandled::Fail,
                        Unhandled::Ignore => WorldUnhandled::Ignore,
                    },
                },
                UsesModel(model.expect("a head world's model is bound")),
            ))
            .id();
        if let Some(retries) = cell.provider_retries {
            world.entity_mut(agent).insert(ProviderRetries(retries));
        }
        for (_, tool) in &tools {
            world.spawn((Grant(*tool), ChildOf(agent)));
        }
        if let Some(conversation) = program.conversation {
            world.entity_mut(agent).insert((
                Remembers(memory.expect("the cell remembers")),
                Conversation(conversation.into()),
            ));
        }
        if let Some(route) = route {
            world.spawn((Route(route), ChildOf(agent)));
        }
        let hooks = corpus::program_hooks(program, OWNER);
        world
            .entity_mut(agent)
            .insert(PolicyVersion(format!("ecs-matrix/v1:{}", hooks.join("+"))));

        agent
    };

    // The hooks as systems; the delta stops despawn their stream; a hook's
    // own dispatch gates the run.
    corpus::world_hooks::install(world, program);
    if program.hooks.contains(&Hook::StopOnTextDelta) {
        app.add_systems(
            RigSchedule,
            stop_text_delta.after(BusSet::Collect).before(RigSet::Fold),
        );
    }
    if program.hooks.contains(&Hook::StopOnToolCallDelta) {
        app.add_systems(
            RigSchedule,
            stop_tool_delta.after(BusSet::Collect).before(RigSet::Fold),
        );
    }
    app.configure_sets(
        RigSchedule,
        (
            RigSet::Advance.run_if(direct_dispatches_landed),
            RigSet::Assemble.run_if(direct_dispatches_landed),
            RigSet::Materialise.run_if(direct_dispatches_landed),
        ),
    );
    (app, agent, recorder, gates)
}

/// The run ended as the program says (the corpus's `assert_ending`, over
/// the world's own log for the answer).
fn assert_ending(world: &World, program: &Program, run: Entity, log: &EffectLog) {
    let ending = (
        world.get::<RunResult>(run).cloned(),
        world.get::<Failed>(run).cloned(),
    );
    match (&ending, program.ending) {
        ((None, Some(Failed(Failure::Cancelled(report)))), Ending::Cancelled(reason))
            if report.kind == ErrorKind::Cancelled && report.message == reason => {}
        ((Some(result), None), Ending::Answer) => {
            assert_eq!(
                result.0,
                program
                    .expected_output
                    .map_or_else(|| corpus::golden_answer(log), str::to_owned),
                "{}: the answer",
                program.fixture
            );
        }
        ((None, Some(Failed(Failure::MaxTurns { .. }))), Ending::MaxTurns)
        | ((None, Some(Failed(Failure::UnknownToolCall { .. }))), Ending::UnknownToolCall)
        | ((None, Some(Failed(Failure::Memory(_)))), Ending::MemoryError) => {}
        ((None, Some(Failed(Failure::Provider(report)))), Ending::ProviderError)
            if report.kind == ErrorKind::ProviderResponse => {}
        ((None, Some(Failed(Failure::Provider(report)))), Ending::Failed(kind))
        | ((None, Some(Failed(Failure::Tool(report)))), Ending::Failed(kind))
        | ((None, Some(Failed(Failure::Cancelled(report)))), Ending::Failed(kind))
            if report.kind == kind => {}
        (other, ending) => panic!(
            "{}: the run ends in {ending:?}, the world says {other:?}",
            program.fixture
        ),
    }
}

/// Tick until `run` ends, then to quiescence (a settled hook's dispatch,
/// an append, a stream a stop left to its handler still land after the
/// run ended). A cell with a cut drives pass by pass and saves the scene
/// there.
pub(crate) async fn drive_run(
    app: &mut App,
    run: Entity,
    cut_after: Option<usize>,
    cut: &mut Option<(rig_ecs::checkpoint::Checkpoint, usize, u64)>,
    recorder: &EffectLogRecorder,
) {
    drive_until(app, run, cut_after, cut, recorder, false).await;
}

async fn drive_until(
    app: &mut App,
    run: Entity,
    cut_after: Option<usize>,
    cut: &mut Option<(rig_ecs::checkpoint::Checkpoint, usize, u64)>,
    recorder: &EffectLogRecorder,
    pause_at_cut: bool,
) {
    let start = Instant::now();
    loop {
        if cut_after.is_some() && cut.is_none() {
            let checkpoint = app.world().contains_resource::<CheckpointCut>();
            if checkpoint {
                app.update();
            } else {
                one_pass(app.world_mut());
            }
            if let Some(tool_turns) = cut_after
                && at_cut(app.world_mut(), run, tool_turns)
            {
                if checkpoint {
                    let records = recorder.log().records;
                    let cursor = app.world().get::<Cursor>(run).unwrap().turn;
                    for _ in 0..3 {
                        app.update();
                    }
                    assert_eq!(
                        serde_json::to_value(recorder.log().records).unwrap(),
                        serde_json::to_value(records).unwrap(),
                        "held updates dispatch nothing"
                    );
                    assert_eq!(app.world().get::<Cursor>(run).unwrap().turn, cursor);
                }
                let next_id = app.world().resource::<IdCounter>().0;
                let scene = save_world(app.world_mut()).expect("every component serializes");
                let at = recorder.log().records.len();
                *cut = Some((scene, at, next_id));
                if pause_at_cut {
                    return;
                }
            }
        } else {
            app.update();
        }
        let world = app.world();
        if world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some() {
            break;
        }
        assert!(
            start.elapsed() < GUARD,
            "the run did not end within {GUARD:?}"
        );
        tokio::task::yield_now().await;
    }
    assert!(
        cut_after.is_none() || cut.is_some(),
        "the run ended before its cut"
    );
    // The append a settled run dispatches lands after `Settled`: wait for
    // its acknowledgement (or its recorded error) before reading the log,
    // as the agent's response boundary includes it.
    loop {
        app.update();
        if append_landed(app.world_mut(), run) {
            break;
        }
        assert!(
            start.elapsed() < GUARD,
            "the run's memory append did not land within {GUARD:?}"
        );
        tokio::task::yield_now().await;
    }
    settle_open_effects(app, start).await;
}

/// Tick until no effect is open, then let the observer settle deliveries.
async fn settle_open_effects(app: &mut App, start: Instant) {
    loop {
        app.update();
        let world = app.world_mut();
        let open = world
            .query_filtered::<(), (With<PendingEffect>, Without<EffectOutcome>)>()
            .iter(world)
            .count();
        if open == 0 {
            break;
        }
        assert!(
            start.elapsed() < GUARD,
            "{open} effects still open after the run ended"
        );
        tokio::task::yield_now().await;
    }
    // Deliveries the observer settles after the last outcome landed.
    for _ in 0..8 {
        app.update();
        tokio::task::yield_now().await;
    }
}

/// Whether the run's memory append, if it remembers and settled, has
/// landed (a run that failed appends nothing).
fn append_landed(world: &mut World, run: Entity) -> bool {
    use rig_core::effect::MemoryOp;

    if world.get::<rig_ecs::agent::Remembering>(run).is_none()
        || world.get::<Settled>(run).is_none()
    {
        return true;
    }
    let mut effects = world.query::<(&ChildOf, &PendingEffect, Option<&EffectOutcome>)>();
    effects.iter(world).any(|(parent, pending, outcome)| {
        parent.parent() == run
            && matches!(
                pending.kind,
                EffectKind::Memory {
                    op: MemoryOp::Append { .. }
                }
            )
            && outcome.is_some()
    })
}

/// Pass by pass until a tool of `run` is in flight and unanswered: the
/// parked tool waiting on the driver's gate.
async fn drive_to_parked_tool(app: &mut App, run: Entity) {
    let start = Instant::now();
    loop {
        one_pass(app.world_mut());
        let world = app.world_mut();
        let turns: Vec<Entity> = world
            .query_filtered::<(Entity, &ChildOf), With<Turn>>()
            .iter(world)
            .filter(|(_, parent)| parent.parent() == run)
            .map(|(turn, _)| turn)
            .collect();
        let parked = world
            .query_filtered::<(&ChildOf, &PendingEffect), (With<InFlight>, Without<EffectOutcome>)>(
            )
            .iter(world)
            .any(|(parent, pending)| {
                turns.contains(&parent.parent()) && pending.kind.family() == EffectFamily::Tool
            });
        if parked {
            // Let the handler task reach its gate before the driver acts.
            for _ in 0..4 {
                tokio::task::yield_now().await;
            }
            return;
        }
        assert!(
            world.get::<Settled>(run).is_none() && world.get::<Failed>(run).is_none(),
            "the run ended before its tool was in flight"
        );
        assert!(
            start.elapsed() < GUARD,
            "the tool was not in flight within {GUARD:?}"
        );
        tokio::task::yield_now().await;
    }
}

/// Pass by pass until the gated stream has published text: the parked
/// stream waiting on the driver's gate.
async fn drive_to_first_text(app: &mut App, run: Entity) {
    let start = Instant::now();
    loop {
        one_pass(app.world_mut());
        let world = app.world_mut();
        let published = world
            .query_filtered::<&Streamed, Without<EffectOutcome>>()
            .iter(world)
            .any(|stream| !stream.text.is_empty());
        if published {
            return;
        }
        assert!(
            world.get::<Settled>(run).is_none() && world.get::<Failed>(run).is_none(),
            "the run ended before its stream published text"
        );
        assert!(
            start.elapsed() < GUARD,
            "the stream published no text within {GUARD:?}"
        );
        tokio::task::yield_now().await;
    }
}

/// The run's graph has the shape the section states: one turn per
/// completion of the run, every effect under it answered (or gone), the
/// utterances the request history the last completion saw plus the
/// answer's, and one ending.
fn assert_graph(app: &mut App, runs: &[Entity], program: &Program, log: &EffectLog) {
    let run = *runs.last().expect("a run");
    let world = app.world_mut();
    let completions = log
        .records
        .iter()
        .filter(|record| {
            record.kind.family() == EffectFamily::Completion && record.parent.is_none()
        })
        .count();
    let turns: Vec<Entity> = world
        .query_filtered::<(Entity, &ChildOf), With<Turn>>()
        .iter(world)
        .filter(|(_, parent)| runs.contains(&parent.parent()))
        .map(|(turn, _)| turn)
        .collect();
    assert_eq!(
        turns.len(),
        completions,
        "{}: one turn per completion the runs dispatched",
        program.fixture
    );
    let mut effects = world.query::<(Entity, &ChildOf, &PendingEffect, Option<&EffectOutcome>)>();
    let under_run: Vec<_> = effects
        .iter(world)
        .filter(|(_, parent, _, _)| {
            runs.contains(&parent.parent()) || turns.contains(&parent.parent())
        })
        .map(|(entity, _, pending, outcome)| (entity, pending.kind.family(), outcome.is_some()))
        .collect();
    assert!(
        under_run.iter().all(|(_, _, answered)| *answered),
        "{}: every effect of the run has landed: {under_run:?}",
        program.fixture
    );
    let ended =
        world.get::<Settled>(run).is_some() as u8 + world.get::<Failed>(run).is_some() as u8;
    assert_eq!(ended, 1, "{}: a run has one ending", program.fixture);
    if program.ending == Ending::Answer
        && program.second_prompt.is_none()
        && !program.hooks.contains(&Hook::PatchHistoryFirst)
    {
        // The last request's history is the run's utterances before the
        // answer; the answer is the last utterance.
        let last = log
            .records
            .iter()
            .rev()
            .find_map(|record| match &record.kind {
                EffectKind::Completion { request, .. } if record.parent.is_none() => Some(request),
                _ => None,
            })
            .expect("a completion");
        let history = last
            .chat_history
            .iter()
            .filter(|message| !matches!(message, rig_core::message::Message::System { .. }))
            .count();
        let utterances = world
            .query_filtered::<&ChildOf, With<Utterance>>()
            .iter(world)
            .filter(|parent| parent.parent() == run)
            .count();
        assert_eq!(
            utterances,
            history + 1,
            "{}: the utterances are the last request's history and the answer",
            program.fixture
        );
    }
}

/// The live entities that are the run's to despawn: every entity but the
/// world's own — its resources (entities in this Bevy; the bus inserts
/// `hold::Transitions` on a run's first hold), its observers and its
/// registered systems. `Entities::len` is not this count: it also counts
/// ids the observers' triggers reserved and freed, and does not return to
/// its pre-spawn value once a witness is installed.
pub(crate) fn live_entities(world: &mut World) -> usize {
    world
        .query_filtered::<Entity, (
            Without<bevy_ecs::resource::IsResource>,
            Without<bevy_ecs::system::SystemIdMarker>,
            Without<bevy_ecs::observer::Observer>,
        )>()
        .iter(world)
        .count()
}

/// Every settled run is the world's to despawn: `despawn_run` returns
/// `Ok`, the live entities return to their count before the spawn, and
/// the agent's `Runs` no longer lists it.
fn assert_despawn(app: &mut App, agent: Entity, run: Entity, before: usize) {
    let world = app.world_mut();
    assert!(
        world
            .get::<Runs>(agent)
            .is_some_and(|runs| runs.runs().contains(&run)),
        "the agent lists its run"
    );
    world.despawn_run(run).expect("a settled run despawns");
    assert!(
        !world
            .get::<Runs>(agent)
            .is_some_and(|runs| runs.runs().contains(&run)),
        "the agent no longer lists the run"
    );
    let after = live_entities(world);
    assert_eq!(
        after, before,
        "the live entities return to the pre-spawn count"
    );
}

/// The batch's results, as the turns published them: every tool-result
/// utterance's call names, in `Order`, and none before every outcome of the
/// batch landed (CONTRACT §8.1: the results are one user utterance, in
/// call order, once every child has an outcome).
#[derive(Resource, Default)]
pub(crate) struct Published(pub Vec<Vec<String>>);

fn result_names(parts: &MessageParts, slots: &[rig_ecs::agent::ToolCallSlot]) -> Vec<String> {
    let MessageParts::User { content } = parts else {
        return vec![];
    };
    content
        .iter()
        .filter_map(|part| match part {
            rig_core::message::UserContent::ToolResult(result) => Some(
                slots
                    .iter()
                    .find(|slot| slot.id == result.call)
                    .expect("known tool result")
                    .name
                    .clone(),
            ),
            _ => None,
        })
        .collect()
}

type PublishedMessages<'w, 's> =
    Query<'w, 's, (Entity, &'static ChildOf), (With<Utterance>, Added<Utterance>)>;

fn observe_publication(
    messages: PublishedMessages,
    children: Query<&Children>,
    content: rig_ecs::agent::content::parts::ContentGraph,
    tools: Query<(&rig_ecs::agent::ToolCallSlot, Option<&EffectOutcome>)>,
    mut published: ResMut<Published>,
) {
    let slots: Vec<_> = tools.iter().map(|(slot, _)| slot.clone()).collect();
    let mut messages: Vec<_> = messages
        .iter()
        .map(|(entity, parent)| {
            let index = children
                .get(parent.parent())
                .ok()
                .and_then(|children| children.iter().position(|child| child == entity));
            (index, entity)
        })
        .collect();
    messages.sort();
    for (_, entity) in messages {
        let parts = content.message(entity).expect("valid published content");
        let names = result_names(&parts, &slots);
        if !names.is_empty() {
            assert!(
                tools.iter().all(|(_, outcome)| outcome.is_some()),
                "no batch result is published before every tool outcome lands"
            );
            published.0.push(names);
        }
    }
}

/// The two-tool cells' batch (§8.1, §10.1): both results surface in call
/// order after the batch settled, whatever the serving policy and the
/// concurrency.
fn assert_batch(app: &mut App, cell: &Cell) {
    let published = app.world().resource::<Published>().0.clone();
    let names: Vec<&str> = cell
        .tools
        .iter()
        .map(|tool| match tool {
            ToolKind::Alpha => "lookup_harbor_label",
            ToolKind::Beta | ToolKind::BrokenBeta => "lookup_orchard_label",
            other => panic!("a two-signal cell, not {other:?}"),
        })
        .collect();
    assert_eq!(
        published.len(),
        1,
        "{}: one tool-result utterance: {published:?}",
        cell.name
    );
    assert_eq!(
        published[0], names,
        "{}: the results in call order",
        cell.name
    );
}

/// Whether the cell is a two-signal batch cell (both tools of §8.1).
fn two_signals(cell: &Cell) -> bool {
    matches!(
        cell.tools,
        [ToolKind::Alpha, ToolKind::Beta] | [ToolKind::Alpha, ToolKind::BrokenBeta]
    )
}

/// The provider report a failed run carries.
fn provider_report(world: &World, run: Entity, what: &str) -> rig_core::error::ErrorReport {
    match world.get::<Failed>(run).map(|failed| &failed.0) {
        Some(Failure::Provider(report)) => report.clone(),
        other => panic!("{what}: a provider failure, not {other:?}"),
    }
}

/// The world's word on the fault beside the record (the oracle's): the
/// failure's facts, the stream's fold, the history that was never
/// committed.
fn assert_fault(app: &mut App, cell: &Cell, run: Entity, log: &EffectLog, gates: &Gates) {
    let Some(fault) = cell.fault else {
        return;
    };
    // The witness names the same ending the run has (§9.1, the #2495
    // funnel): `settled`, `cancelled`, `provider`, `memory`.
    let trace = gates
        .witness
        .as_deref()
        .expect("a failure-row cell is witnessed");
    let expected = match cell.program.ending {
        Ending::Answer => "settled",
        Ending::Cancelled(_) => "cancelled",
        Ending::MemoryError => "memory",
        Ending::ProviderError | Ending::Failed(_) => "provider",
        other => panic!("{}: no witness ending for {other:?}", cell.name),
    };
    let endings = crate::stream_faults::endings(trace);
    assert_eq!(
        endings.last().map(String::as_str),
        Some(expected),
        "{}: the witness's ending: {endings:?}",
        cell.name
    );
    let world = app.world_mut();
    let roles = utterance_roles(world, run);
    match fault {
        Fault::Setup { status, code }
        | Fault::Status {
            status,
            code,
            retry_after: _,
        } => {
            let report = provider_report(world, run, cell.name);
            assert_eq!(report.kind, ErrorKind::ProviderResponse, "{report:?}");
            assert_eq!(report.http_status, Some(status), "{report:?}");
            assert_eq!(
                report.retryable,
                rig_core::error::retryable_status(Some(status)),
                "the status table's verdict on a {status}: {report:?}"
            );
            assert_eq!(report.code.as_deref(), code, "{}: {report:?}", cell.name);
            let response = report
                .provider_response
                .as_ref()
                .expect("the reply is kept on the report");
            assert_eq!(response.status.map(|status| status.as_u16()), Some(status));
            assert!(!response.body.is_empty(), "the body is kept");
            if let Fault::Status {
                retry_after: true, ..
            } = fault
            {
                let headers = response
                    .headers
                    .as_ref()
                    .expect("the reply's headers are kept");
                assert_eq!(
                    headers
                        .get("retry-after")
                        .and_then(|value| value.to_str().ok()),
                    Some("1"),
                    "the retry hint survives onto the report: {headers:?}"
                );
            }
            // Every attempt is its own record, each the same failure.
            for record in &log.records {
                let recorded = record
                    .outcome
                    .as_ref()
                    .expect_err("the record's outcome is the provider's error");
                assert_eq!(recorded.kind, report.kind);
                assert_eq!(recorded.http_status, report.http_status);
                assert_eq!(recorded.retryable, report.retryable);
                assert_eq!(recorded.code, report.code);
            }
            let retries = cell
                .provider_retries
                .unwrap_or(rig_ecs::agent::DEFAULT_PROVIDER_RETRIES);
            let spent = if report.retryable { retries } else { 0 };
            assert_eq!(
                world
                    .get::<ProviderRetried>(run)
                    .map_or(0, |retried| retried.0),
                spent,
                "{}: the retries spent (CONTRACT §5)",
                cell.name
            );
            assert_eq!(
                log.records.len(),
                spent + 1,
                "{}: one record per attempt",
                cell.name
            );
            assert_eq!(roles, [Role::User], "only the prompt is history");
        }
        Fault::TruncatedAfterText | Fault::TruncatedAfterToolCall => {
            let report = provider_report(world, run, cell.name);
            assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
            assert_eq!(report.message, rig_core::serve::stream_truncated().message);
            let stream = sole_stream(world).expect("the stream's effect survived the run");
            assert!(
                stream.errors.is_empty(),
                "EOF is not an item: {:?}",
                stream.errors
            );
            assert!(stream.outcome.is_none(), "{:?}", stream.outcome);
            if fault == Fault::TruncatedAfterText {
                assert!(!stream.text.is_empty(), "the prefix is kept");
            } else {
                assert!(
                    stream.events.iter().any(is_tool_call_progress),
                    "the call streamed before the cut: {:?}",
                    stream.events
                );
            }
            assert_eq!(roles, [Role::User], "the cut turn is not history");
        }
        Fault::ErrorAfterText {
            code,
            message,
            status,
        } => {
            let report = provider_report(world, run, cell.name);
            assert_eq!(report.kind, ErrorKind::ProviderResponse, "{report:?}");
            assert_eq!(report.code.as_deref(), code, "the frame's code: {report:?}");
            assert!(
                report.message.contains(message),
                "the frame's message: {report:?}"
            );
            assert_eq!(report.http_status, status, "the frame's status: {report:?}");
            assert_eq!(
                report.retryable,
                rig_core::error::retryable_status(status),
                "the status table's verdict: {report:?}"
            );
            let recorded = log.records[0]
                .outcome
                .as_ref()
                .expect_err("the record holds the frame's error");
            assert_eq!(
                (recorded.code.as_deref(), recorded.http_status),
                (code, status)
            );
            let stream = sole_stream(world).expect("the stream's effect survived the run");
            assert!(!stream.text.is_empty(), "the prefix is kept");
            assert_eq!(stream.errors.len(), 1, "{:?}", stream.errors);
            assert_eq!(
                stream.errors[0].0,
                stream.events.len(),
                "the error item follows the delivered content"
            );
            assert_eq!(roles, [Role::User]);
        }
        Fault::Refusal | Fault::Filtered { with_text: true } => match cell.program.ending {
            Ending::Answer => {
                let answer = world
                    .get::<RunResult>(run)
                    .expect("the refusal is the answer")
                    .0
                    .clone();
                let stream = sole_stream(world).expect("the stream's effect survived the run");
                assert!(!answer.is_empty(), "the refusal text is the answer");
                assert_eq!(
                    answer, stream.text,
                    "the answer is the text the stream carried, whole"
                );
                assert_eq!(roles, [Role::User, Role::Assistant], "the turn is history");
                if matches!(fault, Fault::Filtered { .. }) {
                    let finish = match &log.records[0].outcome {
                        Ok(Outcome::Completion(response)) => response.finish_reason(),
                        other => panic!("{}: a completion record, not {other:?}", cell.name),
                    };
                    assert_eq!(
                        finish,
                        Some(rig_agent::completion::FinishReason::ContentFilter),
                        "the reason is on the record"
                    );
                }
            }
            Ending::Failed(kind) => {
                let report = provider_report(world, run, cell.name);
                assert_eq!(report.kind, kind, "{report:?}");
                assert!(!report.retryable, "a refusal is not retried: {report:?}");
                assert!(
                    report.message.contains("block_reason=SAFETY"),
                    "the block is named: {report:?}"
                );
                // The provider's verdict on the content is a refusal on the
                // report (CONTRACT §4), the block reason its code.
                assert!(report.refusal, "the block is a refusal: {report:?}");
                assert_eq!(report.code.as_deref(), Some("SAFETY"), "{report:?}");
                let recorded = log.records[0]
                    .outcome
                    .as_ref()
                    .expect_err("the record holds the refusal");
                assert!(recorded.refusal, "{recorded:?}");
                assert_eq!(roles, [Role::User]);
            }
            other => panic!(
                "{}: a refusal ends in Answer or Failed, not {other:?}",
                cell.name
            ),
        },
        Fault::Filtered { with_text: false } => {
            let report = provider_report(world, run, cell.name);
            assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
            assert!(
                report.message.contains(
                    &rig_agent::completion::FinishReason::ContentFilter.no_answer_message()
                ),
                "the answerless turn's reason and remedy: {report:?}"
            );
            assert!(
                log.records[0].outcome.is_ok(),
                "the provider answered; the run failed on what it answered"
            );
            assert_eq!(roles, [Role::User], "nothing is committed");
        }
        Fault::ToolError => {
            let outputs = tool_outputs(log);
            assert_eq!(
                outputs.len(),
                1,
                "{}: one tool record: {outputs:?}",
                cell.name
            );
            assert!(
                outputs[0].contains(BROKEN_ADD),
                "{}: the part is the error's model output: {outputs:?}",
                cell.name
            );
            let result = tool_result(log, 0);
            assert!(result.is_error(), "{result:?}");
            assert!(
                world.get::<RunResult>(run).is_some(),
                "the run answers around it"
            );
        }
        Fault::BatchSecondFails => {
            let outputs = tool_outputs(log);
            assert_eq!(
                outputs.len(),
                2,
                "{}: two tool records: {outputs:?}",
                cell.name
            );
            assert_eq!(
                outputs[0], ALPHA_SIGNAL_OUTPUT,
                "the first call's answer, in call order"
            );
            assert!(
                outputs[1].contains(BROKEN_ORCHARD),
                "the second call's part is the error's output: {outputs:?}"
            );
            assert!(!tool_result(log, 0).is_error());
            assert!(tool_result(log, 1).is_error());
            assert!(
                world.get::<RunResult>(run).is_some(),
                "the run answers around it"
            );
        }
        Fault::StopWhileToolRuns => {
            assert!(
                matches!(
                    world.get::<Failed>(run).map(|failed| &failed.0),
                    Some(Failure::Cancelled(report)) if report.message == CANCEL_ADD_OUTCOME
                ),
                "{:?}",
                world.get::<Failed>(run)
            );
            let outputs = tool_outputs(log);
            assert_eq!(outputs, ["42"], "the tool's record holds its real answer");
            // The call turn was history when the batch was issued; the
            // result never becomes history (CONTRACT §8.1).
            assert_eq!(
                roles,
                [Role::User, Role::Assistant],
                "the tool's result is not committed"
            );
        }
        Fault::FailingLoad => {
            let report = match world.get::<Failed>(run).map(|failed| &failed.0) {
                Some(Failure::Memory(report)) => report.clone(),
                other => panic!("{}: a memory failure, not {other:?}", cell.name),
            };
            assert_eq!(report.kind, ErrorKind::MemoryBackend, "{report:?}");
            let recorded = log.records[0]
                .outcome
                .as_ref()
                .expect_err("the memory record holds the refusal");
            assert_eq!(recorded.kind, report.kind);
            assert_eq!(recorded.message, report.message);
        }
    }
}

/// The `n`th tool record's result.
fn tool_result(log: &EffectLog, n: usize) -> &rig_core::tool::ToolResult {
    log.records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(Outcome::ToolResult { result }) => Some(result),
            _ => None,
        })
        .nth(n)
        .expect("a tool record")
}

/// A scene saved after the run failed loads in a fresh world served by the
/// log's replayers: the failure, the report and the history survive.
fn assert_failed_scene(
    app: &mut App,
    cell: &Cell,
    program: &Program,
    run: Entity,
    log: &EffectLog,
) {
    let world = app.world_mut();
    let failed = world.get::<Failed>(run).expect("the run failed").0.clone();
    let utterances = utterance_roles(world, run);
    let saved = save_world(world).expect("a failed run's scene saves");
    let mut program = *program;
    program.fixture = cell.name;
    let corpus::world::Opened { mut app, .. } =
        corpus::world::open(&program, log, RequestCheck::Payload);
    let world = app.world_mut();
    // The replayers the destination already serves are the whole recorded
    // implementation set: nothing is supplied, and `Strict` holds them to
    // the descriptors the failed run saved.
    let loaded = load_world(&saved, world, RestoreMode::Strict, [])
        .unwrap_or_else(|error| panic!("{}: a failed run's scene loads: {error}", cell.name));
    let run = loaded
        .with::<Run>(world)
        .first()
        .copied()
        .expect("the scene holds the run");
    let reloaded = world
        .get::<Failed>(run)
        .unwrap_or_else(|| panic!("{}: the loaded run stays failed", cell.name))
        .0
        .clone();
    // The scene holds the report's wire form, which carries no headers
    // (`ProviderResponseErrorWire`): compare what the scene can hold.
    let headerless = |failure: &Failure| {
        let mut failure = failure.clone();
        if let Failure::Provider(report) = &mut failure
            && let Some(response) = report.provider_response.as_mut()
        {
            response.headers = None;
        }
        failure
    };
    assert_eq!(
        headerless(&reloaded),
        headerless(&failed),
        "{}: the failure survives the scene",
        cell.name
    );
    assert_eq!(
        utterance_roles(world, run),
        utterances,
        "{}: the history survives the scene",
        cell.name
    );
    assert!(world.get::<RunResult>(run).is_none(), "no answer appears");
}

/// A scene saved while the stream is unfinished with observed progress is
/// refused before anything is spawned (CONTRACT §13).
fn assert_mid_stream_scene_refused(app: &mut App, cell: &Cell, program: &Program, log: &EffectLog) {
    let saved = save_world(app.world_mut()).expect("an in-flight stream's scene still saves");
    let mut program = *program;
    program.fixture = cell.name;
    let corpus::world::Opened { mut app, .. } =
        corpus::world::open(&program, log, RequestCheck::Payload);
    let error = load_world(&saved, app.world_mut(), RestoreMode::Strict, [])
        .err()
        .unwrap_or_else(|| {
            panic!(
                "{}: a scene with an unfinished stream is refused",
                cell.name
            )
        });
    assert!(
        error.message.contains("delivered progress"),
        "{}: refused for its progress, not another reason: {error:?}",
        cell.name
    );
    assert!(
        app.world_mut()
            .query::<&Run>()
            .iter(app.world())
            .next()
            .is_none(),
        "{}: nothing was spawned before the refusal",
        cell.name
    );
}

/// The world cell: the program over `wire` through `spawn_run`, its log,
/// its graph and its despawn asserted against the cell,
/// and its cut resumed where the cell names one.
pub(crate) async fn run_world<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    golden: impl FnOnce(&EffectLog),
) -> EffectLog {
    let program = wire.program(cell);
    let (mut app, mut agent, mut recorder, mut gates) = open(wire, cell, &program);
    let two_signals = two_signals(cell);
    if two_signals {
        app.init_resource::<Published>().add_systems(
            RigSchedule,
            observe_publication
                .after(RigSet::Materialise)
                .before(RigSet::Settle),
        );
    }
    let mut history: Vec<MessageParts> = program
        .history
        .map(|history| {
            history()
                .iter()
                .filter_map(MessageParts::from_message)
                .collect()
        })
        .unwrap_or_default();
    let first = match cell.image {
        Some(image) => super::image::user_content(image, program.prompt),
        None => vec![rig_core::message::UserContent::text(program.prompt)],
    };
    let prompts: Vec<Vec<rig_core::message::UserContent>> = std::iter::once(first)
        .chain(
            program
                .second_prompt
                .map(|prompt| vec![rig_core::message::UserContent::text(prompt)]),
        )
        .collect();
    let last = prompts.len() - 1;
    let mut runs = Vec::new();
    let mut cut = None;
    let mut before = Vec::new();
    let mut saved_head = None;
    let mut delivery_head = None;
    for (n, prompt) in prompts.into_iter().enumerate() {
        before.push(live_entities(app.world_mut()));
        let max_turns = if super::long_tasks::applicable(cell) {
            let current = recorder.log();
            let used = saved_head
                .as_ref()
                .into_iter()
                .flat_map(|head: &EffectLog| &head.records)
                .chain(&current.records)
                .filter(|record| matches!(record.kind, EffectKind::Completion { .. }))
                .count();
            Some(super::long_tasks::remaining_turns(used))
        } else {
            program.max_turns
        };
        let world = app.world_mut();
        let mut run = world.spawn_run(agent, &history, prompt, program.streamed, max_turns);
        if cell.name == "checkpoint_parallel_batch" {
            super::checkpoint_world::bind_parallel_run(world, run);
        }
        if let Some(concurrency) = program.tool_concurrency {
            world.entity_mut(run).insert(ToolPolicy { concurrency });
        }
        stamp_run(world, run, &recorder).expect("the run stamps its program identity");
        crate::goldens::capture_world_program(world, run, &recorder.log());
        if std::env::var("RIG_SPEC_DUMP").is_ok() {
            eprintln!(
                "SPECDUMP {}",
                serde_json::to_string(&rig_cassette::ecs::identity::spec_json(world, run)).unwrap()
            );
        }
        let cut_run = if super::long_tasks::applicable(cell) {
            0
        } else {
            last
        };
        let cut_after = (n == cut_run).then_some(cell.resume_after).flatten();
        if holds_tool_turn(cell)
            && let Some(cut_after) = cut_after
        {
            world.init_resource::<CheckpointCut>();
            hold_after_tool_turn(world, run, "matrix/checkpoint", cut_after).expect("arm cut");
        }

        if n == last && matches!(cell.fault, Some(Fault::StopWhileToolRuns)) {
            // The stop lands while the parked tool is in flight; the tool
            // is left to its handler and the run cannot despawn until it
            // lands.
            drive_to_parked_tool(&mut app, run).await;
            let world = app.world_mut();
            world
                .entity_mut(run)
                .insert(Cancelled(CANCEL_ADD_OUTCOME.to_owned()));
            world.flush();
            assert!(
                matches!(
                    world.get::<Failed>(run).map(|failed| &failed.0),
                    Some(Failure::Cancelled(report)) if report.message == CANCEL_ADD_OUTCOME
                ),
                "{}: the run is cancelled at once: {:?}",
                cell.name,
                world.get::<Failed>(run)
            );
            assert_eq!(
                world.despawn_run(run),
                Err(RunBusy::InFlight),
                "{}: the tool is still in flight",
                cell.name
            );
            gates.tool.add_permits(1);
            drive_run(&mut app, run, None, &mut cut, &recorder).await;
        } else if n == last && cell.scene == Scene::WithToolInFlight {
            // The scene is saved with the tool in flight and no stream
            // progress: an unanswered intent that restarts under its id.
            drive_to_parked_tool(&mut app, run).await;
            let next_id = app.world().resource::<IdCounter>().0;
            let scene = save_world(app.world_mut()).expect("an in-flight tool's scene saves");
            let at = recorder.log().records.len();
            cut = Some((scene, at, next_id));
            gates.tool.add_permits(1);
            drive_run(&mut app, run, None, &mut cut, &recorder).await;
        } else if n == last && cell.scene == Scene::MidStream {
            drive_to_first_text(&mut app, run).await;
            assert_mid_stream_scene_refused(&mut app, cell, &program, &recorder.log());
            gates.stream.add_permits(1);
            drive_run(&mut app, run, None, &mut cut, &recorder).await;
        } else if cell.live_resume && cut_after.is_some() {
            // The live resume consumes the cell's recording once: the first
            // world sends its head, and only the restored world can send
            // the tail. The cassette's strict HTTP match therefore checks
            // the resumed provider request (its history, images included),
            // beyond normalized effect-log replay.
            assert!(
                cut_after.is_some(),
                "{}: a live resume names its cut",
                cell.name
            );
            assert_eq!(n, 0);
            assert!(last == 0 || super::long_tasks::applicable(cell));
            drive_until(&mut app, run, cut_after, &mut cut, &recorder, true).await;
            let (scene, _, _) = cut.take().expect("the tool-result cut");
            let encoded_scene = serde_json::to_string(&scene).expect("scene JSON");
            let encoded_head = serde_json::to_string(&recorder.log()).expect("head JSON");
            if super::long_loop::is_long_loop(cell) {
                super::long_loop::write_cut_evidence(
                    cell,
                    cut_after.expect("cut number"),
                    &encoded_scene,
                    &encoded_head,
                );
            } else {
                super::checkpoint::write_cut_evidence(
                    cell,
                    cut_after.expect("cut number"),
                    &encoded_scene,
                    &encoded_head,
                );
            }
            let witness = gates.witness.clone();
            if cell.name == "checkpoint_parallel_batch" {
                super::checkpoint_world::assert_parallel_complete(app.world());
            }
            if super::stream_delivery::applicable(cell) {
                delivery_head = Some(super::stream_delivery::save_head(app.world(), cell));
            }
            drop(std::mem::replace(&mut app, App::new()));
            let scene: rig_ecs::checkpoint::Checkpoint =
                serde_json::from_str(&encoded_scene).expect("restore scene JSON");
            saved_head =
                Some(serde_json::from_str::<EffectLog>(&encoded_head).expect("restore head JSON"));
            let restore_started = Instant::now();
            (app, agent, recorder, gates) =
                open_inner(wire, cell, &program, None, Some(&scene), witness);
            if super::long_loop::is_long_loop(cell) {
                super::long_loop::write_restore_timing(
                    cell,
                    cut_after.expect("cut number"),
                    encoded_scene.len(),
                    encoded_head.len(),
                    restore_started.elapsed().as_micros(),
                );
            } else if cell.name.starts_with("checkpoint_") {
                eprintln!(
                    "CHECKPOINT_SCENE {}",
                    serde_json::json!({
                        "cell": cell.name, "cut": cut_after,
                        "scene_bytes": encoded_scene.len(), "head_bytes": encoded_head.len(),
                        "restore_us": restore_started.elapsed().as_micros(),
                    })
                );
            }
            run = app
                .world_mut()
                .query_filtered::<Entity, With<Run>>()
                .single(app.world())
                .expect("only the restored run exists");
            stamp_run(app.world_mut(), run, &recorder).expect("stamp restored run");
            if holds_tool_turn(cell) {
                assert!(
                    app.world().get::<ToolTurnHolds>(run).is_some(),
                    "scene retains hold"
                );
                for _ in 0..3 {
                    app.update();
                }
                assert!(
                    recorder.log().records.is_empty(),
                    "restored hold dispatches nothing"
                );
                release_tool_turn_hold(app.world_mut(), run, "matrix/checkpoint")
                    .expect("release restored cut");
            }
            drive_run(&mut app, run, None, &mut cut, &recorder).await;
        } else {
            drive_run(&mut app, run, cut_after, &mut cut, &recorder).await;
        }
        if n < last {
            assert!(
                app.world().get::<RunResult>(run).is_some(),
                "{}: the first run answers, the world says {:?}",
                cell.name,
                app.world().get::<Failed>(run)
            );
        }
        if n < last && super::long_tasks::applicable(cell) {
            super::long_tasks::assert_intermediate(cell);
            history = super::reasoning::assistant_history(app.world_mut(), run)
                .iter()
                .map(|message| MessageParts::from_message(message).expect("committed task history"))
                .collect();
        }
        runs.push(run);
    }
    let run = *runs.last().expect("a run");
    let mut log = recorder.log();
    if cell.name == "checkpoint_parallel_batch" && saved_head.is_none() {
        super::checkpoint_world::assert_parallel_complete(app.world());
    }
    if let Some(head) = saved_head {
        log = join_logs(head, log);
    }
    if !cell.families.is_empty() {
        assert_eq!(
            families(&log),
            cell.families,
            "{}: the record's families; the records: {:?}; the run: {:?}",
            cell.name,
            super::agent::record_summary(&log),
            app.world().get::<Failed>(run)
        );
    }
    assert_eq!(
        app.world().resource::<Policy>().0.serial_per_handler,
        cell.bus.policy().serial_per_handler
    );
    if super::stream_delivery::applicable(cell) {
        super::stream_delivery::assert_complete(app.world(), cell, &log, delivery_head);
    }
    golden(&log);
    assert_ending(app.world(), &program, run, &log);
    assert_graph(&mut app, &runs, &program, &log);
    if super::long_tasks::applicable(cell) {
        super::long_tasks::assert_world(app.world_mut(), &runs, &log);
    }
    if cell.reasoning.is_some() {
        super::reasoning::assert_log(cell, wire.thinking, &log);
        let history = super::reasoning::assistant_history(app.world_mut(), run);
        super::reasoning::assert_history(cell, &log, &history);
        if cell.reasoning == Some(super::cells::ReasoningCase::Capped) {
            let report = provider_report(app.world(), run, cell.name);
            assert!(
                report
                    .message
                    .contains(&rig_agent::completion::FinishReason::Length.no_answer_message()),
                "{report:?}"
            );
        }
        super::reasoning::assert_witness(
            cell,
            &log,
            gates.witness.as_deref().expect("reasoning is witnessed"),
        );
    }
    if cell.image.is_some() {
        super::image::assert_log(cell, &log);
        for run in &runs {
            let history = super::reasoning::assistant_history(app.world_mut(), *run);
            super::image::assert_history(
                cell,
                &history,
                &format!("{}: run {run:?} history", cell.name),
            );
        }
        let answer = &app
            .world()
            .get::<RunResult>(run)
            .expect("the run answered")
            .0;
        super::image::assert_answer(cell, answer);
        assert_eq!(*answer, corpus::golden_answer(&log));
    }
    if two_signals {
        assert_batch(&mut app, cell);
    }
    if super::long_loop::is_long_loop(cell) {
        super::long_loop::assert_log(cell, wire.thinking, &log);
        let history = super::reasoning::assistant_history(app.world_mut(), run);
        super::long_loop::assert_transcript(cell, &log, &history);
    }
    assert_fault(&mut app, cell, run, &log, &gates);
    // (3) the cut, resumed; the failed run's scene, loaded.
    if let Some((scene, at, next_id)) = cut.take() {
        resume(cell, &program, &log, scene, at, next_id);
    }
    if cell.scene == Scene::AfterFailure {
        assert_failed_scene(&mut app, cell, &program, run, &log);
    }
    // Every settled run despawns; the world keeps nothing of it.
    for (run, before) in runs.iter().rev().zip(before.iter().rev()) {
        assert_despawn(&mut app, agent, *run, *before);
    }
    log
}

fn join_logs(mut head: EffectLog, tail: EffectLog) -> EffectLog {
    for (key, family) in tail.header.signature.iter() {
        head.header.signature.insert_if_absent(key.clone(), *family);
    }
    for handler in tail.header.handlers {
        if let Some(previous) = head
            .header
            .handlers
            .iter()
            .find(|previous| previous.key == handler.key)
        {
            assert_eq!(
                previous, &handler,
                "a restored handler retains its descriptor"
            );
        } else {
            head.header.handlers.push(handler);
        }
    }
    for (scope, identity) in tail.header.programs {
        if let Some(previous) = head.header.programs.get(&scope) {
            assert_eq!(previous, &identity, "a restored scope retains its identity");
        } else {
            head.header.programs.insert(scope, identity);
        }
    }
    head.header.stream_errors.extend(tail.header.stream_errors);
    for limitation in tail.header.delivery_limitations {
        if !head.header.delivery_limitations.contains(&limitation) {
            head.header.delivery_limitations.push(limitation);
        }
    }
    if let Some(tail_deliveries) = tail.header.deliveries {
        let deliveries = head.header.deliveries.get_or_insert_with(Vec::new);
        let offset = deliveries
            .iter()
            .map(|delivery| delivery.batch)
            .max()
            .unwrap_or(0);
        // A restored world's passes begin again at one. Its first observed
        // pass follows the final head pass; it cannot share that boundary.
        deliveries.extend(tail_deliveries.into_iter().map(|mut delivery| {
            delivery.batch += offset;
            delivery
        }));
    }
    head.records.extend(tail.records);
    head
}

#[cfg(test)]
#[path = "world/tests.rs"]
mod tests;

/// Run each runtime over its own scripted transport and pin the world's log.
pub(crate) async fn run_scripted<M: CompletionModel + Clone + 'static>(
    cell: &Cell,
    wire: impl Fn() -> Wire<M>,
    golden: impl FnOnce(&EffectLog),
) -> EffectLog {
    super::agent::run_agent(&wire(), cell, |_| {}).await;
    run_world(&wire(), cell, golden).await
}

/// The tail: a fresh world over replayers of the log from the cut, the
/// scene loaded, the hooks installed after the load (no run-start observer
/// fires), the run ticked to the ending; the head's records to the cut
/// and the tail's from it are the whole log.
fn resume(
    cell: &Cell,
    program: &Program,
    log: &EffectLog,
    scene: rig_ecs::checkpoint::Checkpoint,
    at: usize,
    next_id: u64,
) {
    // The process image: the head's log and the checkpoint (the scene as
    // its state) as JSON, and nothing else.
    let mut head = log.clone();
    let tail_records = head.records.split_off(at);
    let (checkpoint, tail) = log.checkpoint(at, scene);
    assert_eq!(tail.records.len(), tail_records.len());
    let checkpoint: Checkpoint<rig_ecs::checkpoint::Checkpoint> =
        serde_json::from_str(&serde_json::to_string(&checkpoint).expect("serde"))
            .expect("a checkpoint restores");
    let scene = checkpoint.state.clone();
    let continuation =
        EffectLog::from_checkpoint(&checkpoint, tail).expect("the tail follows its checkpoint");
    let mut program = *program;
    program.fixture = cell.name;
    let corpus::world::Opened {
        mut app,
        handlers: tail_handlers,
        reached,
        asks,
    } = corpus::world::open(&program, &continuation, RequestCheck::Payload);
    // A key the head dispatched to before the cut and the continuation
    // never does has no replayer in the tail world, yet the checkpoint
    // still requires it: the host builds it from the whole recording and
    // supplies it to the load.
    let served: std::collections::HashSet<_> =
        tail_handlers.iter().map(|(key, _)| key.clone()).collect();
    let supplied: Vec<_> = scene
        .requirements()
        .expect("the scene's requirements")
        .into_iter()
        .filter(|descriptor| !served.contains(&descriptor.key))
        .map(|descriptor| {
            let handler = corpus::world::replayer_handler(
                &program,
                log,
                RequestCheck::Payload,
                &descriptor.key,
                &asks,
            );
            (descriptor.key, handler)
        })
        .collect();
    let world = app.world_mut();
    world.resource_mut::<IdCounter>().0 = next_id;
    // `Strict`: the resumed world serves exactly what the head saved.
    let loaded = load_world(&scene, world, RestoreMode::Strict, supplied)
        .unwrap_or_else(|error| panic!("{}: the scene's handlers are bound: {error}", cell.name));
    let run = loaded
        .with::<Run>(world)
        .first()
        .copied()
        .expect("the scene holds the run");
    if cell.reasoning.is_some() {
        let history = super::reasoning::assistant_history(world, run);
        super::reasoning::assert_history(cell, &head, &history);
    }
    corpus::world_hooks::install(world, &program);
    stamp_run(world, run, &world.resource::<EffectLogResource>().0.clone())
        .expect("the run stamps its program identity");
    let start = Instant::now();
    assert!(
        corpus::world::drive(&mut app, &program, run, start, &continuation, &reached),
        "{}: a resumed program does not cancel when reached",
        cell.name
    );
    corpus::world::assert_ending(&app, &program, run, log);
    let tail = app.world().resource::<EffectLogResource>().log();
    corpus::assert_same_records(&tail, &continuation, "world resume (tail)");
    let mut whole = head;
    whole.records.extend(tail.records);
    corpus::assert_same_records(&whole, log, "world resume");
}

/// The tool outcomes' rendered outputs, in record order.
pub(crate) fn tool_outputs(log: &EffectLog) -> Vec<String> {
    log.records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(Outcome::ToolResult { result }) => Some(result.output().render()),
            _ => None,
        })
        .collect()
}
