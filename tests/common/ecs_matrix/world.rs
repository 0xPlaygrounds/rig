//! The world cell: a cell's program as an agent graph in a Bevy `World`,
//! its hooks the corpus's systems (`corpus::world_hooks`), served by the
//! real adapters over the same cassette as the producer, and asserted in
//! this order: (1) the log the world wrote equals the producer's golden
//! (`crate::ecs_goldens::golden_effects`: kinds, outcomes, events, parent
//! chain — the rig-verify oracle), (2) the run's graph after settle has
//! the shape the section states, (3) where the cell names a cut, a scene
//! saved there loads in a fresh world served by replayers over the log's
//! tail and finishes to the same answer. Every settled run is then
//! despawned (`despawn_run`, CONTRACT: the world keeps nothing of a run by
//! itself).

use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy_app::{App, Update};
use bevy_ecs::prelude::*;
use futures::StreamExt;
use rig::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, ProviderCapabilities,
};
use rig::effect::{EffectFamily, EffectKind, HandlerKey, Outcome};
use rig::error::ErrorKind;
use rig::serve::{
    ErasedHandler,
    adapters::{CompletionAdapter, MemoryAdapter, ToolAdapter},
};
use rig::streaming::{Delta, StreamEvent, StreamEvents, StreamingCompletionResponse};
use rig_ecs::{
    agent::{
        AdditionalParams, Assembling, Cancelled, Conversation, Cursor, DefaultMaxTurns, Failed,
        Failure, Grant, InvalidCalls, MaxTokens, MaxTurns, MessageParts, Order, Output, OutputKind,
        Owner, PolicyVersion, Preamble, Remembers, Route, Run, RunOf, RunResult, Runs, Settled,
        Temperature, ToolChoiceSpec, ToolPolicy, Turn, Unhandled as WorldUnhandled, UsesModel,
        Utterance,
        scene::{WorldScene, load_world, save_world},
    },
    bus::{
        BusSet, EffectLogResource, EffectOutcome, Handlers, IdCounter, InFlight, Intake,
        PendingEffect, Policy, Progress, RigSchedule, Streamed, run_to_quiescence,
    },
    replay::{stamp_legacy_builder_header, stamp_run},
    systems::{Fresh, RigSet, despawn_run, install_agent, spawn_run},
};
use rig_effect_log::{Checkpoint, EffectLog, EffectLogRecorder, RequestCheck};
use tokio::sync::Semaphore;

use super::cells::{Cell, Memory, ToolKind};
use super::corpus::{self, Ending, Hook, LayerAt, Program, Unhandled};
use super::{OWNER, Wire};
use crate::ecs_agent::RuntimeHandler;
use crate::goldens::{Adder, FailingMemory, NoteTaker, WriteNote, families};
use crate::support::{AlphaSignal, BetaSignal};

const GUARD: Duration = Duration::from_secs(180);

/// Pin the bus's task pool to one thread, as rig-verify's world
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

/// A model whose stream parks after the first delta of the given kind:
/// the run's stop must land on that delta, before transport scheduling
/// can publish more of the stream (the anthropic `FirstToolDelta` gate,
/// for both delta hooks).
struct FirstDelta<M> {
    inner: M,
    tool: bool,
}

impl<M: CompletionModel> CompletionModel for FirstDelta<M> {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        self.inner.completion(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let stream = self.inner.stream(request).await?;
        let provider = stream.provider().to_owned();
        let message_id = stream.message_id.clone();
        let tool = self.tool;
        let mut gated = StreamingCompletionResponse::from_events(
            provider,
            gate_events(Box::pin(stream), tool, Arc::new(Semaphore::new(0))),
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
                    StreamEvent::BlockDelta { delta, .. } => match delta {
                        Delta::ToolName { .. } | Delta::ToolArguments { .. } => tool,
                        Delta::Text { text } => !tool && !text.is_empty(),
                        _ => false,
                    },
                    _ => false,
                });
            yield item;
            if boundary {
                crossed = true;
                // Keep the provider stream while the run observes the
                // published delta and despawns its dispatch.
                release.acquire().await.expect("delivery gate open").forget();
            }
        }
    })
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

/// One pass of the schedule, as `run_to_quiescence` runs one: the tick's
/// intake reset, progress cleared. A cell that saves a scene at a cut
/// drives the schedule pass by pass, since an `update` runs to quiescence
/// and would cross the cut.
pub(crate) fn one_pass(world: &mut World) {
    world.resource_mut::<Intake>().0 = 0;
    world.resource_mut::<Progress>().0 = false;
    world.run_schedule(RigSchedule);
}

/// Whether `run` is at the cut: `tool_turns` batches landed, the run
/// wants its next turn (`Assembling`, the cursor at `tool_turns`, no fresh
/// turn yet), and no answered effect is still open.
fn at_cut(world: &mut World, run: Entity, tool_turns: usize) -> bool {
    let assembling = world.get::<Assembling>(run).is_some()
        && world
            .get::<Cursor>(run)
            .is_some_and(|cursor| cursor.turn == tool_turns);
    if !assembling {
        return false;
    }
    let fresh = world
        .query_filtered::<&ChildOf, With<Fresh>>()
        .iter(world)
        .any(|child_of| child_of.parent() == run);
    if fresh {
        return false;
    }
    let open = world
        .query_filtered::<(), (With<EffectOutcome>, With<InFlight>)>()
        .iter(world)
        .count();
    open == 0
}

/// The world over `wire`, with the cell's handlers registered in the
/// producer's order (memory, model, route, the host's note taker, tools,
/// a late route) and the program's agent graph spawned.
pub(crate) fn open<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    program: &Program,
) -> (App, Entity, EffectLogRecorder) {
    one_thread_pool();
    let policy = cell.bus.policy();
    let mut app = App::new();
    rig_ecs::bus::Bus::with_policy(policy).install(app.world_mut());
    install_agent(app.world_mut());
    app.add_systems(Update, run_to_quiescence);
    app.finish();
    app.cleanup();
    let recorder = if cell.events {
        EffectLogRecorder::keeping_stream_events()
    } else {
        EffectLogRecorder::new()
    };
    EffectLogResource::install(app.world_mut(), recorder.clone());
    let world = app.world_mut();
    let runtime = tokio::runtime::Handle::current();
    let layered =
        |handler: ErasedHandler, at: LayerAt| corpus::layered(handler, program, at, &None);
    let register_memory = |world: &mut World| -> Option<Entity> {
        if cell.memory == Memory::None {
            return None;
        }
        let adapter: ErasedHandler = match cell.memory {
            Memory::InMemory => ErasedHandler::new(RuntimeHandler {
                inner: Arc::new(MemoryAdapter::new(
                    rig::memory::InMemoryConversationMemory::new(),
                )),
                runtime: runtime.clone(),
            }),
            Memory::FailingAppend => ErasedHandler::new(RuntimeHandler {
                inner: Arc::new(MemoryAdapter::new(FailingMemory::append_fails())),
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
    let delta_gate = program.hooks.iter().find_map(|hook| match hook {
        Hook::StopOnTextDelta => Some(false),
        Hook::StopOnToolCallDelta => Some(true),
        _ => None,
    });
    let model = match delta_gate {
        Some(tool) => ErasedHandler::new(RuntimeHandler {
            inner: Arc::new(CompletionAdapter::new(
                "default",
                FirstDelta {
                    inner: wire.model.clone(),
                    tool,
                },
            )),
            runtime: runtime.clone(),
        }),
        None => model_handler(wire.model.clone(), "default"),
    };
    let model = Handlers::with(world, |handlers| {
        handlers.register_erased(format!("{OWNER}/model:default"), model)
    })
    .expect("bus installed")
    .expect("fresh model key");
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
            ToolKind::Adder => (
                "add",
                ErasedHandler::new(RuntimeHandler {
                    inner: Arc::new(ToolAdapter::new(Adder)),
                    runtime: runtime.clone(),
                }),
            ),
            ToolKind::Alpha => (
                "lookup_harbor_label",
                ErasedHandler::new(RuntimeHandler {
                    inner: Arc::new(ToolAdapter::new(AlphaSignal)),
                    runtime: runtime.clone(),
                }),
            ),
            ToolKind::Beta => (
                "lookup_orchard_label",
                ErasedHandler::new(RuntimeHandler {
                    inner: Arc::new(ToolAdapter::new(BetaSignal)),
                    runtime: runtime.clone(),
                }),
            ),
            ToolKind::WriteNote => (
                "write_note",
                ErasedHandler::new(RuntimeHandler {
                    inner: Arc::new(ToolAdapter::new(WriteNote)),
                    runtime: runtime.clone(),
                }),
            ),
            ToolKind::Lookup => {
                // The nesting tool is a key the world serves itself
                // (`corpus::world_nesting`): only its descriptor is shared.
                let nesting = program.nesting.expect("the nesting program");
                let descriptor = rig::serve::Serve::descriptor(&corpus::Lookup {
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

    // The program as an agent graph, as the corpus's `spawn_agent` spawns it.
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
            UsesModel(model),
        ))
        .id();
    let mut order = 0u64;
    for (_, tool) in &tools {
        world.spawn((Grant(*tool), Order(order), ChildOf(agent)));
        order += 1;
    }
    if let Some(conversation) = program.conversation {
        world.entity_mut(agent).insert((
            Remembers(memory.expect("the cell remembers")),
            Conversation(conversation.into()),
        ));
    }
    if let Some(route) = route {
        world.spawn((Route(route), Order(order), ChildOf(agent)));
        order += 1;
    }
    world.resource_mut::<rig_ecs::agent::OrderCounter>().0 = order;
    let hooks = corpus::program_hooks(program, OWNER);
    if !hooks.is_empty() {
        world
            .entity_mut(agent)
            .insert(PolicyVersion(format!("ecs-matrix/v1:{}", hooks.join("+"))));
    }

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
    (app, agent, recorder)
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
        ((None, Some(Failed(Failure::MaxTurns { .. }))), Ending::MaxTurns) => {}
        ((None, Some(Failed(Failure::Provider(report)))), Ending::ProviderError)
            if report.kind == ErrorKind::ProviderResponse => {}
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
    cut: &mut Option<(WorldScene, usize, u64)>,
    recorder: &EffectLogRecorder,
) {
    let start = Instant::now();
    loop {
        if cut_after.is_some() && cut.is_none() {
            one_pass(app.world_mut());
            if let Some(tool_turns) = cut_after
                && at_cut(app.world_mut(), run, tool_turns)
            {
                let next_id = app.world().resource::<IdCounter>().0;
                let scene = save_world(app.world_mut()).expect("every component serializes");
                let at = recorder.log().records.len();
                *cut = Some((scene, at, next_id));
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
    use rig::effect::MemoryOp;
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
            .filter(|message| !matches!(message, rig::message::Message::System { .. }))
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
    despawn_run(world, run).expect("a settled run despawns");
    assert!(
        !world
            .get::<Runs>(agent)
            .is_some_and(|runs| runs.runs().contains(&run)),
        "the agent no longer lists the run"
    );
    let after = world.entities().len() as usize;
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

fn result_names(
    parts: &rig_ecs::agent::Parts,
    slots: &[rig_ecs::agent::ToolCallSlot],
) -> Vec<String> {
    let MessageParts::User { content } = &parts.0 else {
        return vec![];
    };
    content
        .iter()
        .filter_map(|part| match part {
            rig::message::UserContent::ToolResult(result) => Some(
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

type PublishedMessages<'w, 's> = Query<
    'w,
    's,
    (&'static Order, &'static rig_ecs::agent::Parts),
    (With<Utterance>, Added<rig_ecs::agent::Parts>),
>;

fn observe_publication(
    messages: PublishedMessages,
    tools: Query<(&rig_ecs::agent::ToolCallSlot, Option<&EffectOutcome>)>,
    mut published: ResMut<Published>,
) {
    let slots: Vec<_> = tools.iter().map(|(slot, _)| slot.clone()).collect();
    let mut messages: Vec<_> = messages.iter().collect();
    messages.sort_by_key(|(order, _)| order.0);
    for (_, parts) in messages {
        let names = result_names(parts, &slots);
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
            ToolKind::Beta => "lookup_orchard_label",
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

/// The world cell: the program over `wire` through `spawn_run`, its log
/// compared to the golden `golden`, its graph and its despawn asserted,
/// and its cut resumed where the cell names one.
pub(crate) async fn run_world<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    golden: impl FnOnce(&EffectLog),
) -> EffectLog {
    let program = wire.program(cell);
    let (mut app, agent, recorder) = open(wire, cell, &program);
    let two_signals = cell.tools == [ToolKind::Alpha, ToolKind::Beta];
    if two_signals {
        app.init_resource::<Published>().add_systems(
            RigSchedule,
            observe_publication
                .after(RigSet::Materialise)
                .before(RigSet::Settle),
        );
    }
    let declared = cell.bus.declared().then_some(cell.bus.policy());
    let history: Vec<MessageParts> = program
        .history
        .map(|history| {
            history()
                .iter()
                .filter_map(MessageParts::from_message)
                .collect()
        })
        .unwrap_or_default();
    let prompts: Vec<&str> = std::iter::once(program.prompt)
        .chain(program.second_prompt)
        .collect();
    let last = prompts.len() - 1;
    let mut runs = Vec::new();
    let mut cut = None;
    let mut before = Vec::new();
    for (n, prompt) in prompts.into_iter().enumerate() {
        before.push(app.world().entities().len() as usize);
        let world = app.world_mut();
        let run = spawn_run(
            world,
            agent,
            &history,
            prompt,
            program.streamed,
            program.max_turns,
        );
        if let Some(concurrency) = program.tool_concurrency {
            world.entity_mut(run).insert(ToolPolicy { concurrency });
        }
        if n == 0 {
            stamp_legacy_builder_header(
                world,
                agent,
                &recorder,
                declared,
                corpus::program_hooks(&program, OWNER),
            );
        }
        stamp_run(world, run, &recorder).expect("the run stamps its program identity");
        let cut_after = (n == last).then_some(cell.resume_after).flatten();
        drive_run(&mut app, run, cut_after, &mut cut, &recorder).await;
        if n < last {
            assert!(
                app.world().get::<RunResult>(run).is_some(),
                "{}: the first run answers, the world says {:?}",
                cell.name,
                app.world().get::<Failed>(run)
            );
        }
        runs.push(run);
    }
    let run = *runs.last().expect("a run");
    let log = recorder.log();
    if !cell.families.is_empty() {
        assert_eq!(
            families(&log),
            cell.families,
            "{}: the record's families",
            cell.name
        );
    }
    assert_eq!(
        app.world().resource::<Policy>().0.serial_per_handler,
        cell.bus.policy().serial_per_handler
    );
    // (1) the record is the producer's: the caller names the golden at its
    // call site (`crate::ecs_goldens::golden_effects("…", log)`).
    golden(&log);
    // (2) the graph.
    assert_ending(app.world(), &program, run, &log);
    assert_graph(&mut app, &runs, &program, &log);
    if two_signals {
        assert_batch(&mut app, cell);
    }
    // (3) the cut, resumed.
    if let Some((scene, at, next_id)) = cut.take() {
        resume(cell, &program, &log, scene, at, next_id);
    }
    // Every settled run despawns; the world keeps nothing of it.
    for (run, before) in runs.iter().rev().zip(before.iter().rev()) {
        assert_despawn(&mut app, agent, *run, *before);
    }
    log
}

/// The tail: a fresh world over replayers of the log from the cut, the
/// scene loaded, the hooks installed after the load (no run-start observer
/// fires), the run ticked to the ending; the head's records to the cut
/// and the tail's from it are the whole log.
fn resume(
    cell: &Cell,
    program: &Program,
    log: &EffectLog,
    scene: WorldScene,
    at: usize,
    next_id: u64,
) {
    // The process image: the head's log and the checkpoint (the scene as
    // its state) as JSON, and nothing else.
    let mut head = log.clone();
    let tail_records = head.records.split_off(at);
    let (checkpoint, tail) = log.checkpoint(at, scene);
    assert_eq!(tail.records.len(), tail_records.len());
    let checkpoint: Checkpoint<WorldScene> =
        serde_json::from_str(&serde_json::to_string(&checkpoint).expect("serde"))
            .expect("a checkpoint restores");
    let scene = checkpoint.state.clone();
    let continuation =
        EffectLog::from_checkpoint(&checkpoint, tail).expect("the tail follows its checkpoint");
    let mut program = *program;
    program.fixture = cell.name;
    let corpus::world::Opened {
        mut app,
        handlers: _,
        reached,
    } = corpus::world::open(&program, &continuation, RequestCheck::Payload);
    let world = app.world_mut();
    world.resource_mut::<IdCounter>().0 = next_id;
    let loaded = load_world(&scene, world).expect("the scene's handlers are bound");
    let run = loaded
        .graph
        .iter()
        .copied()
        .find(|entity| world.get::<Run>(*entity).is_some())
        .expect("the scene holds the run");
    let agent = world.get::<RunOf>(run).expect("the run's agent").0;
    corpus::world_hooks::install(world, &program);
    stamp_legacy_builder_header(
        world,
        agent,
        &world.resource::<EffectLogResource>().0.clone(),
        log.header.bus,
        corpus::program_hooks(&program, OWNER),
    );
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
