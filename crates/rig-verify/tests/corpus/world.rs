//! The corpus's third interpreter, agent half: a `Program` as an agent
//! graph in a Bevy `World`, its run through `rig_ecs`'s systems against
//! the golden's replayers, its log compared to the golden as the other two
//! interpreters' are.
//!
//! What this interpreter supports is what stages 2 and 3 build: a
//! completion-and-tools program with no hooks, no memory, no routes, no
//! retrieval, no layers, one prompt — the `lookup` tool's nesting included
//! (a key the world serves: `world_nesting`). Every other program is
//! reported `unsupported` with the set or entity it waits for, and passes;
//! the union of those lines over the corpus is the status table the PR
//! prints.

use std::time::{Duration, Instant};

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{
    effect::{EffectFamily, HandlerKey},
    error::ErrorKind,
    serve::ServingPolicy,
};
use rig_ecs::{
    agent::{
        AdditionalParams, Context, Conversation, DefaultMaxTurns, DocumentId, DocumentText, Failed,
        Failure, Grant, InvalidCalls, MaxTokens, MaxTurns, MessageParts, Order, Output, OutputKind,
        Owner, Preamble, Remembers, Retrievable, Retrieval, RetrievalKind, Retrieves, RunResult,
        Settled, Temperature, ToolChoiceSpec, ToolPolicy, Unhandled as WorldUnhandled, UsesModel,
    },
    bus::{BusPlugin, EffectLogResource, EffectOutcome, Handlers, IdCounter, PendingEffect},
    replay::stamp_header,
    systems::{AgentPlugin, spawn_run},
};
use rig_effect_log::{EffectLogRecorder, EffectLogReplayer};

use super::{
    Ending, Output as CorpusOutput, Program, Unhandled, assert_same_records, golden, golden_answer,
    keeps_events, run_spec,
};

const GUARD: Duration = Duration::from_secs(30);

/// The world interpreter's cell: replays `program` through a world — every
/// program of the corpus, the two-run ones as two runs on one agent.
pub fn world_agent_reproduces(program: &Program) {
    let log = golden(program.fixture);
    EffectLogReplayer::check_header(&log).expect("a current format");
    // A host-bus golden names no policy: the replay's host runs the
    // producer's where the program names it, as `Replay::open` does.
    let mut policy = log.header.bus.unwrap_or_default();
    if program.host_serial {
        assert!(log.header.bus.is_none(), "a host-bus program");
        policy.serial_per_handler = true;
    }
    // One pool thread, so same-key dispatches reach their replayer in
    // dispatch order (see the registration below). Process-wide: nextest
    // runs each cell in its own process.
    bevy_tasks::IoTaskPool::get_or_init(|| {
        bevy_tasks::TaskPoolBuilder::new()
            .num_threads(1)
            .thread_name("world-agent-io".to_owned())
            .build()
    });
    let mut app = App::new();
    app.add_plugins((
        BusPlugin::with_policy(ServingPolicy {
            command_capacity: 1_000,
            ..policy
        })
        .ambiguity_detection(LogLevel::Error),
        AgentPlugin::default(),
    ));
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    // rig-bus mints from 1; so does the world here, so ids read alike.
    world.resource_mut::<IdCounter>().0 = 1;

    // The golden's replayers, by position per key, as the other
    // interpreters register them: rig-bus minted an id for a dispatch a
    // hook or a layer then denied, so the ids do not align with the
    // world's, which mints only what it dispatches. Position holds because
    // the pool below is one thread: a handler task's first poll — where the
    // replayer pops its record — comes in spawn order, which is `Seq` order.
    // The world a suspending layer asks: a thread answering as the program
    // says, signalling when it holds an answer forever.
    let reached = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let asks = program
        .layers
        .iter()
        .find_map(|spec| match spec.layer {
            super::LayerKind::Approval(answer) => Some(answer),
            _ => None,
        })
        .map(|answer| approval_world(answer, std::sync::Arc::clone(&reached)));
    let owner = program.owner.to_owned();
    let layer_at = |key: &HandlerKey| -> Option<super::LayerAt> {
        match key.as_str() {
            k if k == format!("{owner}/tool:add#0") => Some(super::LayerAt::Tool),
            k if k == format!("{owner}/model:default") => Some(super::LayerAt::Model),
            k if k == format!("{owner}/memory") => Some(super::LayerAt::Memory),
            super::NOTE_KEY => Some(super::LayerAt::Note),
            _ => None,
        }
    };
    let mut handler_entities: Vec<(HandlerKey, Entity)> = Vec::new();
    Handlers::with(world, |handlers| {
        for replayer in EffectLogReplayer::for_log(&log).expect("the golden's replayers") {
            let key = replayer.key().clone();
            // The program's layers, on the handler exactly as `Replay::open`
            // wraps them: the world registers the layered handler.
            if let Some(at) = layer_at(&key)
                && program.layers.iter().any(|spec| spec.at == at)
            {
                let handler = super::layered(
                    rig_core::serve::ErasedHandler::new(replayer),
                    program,
                    at,
                    &asks,
                );
                let entity = handlers
                    .register_erased(key.clone(), handler)
                    .expect("a fresh key");
                handler_entities.push((key, entity));
                continue;
            }
            // The nesting program's keys are program, not record: the world
            // serves them itself (`world_nesting`), the replayers answer
            // only the leaves.
            if program.nesting.is_some() && super::world_nesting::is_served_by_the_world(&key) {
                let entity = handlers
                    .register_open(
                        key.clone(),
                        rig_core::serve::Serve::descriptor(&replayer).family,
                    )
                    .expect("a fresh key");
                handler_entities.push((key, entity));
                continue;
            }
            let entity = handlers
                .register_erased(key.clone(), rig_core::serve::ErasedHandler::new(replayer))
                .expect("a fresh key");
            handler_entities.push((key, entity));
        }
    })
    .expect("a bus");
    if let Some(nesting) = program.nesting {
        super::world_nesting::install(world, nesting, program.owner);
    }
    super::world_hooks::install(world, program);
    let recorder = if keeps_events(&log) {
        EffectLogRecorder::keeping_stream_events()
    } else {
        EffectLogRecorder::new()
    };
    EffectLogResource::install(world, recorder);

    let agent = spawn_agent(world, program, &handler_entities);
    stamp_header(
        world,
        agent,
        &world.resource::<EffectLogResource>().0.clone(),
        log.header.bus,
        super::program_hooks(program, program.owner),
    );
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
    let start = Instant::now();
    for (n, prompt) in prompts.into_iter().enumerate() {
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
        rig_ecs::replay::stamp_run(world, run, &world.resource::<EffectLogResource>().0.clone());
        if !drive(&mut app, program, run, start, &log, &reached) {
            return;
        }
        // The first of two runs answers; the program's ending is the last's.
        if n < last {
            let result = app.world().get::<RunResult>(run).cloned();
            assert!(
                result.is_some(),
                "{}: the first run answers, the world says {:?}",
                program.fixture,
                app.world().get::<Failed>(run)
            );
        } else {
            assert_ending(&app, program, run, &log);
        }
    }

    let world = app.world();
    // The world's log is in begin order, as rig-bus's; the oracle asserts
    // it is dispatch order. The records carry the run's `Scope`, which the
    // goldens do not have and `as_data` does not compare.
    let replayed = world.resource::<EffectLogResource>().log();
    assert_same_records(&replayed, &log, "world agent");
    assert_header(&replayed, &log, program);
}

/// Tick the app until `run` ends and the world is quiescent. `false` when
/// the program's cancel-when-reached dropped the run (the records were
/// asserted; nothing more runs).
fn drive(
    app: &mut App,
    program: &Program,
    run: Entity,
    start: Instant,
    log: &rig_effect_log::EffectLog,
    reached: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> bool {
    loop {
        app.update();
        if program.cancel_when_reached
            && (super::world_nesting::reached(app.world_mut())
                || reached.load(std::sync::atomic::Ordering::SeqCst))
        {
            // The producer dropped the run once the never-answering handler
            // was reached: the run entity goes, and the whole tree with it
            // — the tool call and its child are cancelled, a queued child
            // never begins.
            app.world_mut().despawn(run);
            app.update();
            app.update();
            let replayed = app.world().resource::<EffectLogResource>().log();
            assert_same_records(&replayed, log, "world agent");
            return false;
        }
        let world = app.world();
        if world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some() {
            break;
        }
        assert!(
            start.elapsed() < GUARD,
            "{}: the run did not end within {GUARD:?}",
            program.fixture
        );
        std::thread::yield_now();
    }
    // To quiescence: a settled hook's dispatch, a stream a stop left to its
    // handler, still land after the run ended.
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
            "{}: {open} effects still open after the run ended",
            program.fixture
        );
        std::thread::yield_now();
    }
    true
}

/// The run ended as the program says.
fn assert_ending(app: &App, program: &Program, run: Entity, log: &rig_effect_log::EffectLog) {
    let world = app.world();
    let ending = (
        world.get::<RunResult>(run).cloned(),
        world.get::<Failed>(run).cloned(),
    );
    match (&ending, program.ending) {
        // The producer dropped its stream at the first delta; the replayer
        // answers the record as the cancel it was, and the run ends there.
        ((None, Some(Failed(Failure::Cancelled(report)))), _)
            if program.cancel_after_first_delta && report.kind == ErrorKind::Cancelled => {}
        // A hook's stop: the reason is the run's.
        ((None, Some(Failed(Failure::Cancelled(report)))), Ending::Cancelled(reason))
            if report.kind == ErrorKind::Cancelled && report.message == reason => {}
        ((Some(result), None), Ending::Answer) => {
            assert_eq!(
                result.0,
                program
                    .expected_output
                    .map_or_else(|| golden_answer(log), str::to_owned),
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

/// The replayed header is the golden's: spec hash, hooks, required row,
/// signature; and the world's identity computation agrees with the harness.
fn assert_header(
    replayed: &rig_effect_log::EffectLog,
    log: &rig_effect_log::EffectLog,
    program: &Program,
) {
    assert_eq!(
        replayed.header.run_spec, log.header.run_spec,
        "{}: the header's spec hash is this program's",
        program.fixture
    );
    assert_eq!(
        replayed.header.hooks, log.header.hooks,
        "{}: the hook list",
        program.fixture
    );
    for (key, family) in log.header.required.iter() {
        assert_eq!(
            replayed.header.required.get(key),
            Some(family),
            "{}: the required row names `{key}`",
            program.fixture
        );
    }
    assert_eq!(
        replayed.header.signature, log.header.signature,
        "{}: the signature",
        program.fixture
    );
    // The world's own identity computation agrees with the harness's.
    let expected = rig_effect_log::stable_hash(&rig_agent::run::RunSpec {
        max_turns: Some(program.default_max_turns.unwrap_or(1)),
        max_invalid_tool_call_retries: 0,
        unhandled_invalid_tool_call: rig_agent::run::UnhandledInvalidToolCall::Fail,
        ..run_spec(program)
    })
    .ok();
    assert_eq!(
        replayed.header.run_spec, expected,
        "{}: run_spec",
        program.fixture
    );
}

/// The program as an agent graph: the agent entity with one component per
/// setting, `UsesModel` to the golden's model handler entity, a grant per
/// advertised tool (the required row's tool keys, in key order), a context
/// link per static document.
fn spawn_agent(world: &mut World, program: &Program, handlers: &[(HandlerKey, Entity)]) -> Entity {
    let model_key = HandlerKey::from(format!("{}/model:default", program.owner));
    let model = handlers
        .iter()
        .find(|(key, _)| *key == model_key)
        .map(|(_, entity)| *entity)
        .expect("the golden serves the model");
    let mode = match program.output_mode {
        None => OutputKind::Auto,
        Some(CorpusOutput::Native) => OutputKind::Native,
        Some(CorpusOutput::Tool) => OutputKind::Tool,
        Some(CorpusOutput::Prompted) => OutputKind::Prompted,
    };
    let agent = world
        .spawn((
            Owner(program.owner.to_owned()),
            Preamble(program.spec_preamble()),
            Temperature(program.temperature),
            MaxTokens(program.max_tokens),
            AdditionalParams(program.additional_params.map(|params| params())),
            ToolChoiceSpec(program.tool_choice.map(super::Choice::tool_choice)),
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
    for (n, text) in program.context.iter().enumerate() {
        let document = world
            .spawn((
                DocumentId(format!("static_doc_{n}")),
                DocumentText((*text).to_owned()),
            ))
            .id();
        world.spawn((Context(document), Order(order), ChildOf(agent)));
        order += 1;
    }
    let log = golden(program.fixture);
    let mut tools: Vec<&HandlerKey> = log
        .header
        .required
        .iter()
        .filter(|(_, family)| **family == EffectFamily::Tool)
        .map(|(key, _)| key)
        .collect();
    tools.sort();
    for key in tools {
        let tool = handlers
            .iter()
            .find(|(bound, _)| bound == key)
            .map(|(_, entity)| *entity)
            .expect("the golden describes every required tool");
        let name = key
            .as_str()
            .rsplit_once('/')
            .and_then(|(_, tail)| tail.strip_prefix("tool:"))
            .and_then(|tail| tail.rsplit_once('#'))
            .map(|(name, _)| name)
            .expect("a tool key names its tool");
        let mut grant = world.spawn((Grant(tool), Order(order), ChildOf(agent)));
        if program.retrievable.contains(&name) {
            grant.insert(Retrievable);
        }
        order += 1;
    }
    // The indexes, in the producer's order: the context index first, then
    // the tool index (`dynamic_context` before `retrieved_tools`).
    let handler = |key: HandlerKey| {
        handlers
            .iter()
            .find(|(bound, _)| *bound == key)
            .map(|(_, entity)| *entity)
    };
    if let Some(samples) = program.dynamic_context {
        let index = handler(HandlerKey::from(format!(
            "{}/retrieve:context#0",
            program.owner
        )))
        .expect("the golden serves the context index");
        world.spawn((
            Retrieves(index),
            Retrieval {
                samples: samples as u64,
                what: RetrievalKind::Documents,
            },
            Order(order),
            ChildOf(agent),
        ));
        order += 1;
    }
    if let Some(samples) = program.retrieved_tools {
        let index = handler(HandlerKey::from(format!(
            "{}/retrieve:tools#0",
            program.owner
        )))
        .expect("the golden serves the tool index");
        world.spawn((
            Retrieves(index),
            Retrieval {
                samples: samples as u64,
                what: RetrievalKind::Tools,
            },
            Order(order),
            ChildOf(agent),
        ));
        order += 1;
    }
    if let Some(conversation) = program.conversation {
        let memory = handler(HandlerKey::from(format!("{}/memory", program.owner)))
            .expect("the golden serves the memory");
        world
            .entity_mut(agent)
            .insert((Remembers(memory), Conversation(conversation.to_owned())));
    }
    if let Some(label) = program.route {
        let key = HandlerKey::from(format!("{}/model:{label}", program.owner));
        let route = handlers
            .iter()
            .find(|(bound, _)| *bound == key)
            .map(|(_, entity)| *entity)
            .expect("the golden serves the route");
        world.spawn((rig_ecs::agent::Route(route), Order(order), ChildOf(agent)));
        order += 1;
    }
    world.resource_mut::<rig_ecs::agent::OrderCounter>().0 = order;
    agent
}

/// The world a suspending layer asks, as a thread: answers as the program
/// says; on `Never`, signals `reached` and holds the answer forever.
fn approval_world(
    answer: super::Answer,
    reached: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> super::Asks {
    let (asks, mut asked): (super::Asks, _) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        let mut held = Vec::new();
        while let Some((_, decide)) = asked.blocking_recv() {
            match answer {
                super::Answer::Approve => {
                    let _ = decide.send(rig_core::serve::Decision::Proceed);
                }
                super::Answer::Deny => {
                    let _ = decide.send(rig_core::serve::Decision::deny(super::WORLD_DENY_REASON));
                }
                super::Answer::Never => {
                    reached.store(true, std::sync::atomic::Ordering::SeqCst);
                    held.push(decide);
                }
            }
        }
    });
    asks
}
