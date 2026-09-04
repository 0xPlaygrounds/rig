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
        AdditionalParams, Context, DefaultMaxTurns, DocumentId, DocumentText, Failed, Failure,
        Grant, InvalidCalls, MaxTokens, MaxTurns, MessageParts, Order, Output, OutputKind, Owner,
        Preamble, RunResult, Settled, Temperature, ToolChoiceSpec, ToolPolicy,
        Unhandled as WorldUnhandled, UsesModel,
    },
    bus::{BusPlugin, EffectLogResource, Handlers, IdCounter},
    replay::stamp_header,
    systems::{AgentPlugin, spawn_run},
};
use rig_effect_log::{EffectLogRecorder, EffectLogReplayer};

use super::{
    Ending, Output as CorpusOutput, Program, Unhandled, assert_same_records, golden, golden_answer,
    keeps_events, run_spec,
};

const GUARD: Duration = Duration::from_secs(30);

/// Why a program is not this interpreter's yet: the set or entity it waits
/// for, by stage.
pub fn unsupported(program: &Program) -> Option<&'static str> {
    if !program.hooks.is_empty() {
        return Some("hooks as systems and observers (stage 4)");
    }
    if program.conversation.is_some() {
        return Some("conversation memory (stage 5)");
    }
    if program.route.is_some() || program.late_route.is_some() {
        return Some("model routing (stage 4)");
    }
    if program.dynamic_context.is_some() || program.retrieved_tools.is_some() {
        return Some("retrieval (stage 4)");
    }
    if !program.layers.is_empty() {
        return Some("layers on the program's handlers (stage 4)");
    }
    if program.second_prompt.is_some() {
        return Some("two runs on one agent (stage 5)");
    }
    if matches!(program.ending, Ending::MemoryError) {
        return Some("a memory failure (stage 5)");
    }
    let log = golden(program.fixture);
    if log.records.iter().any(|record| {
        matches!(
            record.kind.family(),
            EffectFamily::Memory | EffectFamily::Retrieve
        )
    }) {
        return Some("memory or retrieval dispatches (stage 5)");
    }
    None
}

/// The world interpreter's cell: replays `program` through a world, or
/// reports why it cannot yet.
pub fn world_agent_reproduces(program: &Program) {
    if let Some(why) = unsupported(program) {
        eprintln!("world_agent: {} — unsupported: {why}", program.fixture);
        return;
    }
    let log = golden(program.fixture);
    EffectLogReplayer::check_header(&log).expect("a current format");
    // A host-bus golden names no policy: the replay's host runs the
    // producer's where the program names it, as `Replay::open` does.
    let mut policy = log.header.bus.unwrap_or_default();
    if program.host_serial {
        assert!(log.header.bus.is_none(), "a host-bus program");
        policy.serial_per_handler = true;
    }
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
    // interpreters register them: the world mints its own ids.
    let mut handler_entities: Vec<(HandlerKey, Entity)> = Vec::new();
    Handlers::with(world, |handlers| {
        for replayer in EffectLogReplayer::for_log(&log).expect("the golden's replayers") {
            let key = replayer.key().clone();
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
    let run = spawn_run(
        world,
        agent,
        &history,
        program.prompt,
        program.streamed,
        program.max_turns,
    );
    if let Some(concurrency) = program.tool_concurrency {
        world.entity_mut(run).insert(ToolPolicy { concurrency });
    }

    let start = Instant::now();
    loop {
        app.update();
        if program.cancel_when_reached && super::world_nesting::reached(app.world_mut()) {
            // The producer dropped the run once the never-answering handler
            // was reached: the run entity goes, and the whole tree with it
            // — the tool call and its child are cancelled, a queued child
            // never begins.
            app.world_mut().despawn(run);
            app.update();
            app.update();
            let replayed = app.world().resource::<EffectLogResource>().log();
            assert_same_records(&replayed, &log, "world agent");
            return;
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
        ((Some(result), None), Ending::Answer) => {
            assert_eq!(
                result.0,
                program
                    .expected_output
                    .map_or_else(|| golden_answer(&log), str::to_owned),
                "{}: the answer",
                program.fixture
            );
        }
        ((None, Some(Failed(Failure::MaxTurns { .. }))), Ending::MaxTurns)
        | ((None, Some(Failed(Failure::UnknownToolCall { .. }))), Ending::UnknownToolCall) => {}
        ((None, Some(Failed(Failure::Provider(report)))), Ending::ProviderError)
            if report.kind == ErrorKind::ProviderResponse => {}
        ((None, Some(Failed(Failure::Provider(report)))), Ending::Failed(kind))
        | ((None, Some(Failed(Failure::Tool(report)))), Ending::Failed(kind))
            if report.kind == kind => {}
        (other, ending) => panic!(
            "{}: the run ends in {ending:?}, the world says {other:?}",
            program.fixture
        ),
    }

    // The world's log is in begin order, as rig-bus's; the oracle asserts
    // it is dispatch order. The records carry the run's `Scope`, which the
    // goldens do not have and `as_data` does not compare.
    let replayed = world.resource::<EffectLogResource>().log();
    assert_same_records(&replayed, &log, "world agent");
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
        world.spawn((Grant(tool), Order(order), ChildOf(agent)));
        order += 1;
    }
    world.resource_mut::<rig_ecs::agent::OrderCounter>().0 = order;
    agent
}
