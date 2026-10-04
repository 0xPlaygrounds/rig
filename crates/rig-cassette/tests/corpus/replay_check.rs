//! Replay an effect log through a Bevy world by effect id and check the
//! world's own log is the log again. The world-replay targets run it over
//! every committed golden; a producer whose golden is not committed runs it
//! over the log it just recorded (`rig_test_support::goldens`).

#![allow(clippy::expect_used, clippy::panic)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use bevy_app::App;
use bevy_ecs::schedule::LogLevel;
use rig_cassette::{
    ecs::{EffectLogResource, Replay, ReplayPlugin, identity::check_replayable},
    effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer},
};
use rig_core::{
    effect::{EffectKind, HandlerDescriptor},
    serve::{Dispatch, ErasedHandler, Reply, Serve, ServingPolicy},
};
use rig_ecs::{
    RigPlugin,
    agent::RunOf,
    bus::{BusPlugin, EffectOutcome, Handlers, Issued, Scope, Streamed},
    checkpoint::{Checkpoint, RestoreMode, load_world},
};

const GUARD: Duration = Duration::from_secs(30);

/// A world log's saved program scenes, by scope.
pub type Programs = BTreeMap<String, (ServingPolicy, Checkpoint)>;

/// A world under the log's own serving policy (its header's `bus`, or the
/// default when it was recorded on a host's bus), with the intake bound
/// lifted so one tick takes every record it can.
fn agent_world(log: &EffectLog) -> App {
    let mut app = App::new();
    let policy = ServingPolicy {
        command_capacity: 10_000,
        ..log.header.bus.unwrap_or_default()
    };
    app.add_plugins(BusPlugin::with_policy(policy).ambiguity_detection(LogLevel::Error));
    app.add_plugins(ReplayPlugin);
    app.finish();
    app.cleanup();
    app
}

/// Replay an agent log through a fresh world by id and compare it record
/// by record: the same ids, keys, requests, outcomes, parents, scopes and
/// kept events. Returns the number of records replayed.
pub fn replay_through_a_world(name: &str, log: &EffectLog) -> usize {
    let mut app = agent_world(log);
    Replay::default()
        .register(app.world_mut(), log)
        .unwrap_or_else(|report| panic!("{name}: the golden registers: {report}"));
    // The golden's recorder kept events, or it did not: the world's does
    // the same, so the log it writes is comparable field for field.
    let recorder = if log.records.iter().any(|record| record.events.is_some()) {
        EffectLogRecorder::keeping_stream_events()
    } else {
        EffectLogRecorder::new()
    };
    EffectLogResource::install(app.world_mut(), recorder);
    let entities = Replay::load(app.world_mut(), log);
    assert_eq!(
        entities.len(),
        log.records.len(),
        "{name}: one entity per record"
    );

    let start = Instant::now();
    loop {
        app.update();
        let world = app.world();
        if entities
            .iter()
            .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
        {
            break;
        }
        assert!(
            start.elapsed() < GUARD,
            "{name}: not replayed within {GUARD:?}"
        );
        std::thread::yield_now();
    }

    let world = app.world();
    let mut records: Vec<_> = log.records.iter().collect();
    records.sort_by_key(|record| record.id);
    for (entity, record) in entities.iter().zip(&records) {
        assert_eq!(
            world.get::<Issued>(*entity).expect("issued").0,
            record.id,
            "{name}: the recorded id"
        );
        let outcome = world.get::<EffectOutcome>(*entity).expect("answered");
        assert_eq!(
            serde_json::to_value(&outcome.0).expect("serde"),
            serde_json::to_value(&record.outcome).expect("serde"),
            "{name}: record {} replays its outcome",
            record.id
        );
        if let Some(events) = &record.events {
            let streamed = world
                .get::<Streamed>(*entity)
                .unwrap_or_else(|| panic!("{name}: record {} streamed", record.id));
            assert_eq!(
                serde_json::to_value(&streamed.events).expect("serde"),
                serde_json::to_value(events).expect("serde"),
                "{name}: record {} replays its events in order",
                record.id
            );
        }
    }
    // The world's log is in begin order, as rig-agent's bus's is; under serial
    // serving that is not id order in either runtime, so both sides are
    // compared by id.
    let mut replayed = world.resource::<EffectLogResource>().log().records;
    replayed.sort_by_key(|record| record.id);
    assert_eq!(replayed.len(), records.len(), "{name}: the world's log");
    for (mine, theirs) in replayed.iter().zip(&records) {
        assert_eq!(mine.id, theirs.id, "{name}");
        assert_eq!(mine.key, theirs.key, "{name}");
        assert_eq!(mine.parent, theirs.parent, "{name}: causality survives");
        assert_eq!(mine.scope, theirs.scope, "{name}: the scope survives");
        assert_eq!(
            serde_json::to_value(&mine.kind).expect("serde"),
            serde_json::to_value(&theirs.kind).expect("serde"),
            "{name}: the request is the record's"
        );
        assert_eq!(
            serde_json::to_value(&mine.outcome).expect("serde"),
            serde_json::to_value(&theirs.outcome).expect("serde"),
            "{name}"
        );
        assert_eq!(
            serde_json::to_value(&mine.events).expect("serde"),
            serde_json::to_value(&theirs.events).expect("serde"),
            "{name}: kept events are kept again"
        );
    }
    records.len()
}

struct ToolTripwire {
    descriptor: HandlerDescriptor,
    calls: Arc<AtomicUsize>,
}

impl Serve for ToolTripwire {
    type Family = rig_core::effect::family::Dynamic;
    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }
    // Never answering, rather than panicking, leaves one way for a replay
    // to end: the loop's call count. A panic on the pool would race it and
    // unwind through the collector whenever it won.
    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        self.calls.fetch_add(1, Ordering::SeqCst);
        std::future::pending().await
    }
}

struct ConfigurationHandler {
    descriptor: HandlerDescriptor,
    replayer: EffectLogReplayer,
}

impl Serve for ConfigurationHandler {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        self.replayer.serve(kind, dispatch).await
    }
}

/// Restore each saved program scene of a world log into a fresh world and
/// check the run it configures is replay-compatible with the log, and a
/// stale policy is refused. Returns the one serving policy they share.
pub fn check_programs(name: &str, log: &EffectLog, programs: &Programs) -> ServingPolicy {
    assert_eq!(
        programs.keys().collect::<Vec<_>>(),
        log.header.programs.keys().collect::<Vec<_>>(),
        "{name}: exact scope coverage"
    );
    let policy = programs.values().next().expect("at least one program").0;
    for (scope, (serving, scene)) in programs {
        assert_eq!(*serving, policy, "{name}: one serving policy per world log");
        let mut app = App::new();
        app.add_plugins(RigPlugin::with_policy(*serving));
        app.finish();
        app.cleanup();
        let handlers = scene
            .requirements()
            .expect("scene requirements")
            .into_iter()
            .map(|descriptor| {
                let key = descriptor.key.clone();
                assert!(
                    log.header.handlers.contains(&descriptor),
                    "{name}/{scope}: the log declares the saved handler"
                );
                let replayer = EffectLogReplayer::for_key_by_id(log, &key)
                    .unwrap_or_else(|error| panic!("{name}/{scope}: {error}"));
                // This app checks saved declarations without executing policies.
                // The separate bus app replays exchanges without these layer names.
                (
                    key,
                    ErasedHandler::new(ConfigurationHandler {
                        descriptor,
                        replayer,
                    }),
                )
            })
            .collect::<Vec<_>>();
        let loaded = load_world(scene, app.world_mut(), RestoreMode::Strict, handlers)
            .unwrap_or_else(|error| panic!("{name}/{scope}: restore configuration: {error}"));
        let runs: Vec<_> = loaded
            .with::<RunOf>(app.world())
            .into_iter()
            .filter(|run| {
                app.world()
                    .get::<Scope>(*run)
                    .is_some_and(|value| &value.0 == scope)
            })
            .collect();
        assert_eq!(runs.len(), 1, "{name}/{scope}: exactly one configured run");
        let run = *runs.first().expect("one configured run");
        check_replayable(app.world_mut(), run, log)
            .unwrap_or_else(|error| panic!("{name}/{scope}: compatibility: {error}"));
        let mut stale = log.clone();
        stale.header.programs.get_mut(scope).expect("scope").policy ^= 1;
        assert!(
            check_replayable(app.world_mut(), run, &stale).is_err(),
            "{name}/{scope}: stale policy must refuse"
        );
    }
    policy
}

/// Replay a world log by id under `policy` and require every recorded
/// field to survive. A long task's tools are tripwires that must never
/// run; `inject_live_tool` puts one in place of the recorded handler.
pub fn replay_world_log(
    name: &str,
    log: &EffectLog,
    policy: ServingPolicy,
    inject_live_tool: bool,
) {
    let mut app = App::new();
    app.add_plugins(BusPlugin::with_policy(ServingPolicy {
        command_capacity: 10_000,
        ..policy
    }));
    app.add_plugins(ReplayPlugin);
    app.finish();
    app.cleanup();
    let tool_calls = Arc::new(AtomicUsize::new(0));
    if name.contains("long_task") {
        Handlers::with(app.world_mut(), |handlers| {
            for descriptor in &log.header.handlers {
                if descriptor.family.family() == rig_core::effect::EffectFamily::Tool {
                    handlers
                        .register_erased(
                            descriptor.key.clone(),
                            ErasedHandler::new(ToolTripwire {
                                descriptor: descriptor.clone(),
                                calls: tool_calls.clone(),
                            }),
                        )
                        .expect("live tripwire registration");
                }
            }
        })
        .expect("bus handlers");
    }
    Replay::default()
        .register(app.world_mut(), log)
        .unwrap_or_else(|error| panic!("{name}: register replay: {error}"));
    if inject_live_tool {
        let descriptor = log
            .header
            .handlers
            .iter()
            .find(|descriptor| descriptor.family.family() == rig_core::effect::EffectFamily::Tool)
            .expect("task tool descriptor")
            .clone();
        Handlers::with(app.world_mut(), |handlers| {
            handlers
                .register_erased(
                    descriptor.key.clone(),
                    ErasedHandler::new(ToolTripwire {
                        descriptor,
                        calls: tool_calls.clone(),
                    }),
                )
                .expect("inject live task tool");
        })
        .expect("bus handlers");
    }
    let recorder = if log.records.iter().any(|record| record.events.is_some()) {
        EffectLogRecorder::keeping_stream_events()
    } else {
        EffectLogRecorder::new()
    };
    EffectLogResource::install(app.world_mut(), recorder);
    let entities = Replay::load(app.world_mut(), log);
    assert_eq!(
        entities.len(),
        log.records.len(),
        "{name}: all records loaded"
    );
    let started = Instant::now();
    loop {
        app.update();
        assert_eq!(
            tool_calls.load(Ordering::SeqCst),
            0,
            "effect replay must never execute a live task tool"
        );
        assert!(
            app.world()
                .get_resource::<rig_cassette::ecs::ReplayFailure>()
                .is_none(),
            "{name}: delivery replay failed"
        );
        if entities
            .iter()
            .all(|entity| app.world().get::<EffectOutcome>(*entity).is_some())
        {
            break;
        }
        assert!(
            started.elapsed() < GUARD,
            "{name}: replay exceeded {GUARD:?}"
        );
        std::thread::yield_now();
    }
    let mut actual = app.world().resource::<EffectLogResource>().log().records;
    let mut expected = log.records.clone();
    actual.sort_by_key(|record| record.id);
    expected.sort_by_key(|record| record.id);
    assert_eq!(
        serde_json::to_value(actual).expect("replayed records"),
        serde_json::to_value(expected).expect("world records"),
        "{name}: every recorded field survives by-id replay"
    );
}
