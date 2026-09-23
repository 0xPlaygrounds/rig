//! Validate native program configurations and replay world goldens by effect id.

#![allow(clippy::expect_used, clippy::panic)]

#[path = "corpus/fixtures.rs"]
mod fixtures;

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use bevy_app::App;
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
    bus::{BusPlugin, EffectOutcome, Handlers, Scope},
    checkpoint::{Checkpoint, RestoreMode, load_world},
};

const EXPECTED_GOLDENS: usize = 1217;
const GUARD: Duration = Duration::from_secs(30);
type Programs = BTreeMap<String, (ServingPolicy, Checkpoint)>;

struct ToolTripwire {
    descriptor: HandlerDescriptor,
    calls: Arc<AtomicUsize>,
}

impl Serve for ToolTripwire {
    type Family = rig_core::effect::family::Dynamic;
    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }
    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        self.calls.fetch_add(1, Ordering::SeqCst);
        panic!("effect replay invoked a live task tool")
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

fn check_programs(name: &str, log: &EffectLog, programs: &Programs) -> ServingPolicy {
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

fn replay(name: &str, log: &EffectLog, policy: ServingPolicy, inject_live_tool: bool) {
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

#[test]
#[should_panic(expected = "live task tool")]
fn a_live_tool_replacing_the_recorded_handler_is_detected() {
    let name = "openai_chat_long_task_inventory_restore";
    let text = std::fs::read_to_string(
        fixtures::effects_dir()
            .join("world")
            .join(format!("{name}.effects.json")),
    )
    .expect("task golden");
    let log: EffectLog = serde_json::from_str(&text).expect("task log");
    replay(name, &log, ServingPolicy::default(), true);
}

#[test]
fn every_world_golden_checks_its_programs_and_replays_by_id() {
    let directory = fixtures::effects_dir().join("world");
    let mut names: Vec<_> = std::fs::read_dir(&directory)
        .expect("world corpus")
        .map(|entry| {
            entry
                .expect("world fixture")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter_map(|name| name.strip_suffix(".effects.json").map(str::to_owned))
        .collect();
    names.sort();
    assert_eq!(
        names.len(),
        EXPECTED_GOLDENS,
        "update the count with the world corpus"
    );
    for name in &names {
        let text = std::fs::read_to_string(directory.join(format!("{name}.effects.json")))
            .expect("world golden");
        let log: EffectLog = serde_json::from_str(&text).expect("world log");
        let text = std::fs::read_to_string(directory.join(format!("{name}.programs.json")))
            .expect("world program scenes");
        let programs: Programs = serde_json::from_str(&text).expect("world program scenes decode");
        let policy = check_programs(name, &log, &programs);
        replay(name, &log, policy, false);
    }
    eprintln!("{} world goldens checked and replayed by id", names.len());
}
