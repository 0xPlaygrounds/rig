//! Isolated assembly, materialisation and whole-world scene-save measurements.
//!
//! Run with `cargo bench -p rig-ecs --features app,testing --bench run_assembly_cost`.
//! Sizes: RIG_ASSEMBLY_HISTORY_SIZES=0,32,128,512;
//! RIG_ASSEMBLY_UNRELATED_RUNS=0,8,32; RIG_ASSEMBLY_REPEATS=3;
//! RIG_ASSEMBLY_TOOL_TURNS=8; RIG_ASSEMBLY_RESULT_BYTES=8192;
//! RIG_ASSEMBLY_SCENE_SAVES=3. JSON: RIG_ASSEMBLY_MEASURE_OUT=/absolute/path.json.
//! Each scenario/repetition gets a fresh app, handlers, resources and caches.
//! Unrelated runs finish before measurement and retain history/effects in the world.
//! Timers bracket schedule sets, including scheduling and their deferred flushes,
//! not individual function bodies. Setup, handler dispatch/poll/wait, instrumentation,
//! scene destruction and JSON/file output are outside the measured regions.
//! Wall-clock scheduling interference is still possible; these are not CPU timings.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout
)]

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use bevy_app::App;
use bevy_ecs::{prelude::*, schedule::LogLevel};
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, Message},
    serve::{Dispatch, Reply, Serve},
    tool::{ToolOutput, ToolResult},
};
use rig_ecs::{
    RigPlugin,
    agent::{
        Failed, Grant, MaxTurns, MessageParts, Order, Owner, RunResult, Settled, UsesModel,
        Utterance,
        content::cache::AssemblyStats,
        scene::{SceneKind, save_world},
    },
    bus::{Bus, Handlers, PendingEffect, RigSchedule},
    systems::{Materialised, RigSet, RunCommands, RunConfig},
    testing::{Scripted, tick_until},
};

const MODEL: &str = "bench/model:default";
const TOOL: &str = "bench/tool:read#0";
const ANSWER: &str = "benchmark complete";
const HISTORY_BYTES: usize = 128;

struct Reader {
    bytes: usize,
    calls: Arc<AtomicUsize>,
}

impl Serve for Reader {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(TOOL),
            family: FamilyDescriptor::Tool {
                name: "read".to_owned(),
                description: "reads a chunk".to_owned(),
                parameters: serde_json::json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let EffectKind::ToolCall { .. } = kind else {
            return Reply::Outcome(Err(ErrorReport::new(ErrorKind::Request, "a tool call")));
        };
        self.calls.fetch_add(1, Ordering::Relaxed);
        Reply::Outcome(Ok(Outcome::ToolResult {
            result: ToolResult::success(ToolOutput::text("x".repeat(self.bytes))),
        }))
    }
}

#[derive(Default, serde::Serialize)]
struct Pass {
    pass: usize,
    chat_histories: Vec<usize>,
    assemblies: u64,
    renders: u64,
    hits: u64,
    evictions: u64,
    materialised_turns: usize,
    written_utterances: usize,
    assembly_ms: f64,
    materialisation_ms: f64,
}

#[derive(Resource)]
struct Measurements {
    run: Entity,
    started: Option<Instant>,
    utterances_before_materialisation: usize,
    previous: AssemblyStats,
    current: Pass,
    rows: Vec<Pass>,
    passes: usize,
    assembly_ms: f64,
    materialisation_ms: f64,
}

fn start_assembly(mut measured: ResMut<Measurements>) {
    measured.passes += 1;
    measured.current = Pass {
        pass: measured.passes,
        ..Default::default()
    };
    measured.started = Some(Instant::now());
}

fn finish_assembly(
    mut measured: ResMut<Measurements>,
    stats: Res<AssemblyStats>,
    effects: Query<(&PendingEffect, &ChildOf), Added<PendingEffect>>,
    parents: Query<&ChildOf>,
) {
    let elapsed = measured
        .started
        .take()
        .expect("assembly started")
        .elapsed()
        .as_secs_f64()
        * 1e3;
    measured.assembly_ms += elapsed;
    measured.current.assembly_ms = elapsed;
    measured.current.assemblies = stats.assemblies - measured.previous.assemblies;
    measured.current.renders = stats.renders - measured.previous.renders;
    measured.current.hits = stats.hits - measured.previous.hits;
    measured.current.evictions = stats.evictions - measured.previous.evictions;
    measured.previous = *stats;
    for (effect, parent) in &effects {
        if parents
            .get(parent.parent())
            .is_ok_and(|parent| parent.parent() == measured.run)
            && let EffectKind::Completion { request, .. } = &effect.kind
        {
            measured
                .current
                .chat_histories
                .push(request.chat_history.len());
        }
    }
}

fn start_materialisation(
    mut measured: ResMut<Measurements>,
    children: Query<&Children>,
    utterances: Query<(), With<Utterance>>,
) {
    measured.utterances_before_materialisation = children
        .get(measured.run)
        .map_or(0, |owned| utterances.iter_many(owned.iter()).count());
    measured.started = Some(Instant::now());
}

fn finish_materialisation(
    mut measured: ResMut<Measurements>,
    turns: Query<&ChildOf, Added<Materialised>>,
    children: Query<&Children>,
    utterances: Query<(), With<Utterance>>,
) {
    let elapsed = measured
        .started
        .take()
        .expect("materialisation started")
        .elapsed()
        .as_secs_f64()
        * 1e3;
    measured.materialisation_ms += elapsed;
    measured.current.materialisation_ms = elapsed;
    measured.current.materialised_turns =
        turns.iter().filter(|p| p.parent() == measured.run).count();
    let utterances_now = children
        .get(measured.run)
        .map_or(0, |owned| utterances.iter_many(owned.iter()).count());
    measured.current.written_utterances =
        utterances_now - measured.utterances_before_materialisation;
    // Idle polling pass counts/totals remain visible, without unbounded row storage.
    if measured.current.assemblies != 0
        || measured.current.materialised_turns != 0
        || measured.current.written_utterances != 0
    {
        let row = std::mem::take(&mut measured.current);
        measured.rows.push(row);
    }
}

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(RigPlugin {
        bus: Bus::default().ambiguity_detection(LogLevel::Error),
    });
    app.finish();
    app.cleanup();
    app
}

fn register(app: &mut App, key: &str, handler: impl Serve + 'static) -> Entity {
    Handlers::with(app.world_mut(), |handlers| handlers.register(key, handler))
        .expect("installed bus")
        .expect("unique benchmark handler")
}

fn ended(world: &World, run: Entity) -> bool {
    world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
}

fn assert_answer(world: &World, run: Entity) {
    assert!(
        world.get::<Settled>(run).is_some(),
        "{:?}",
        world.get::<Failed>(run)
    );
    assert_eq!(
        world.get::<RunResult>(run).expect("settled answer").0,
        ANSWER
    );
}

fn scenario(
    history_size: usize,
    unrelated: usize,
    repeat: usize,
    options: &Options,
) -> serde_json::Value {
    let mut app = app();
    let history: Vec<MessageParts> = (0..history_size)
        .map(|n| {
            let text = format!("history {n}: {}", "h".repeat(HISTORY_BYTES));
            let message = if n % 2 == 0 {
                Message::user(text)
            } else {
                Message::assistant(text)
            };
            MessageParts::from_message(&message).expect("user or assistant history")
        })
        .collect();
    let (idle_model, _) = Scripted::new(
        "bench/model:unrelated",
        vec![vec![AssistantContent::text(ANSWER)]; unrelated],
    );
    let idle_model = register(&mut app, "bench/model:unrelated", idle_model);
    let idle_agent = app
        .world_mut()
        .spawn((
            Owner("unrelated".into()),
            UsesModel(idle_model),
            MaxTurns(1),
        ))
        .id();
    let idle_runs: Vec<Entity> = (0..unrelated)
        .map(|_| {
            app.world_mut().spawn_run(
                idle_agent,
                "background",
                RunConfig {
                    history: &history,
                    ..Default::default()
                },
            )
        })
        .collect();
    if !idle_runs.is_empty() {
        tick_until(&mut app, "unrelated runs settle", |world| {
            idle_runs.iter().all(|run| ended(world, *run))
        });
        for run in &idle_runs {
            assert_answer(app.world(), *run);
        }
    }

    let mut script: Vec<Vec<AssistantContent>> = (0..options.tool_turns)
        .map(|n| {
            vec![AssistantContent::tool_call(
                format!("c{n}"),
                "read",
                serde_json::json!({"n": n}),
            )]
        })
        .collect();
    script.push(vec![AssistantContent::text(ANSWER)]);
    let (model, requests) = Scripted::new(MODEL, script);
    let model = register(&mut app, MODEL, model);
    let calls = Arc::new(AtomicUsize::new(0));
    let tool = register(
        &mut app,
        TOOL,
        Reader {
            bytes: options.result_bytes,
            calls: Arc::clone(&calls),
        },
    );
    let agent = app
        .world_mut()
        .spawn((
            Owner("bench".into()),
            UsesModel(model),
            MaxTurns(options.tool_turns + 2),
        ))
        .id();
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    let run = app.world_mut().spawn_run(
        agent,
        "read everything",
        RunConfig {
            history: &history,
            ..Default::default()
        },
    );
    let before = *app.world().resource::<AssemblyStats>();
    app.insert_resource(Measurements {
        run,
        started: None,
        previous: before,
        current: Pass::default(),
        rows: Vec::new(),
        passes: 0,
        assembly_ms: 0.0,
        materialisation_ms: 0.0,
        utterances_before_materialisation: 0,
    });
    app.world_mut().resource_mut::<Schedules>().add_systems(
        RigSchedule,
        (
            start_assembly
                .after(RigSet::Select)
                .before(RigSet::Assemble),
            finish_assembly.in_set(RigSet::Patch),
            start_materialisation
                .after(RigSet::Judge)
                .before(RigSet::Materialise),
            finish_materialisation.in_set(RigSet::Checkpoint),
        ),
    );
    tick_until(&mut app, "measured run settles", |world| ended(world, run));
    assert_answer(app.world(), run);
    assert_eq!(calls.load(Ordering::Relaxed), options.tool_turns);
    assert_eq!(
        requests.lock().expect("captured requests").len(),
        options.tool_turns + 1
    );
    let measured = app
        .world_mut()
        .remove_resource::<Measurements>()
        .expect("measurements");
    assert_eq!(
        measured.rows.iter().map(|p| p.assemblies).sum::<u64>(),
        (options.tool_turns + 1) as u64
    );
    assert_eq!(
        measured
            .rows
            .iter()
            .map(|p| p.materialised_turns)
            .sum::<usize>(),
        options.tool_turns + 1
    );
    let after = *app.world().resource::<AssemblyStats>();
    let mut saves = Vec::with_capacity(options.scene_saves);
    for _ in 0..options.scene_saves {
        let started = Instant::now();
        let scene = save_world(app.world_mut()).expect("valid settled scene");
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        assert_eq!(
            scene
                .graph
                .entities
                .iter()
                .filter(|entity| entity.kind == SceneKind::Run)
                .count(),
            unrelated + 1
        );
        saves.push(serde_json::json!({
            "scene_save_ms": elapsed,
            "graph_entities": scene.graph.entities.len(),
            "effects": scene.effects.effects.len(),
        }));
        std::hint::black_box(scene);
    }
    serde_json::json!({
        "initial_history": history_size, "unrelated_settled_runs": unrelated, "repeat": repeat,
        "schedule_passes": measured.passes,
        "all_passes_cumulative": {
            "assembly_ms": measured.assembly_ms, "materialisation_ms": measured.materialisation_ms,
            "assemblies": after.assemblies - before.assemblies,
            "renders": after.renders - before.renders, "hits": after.hits - before.hits,
            "evictions": after.evictions - before.evictions,
        },
        "productive_passes_cumulative": {
            "assembly_ms": measured.rows.iter().map(|p| p.assembly_ms).sum::<f64>(),
            "materialisation_ms": measured.rows.iter().map(|p| p.materialisation_ms).sum::<f64>(),
        },
        "productive_passes": measured.rows, "scene_saves": saves,
    })
}

struct Options {
    histories: Vec<usize>,
    unrelated: Vec<usize>,
    repeats: usize,
    tool_turns: usize,
    result_bytes: usize,
    scene_saves: usize,
}

fn sizes(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse()
                .expect("comma-separated nonnegative integers")
        })
        .collect()
}

fn positive(name: &str, default: usize) -> usize {
    let value =
        std::env::var(name).map_or(default, |value| value.parse().expect("positive integer"));
    assert!(value > 0, "{name} must be positive");
    value
}

fn main() {
    let options = Options {
        histories: sizes("RIG_ASSEMBLY_HISTORY_SIZES", "0,32,128,512"),
        unrelated: sizes("RIG_ASSEMBLY_UNRELATED_RUNS", "0,8,32"),
        repeats: positive("RIG_ASSEMBLY_REPEATS", 3),
        tool_turns: positive("RIG_ASSEMBLY_TOOL_TURNS", 8),
        result_bytes: positive("RIG_ASSEMBLY_RESULT_BYTES", 8192),
        scene_saves: positive("RIG_ASSEMBLY_SCENE_SAVES", 3),
    };
    let mut scenarios = Vec::new();
    for &history in &options.histories {
        for &unrelated in &options.unrelated {
            for repeat in 0..options.repeats {
                let result = scenario(history, unrelated, repeat, &options);
                println!("{result}");
                scenarios.push(result);
            }
        }
    }
    if let Ok(path) = std::env::var("RIG_ASSEMBLY_MEASURE_OUT") {
        let report = serde_json::json!({
            "history_sizes": options.histories, "unrelated_run_counts": options.unrelated,
            "repeats": options.repeats, "tool_turns": options.tool_turns,
            "result_bytes": options.result_bytes, "history_payload_bytes": HISTORY_BYTES,
            "scene_saves_per_scenario": options.scene_saves,
            "assembly_region": "after Select to Patch (Assemble and deferred publication)",
            "materialisation_region": "after Judge to Checkpoint (Materialise and deferred publication)",
            "scene_region": "save_world on settled world; excludes destruction and JSON encoding",
            "counter_limits": "renders/hits count message operations, not bytes, lookups, comparisons or allocations; productive rows are per-pass deltas; all-pass totals include idle passes; no asymptotic timing claim",
            "scenarios": scenarios,
        });
        std::fs::write(
            path,
            serde_json::to_string_pretty(&report).expect("JSON report"),
        )
        .expect("write report");
    }
}
