//! Concurrent live delivery must reproduce the same effect identities on replay.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::type_complexity,
    clippy::indexing_slicing,
    dead_code
)]
mod run_support;

use std::sync::{Arc, Mutex};

use bevy_ecs::prelude::*;
use futures::channel::oneshot;
use rig_core::{
    completion::{CompletionRequest, CompletionResponse, ModelRef, ProviderCapabilities, Usage},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    message::{AssistantContent, Message, UserContent},
    serve::{OutcomeSink, Serve},
    tool::{ToolOutput, ToolResult},
};
use rig_ecs::{
    agent::{Failed, Grant, MaxTurns, Order, PolicyVersion, RunResult, Settled},
    bus::{Bound, EffectLogResource, Handlers, Replay},
    replay::stamp_run,
    systems::spawn_run,
};
use rig_effect_log::EffectLogRecorder;
use run_support::*;

const MODEL: &str = "t/model:default";
const ADD: &str = "t/tool:add#0";

fn prompt_of(request: &CompletionRequest) -> String {
    request
        .chat_history
        .iter()
        .find_map(|m| match m {
            Message::User { content } => content.iter().find_map(|p| match p {
                UserContent::Text(t) => Some(t.text.clone()),
                _ => None,
            }),
            _ => None,
        })
        .unwrap_or_default()
}

fn has_tool_result(request: &CompletionRequest) -> bool {
    request.chat_history.iter().any(|m| match m {
        Message::User { content } => content
            .iter()
            .any(|p| matches!(p, UserContent::ToolResult(_))),
        _ => false,
    })
}

fn bound_entity(world: &mut World, key: &str) -> Entity {
    world
        .query::<(Entity, &Bound)>()
        .iter(world)
        .find(|(_, b)| b.key == HandlerKey::from(key))
        .map(|(e, _)| e)
        .expect("key bound")
}

fn ended(app: &mut bevy_app::App, run: Entity, what: &str) {
    tick_until(app, what, |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
}

fn ending(world: &World, run: Entity) -> String {
    if let Some(RunResult(r)) = world.get::<RunResult>(run) {
        return format!("Settled({r:?})");
    }
    match world.get::<Failed>(run) {
        Some(Failed(f)) => format!("Failed({f:?})"),
        None => "unfinished".into(),
    }
}

// ---------------------------------------------------------------------------
// P1. Two concurrent runs whose model answers ARRIVE in the opposite order of
// their dispatch: the log records dispatch order and ids only; a by-id replay
// of the same program mints the tool calls' ids in a different interleaving
// and the replayer answers the wrong record.

struct Latched {
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
    gate: Arc<Mutex<Option<oneshot::Receiver<()>>>>,
}

impl Serve for Latched {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(MODEL),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("latched"),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }
    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let EffectKind::Completion { request, .. } = kind else {
            unreachable!()
        };
        self.requests.lock().unwrap().push(request.clone());
        let choice = if has_tool_result(&request) {
            vec![AssistantContent::text("done")]
        } else if prompt_of(&request) == "one" {
            let gate = self.gate.lock().unwrap().take();
            if let Some(gate) = gate {
                let _ = gate.await;
            }
            vec![call("c-one", "add", serde_json::json!({"x": 1, "y": 0}))]
        } else {
            vec![call("c-two", "add", serde_json::json!({"x": 2, "y": 0}))]
        };
        sink.resolve(Ok(Outcome::Completion(CompletionResponse::new(
            choice,
            Usage::new(),
            "latched",
        ))))
        .await;
    }
}

struct Releasing {
    release: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

impl Serve for Releasing {
    type Family = rig_core::effect::family::Tool;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(ADD),
            family: FamilyDescriptor::Tool {
                name: "add".into(),
                description: "adds x and y".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }
    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let EffectKind::ToolCall { args, .. } = kind else {
            unreachable!()
        };
        let parsed: serde_json::Value = serde_json::from_str(&args).unwrap();
        let x = parsed["x"].as_i64().unwrap();
        if x == 2
            && let Some(release) = self.release.lock().unwrap().take()
        {
            let _ = release.send(());
        }
        sink.resolve(Ok(Outcome::ToolResult {
            result: ToolResult::success(ToolOutput::json(serde_json::json!(x * 10))),
        }))
        .await;
    }
}

fn two_run_agent(app: &mut bevy_app::App, model: Entity, tool: Entity) -> Entity {
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert((MaxTurns(2), PolicyVersion("probe/v1".into())));
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    agent
}

#[test]
fn concurrent_runs_replay_the_live_tool_identities() {
    let (tx, rx) = oneshot::channel();
    let mut live = app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    let model = register(
        &mut live,
        MODEL,
        Latched {
            requests: Arc::default(),
            gate: Arc::new(Mutex::new(Some(rx))),
        },
    );
    let tool = register(
        &mut live,
        ADD,
        Releasing {
            release: Arc::new(Mutex::new(Some(tx))),
        },
    );
    let agent = two_run_agent(&mut live, model, tool);
    let one = spawn_run(live.world_mut(), agent, &[], "one", false, None);
    let two = spawn_run(live.world_mut(), agent, &[], "two", false, None);
    stamp_run(live.world_mut(), one, &recorder);
    stamp_run(live.world_mut(), two, &recorder);
    ended(&mut live, one, "run one");
    ended(&mut live, two, "run two");
    let log: rig_effect_log::EffectLog =
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
    assert_eq!(ending(live.world(), one), "Settled(\"done\")");
    assert_eq!(ending(live.world(), two), "Settled(\"done\")");
    // The live interleaving: two's tool call (x=2) got the lower id.
    let tool_ids: Vec<(u64, String)> = log
        .iter()
        .filter_map(|r| match &r.kind {
            EffectKind::ToolCall { args, .. } => Some((r.id.as_u64(), args.clone())),
            _ => None,
        })
        .collect();
    assert!(
        tool_ids[0].1.contains("\"x\":2"),
        "the probe needs two's tool call to be dispatched first: {tool_ids:?}"
    );

    // Replay the same program, unchanged, over by-id replayers.
    let mut replay = app();
    Handlers::with(replay.world_mut(), |h| Replay::default().register(h, &log))
        .unwrap()
        .unwrap();
    let model = bound_entity(replay.world_mut(), MODEL);
    let tool = bound_entity(replay.world_mut(), ADD);
    let agent = two_run_agent(&mut replay, model, tool);
    let one = spawn_run(replay.world_mut(), agent, &[], "one", false, None);
    let two = spawn_run(replay.world_mut(), agent, &[], "two", false, None);
    ended(&mut replay, one, "replayed one");
    ended(&mut replay, two, "replayed two");
    let e1 = ending(replay.world(), one);
    let e2 = ending(replay.world(), two);
    assert_eq!(e1, "Settled(\"done\")");
    assert_eq!(e2, "Settled(\"done\")");
}

fn answer_coincident_models(
    effects: Query<
        (Entity, &rig_ecs::bus::PendingEffect),
        (
            With<rig_ecs::bus::InFlight>,
            Without<rig_ecs::bus::EffectOutcome>,
        ),
    >,
    mut commands: Commands,
) {
    for (entity, effect) in &effects {
        if effect.key != HandlerKey::from(MODEL) {
            continue;
        }
        let EffectKind::Completion { request, .. } = &effect.kind else {
            continue;
        };
        let choice = if has_tool_result(request) {
            vec![AssistantContent::text("done")]
        } else {
            let (id, x) = if prompt_of(request) == "one" {
                ("c-one", 1)
            } else {
                ("c-two", 2)
            };
            vec![call(id, "add", serde_json::json!({"x": x, "y": 0}))]
        };
        commands
            .entity(entity)
            .insert(rig_ecs::bus::WorldOutcome::new(Ok(Outcome::Completion(
                CompletionResponse::new(choice, Usage::new(), "coincident"),
            ))));
    }
}

#[derive(Component)]
struct ArchetypeOnly;

fn change_turn_archetype(
    turns: Query<(Entity, &ChildOf), (With<rig_ecs::agent::Outputs>, Without<ArchetypeOnly>)>,
    runs: Query<&rig_ecs::agent::RunSeq>,
    mut commands: Commands,
) {
    for (turn, parent) in &turns {
        if runs.get(parent.parent()).is_ok_and(|seq| seq.0 == 1) {
            commands.entity(turn).insert(ArchetypeOnly);
        }
    }
}

#[test]
fn coincident_model_answers_ignore_irrelevant_turn_archetypes() {
    use rig_ecs::{
        bus::{BusSet, RigSchedule},
        systems::RigSet,
    };
    let mut live = app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    let model = Handlers::with(live.world_mut(), |handlers| {
        handlers.register_open(
            MODEL,
            FamilyDescriptor::Completion {
                model: ModelRef::new("coincident"),
                capabilities: ProviderCapabilities::default(),
            },
        )
    })
    .unwrap()
    .unwrap();
    let tool = register(
        &mut live,
        ADD,
        Releasing {
            release: Arc::new(Mutex::new(None)),
        },
    );
    live.world_mut().resource_mut::<Schedules>().add_systems(
        RigSchedule,
        answer_coincident_models
            .after(BusSet::Dispatch)
            .before(BusSet::Collect),
    );
    let agent = two_run_agent(&mut live, model, tool);
    let one = spawn_run(live.world_mut(), agent, &[], "one", false, None);
    let two = spawn_run(live.world_mut(), agent, &[], "two", false, None);
    ended(&mut live, one, "live one");
    ended(&mut live, two, "live two");
    let log = recorder.log();
    let mut replay = app();
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::policy_visible().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    replay.world_mut().resource_mut::<Schedules>().add_systems(
        RigSchedule,
        change_turn_archetype
            .after(RigSet::Fold)
            .before(RigSet::Judge),
    );
    let model = bound_entity(replay.world_mut(), MODEL);
    let tool = bound_entity(replay.world_mut(), ADD);
    let agent = two_run_agent(&mut replay, model, tool);
    let one = spawn_run(replay.world_mut(), agent, &[], "one", false, None);
    let two = spawn_run(replay.world_mut(), agent, &[], "two", false, None);
    ended(&mut replay, one, "replay one");
    ended(&mut replay, two, "replay two");
    assert_eq!(ending(replay.world(), one), "Settled(\"done\")");
    assert_eq!(ending(replay.world(), two), "Settled(\"done\")");
}
