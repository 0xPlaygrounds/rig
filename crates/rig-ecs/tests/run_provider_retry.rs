//! A completion lost to a retryable provider failure is re-issued inside the
//! run (CONTRACT §5): the same request over the same history, no tool
//! re-run, every attempt its own record, time left to the host.
//!
//! | claim | test |
//! |---|---|
//! | a 503 after a tool batch re-issues the completion; the tool ran once; the answer settles; the log holds both attempts and replays them | `a_retryable_failure_after_tool_work_is_reissued_and_the_tool_runs_once` |
//! | the retried attempt is not a model call against `MaxTurns`; the witness gets the fact | same |
//! | a spent budget ends the run `Failed(Provider)` with the last report | `a_spent_budget_ends_the_run_with_the_last_report` |
//! | a non-retryable report after tool work ends the run on the first failure | `a_non_retryable_failure_after_tool_work_ends_the_run_at_once` |
//! | `ProviderRetries(0)` is the old behaviour | `a_zero_budget_never_retries` |
//! | a backoff is a host hold on the re-issued effect; a cancel during it ends the run `Cancelled` with no further request | `a_host_hold_is_where_a_backoff_goes_and_a_cancel_during_it_ends_the_run` |
//! | a checkpoint saved during that hold resumes into the retry, not a fresh prompt | `a_checkpoint_saved_during_the_hold_resumes_into_the_retry` |
//! | a stream cut before its terminal record is a transport fault: retryable, re-issued, answered | `a_truncated_stream_is_reissued` |

use crate::run_support;

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

use bevy_ecs::prelude::*;
use rig_cassette::ecs::EffectLogResource;
use rig_cassette::ecs::Replay;
use rig_cassette::ecs::identity::stamp_run;
use rig_cassette::effect_log::{EffectLog, EffectLogRecorder};
use rig_core::{
    completion::{CompletionRequest, CompletionResponse, ModelRef, ProviderCapabilities, Usage},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    observe::{Action, ObservationLog},
    serve::{Dispatch, Reply, Serve},
    streaming::{BlockId, BlockKind, Delta, StreamEvent},
};
use rig_ecs::{
    agent::{
        Cancelled, Cursor, Failed, Failure, Grant, MaxTurns, ProviderRetried, ProviderRetries,
        RunResult, Settled,
    },
    bus::{Bound, BusSet, Held, PendingEffect, RigSchedule, Witnessing},
    checkpoint::{Checkpoint, RestoreMode, load_world, save_world},
    systems::RunCommands,
};
use run_support::*;

const MODEL: &str = "t/model:default";
const ADD: &str = "t/tool:add#0";

/// A model that answers a script of outcomes, one per request, in order.
struct Flaky {
    script: Mutex<VecDeque<Result<Vec<AssistantContent>, ErrorReport>>>,
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl Serve for Flaky {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(MODEL),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new(MODEL),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let EffectKind::Completion { request, .. } = kind else {
            return Reply::Outcome(Err(ErrorReport::new(ErrorKind::Request, "a completion")));
        };
        self.requests.lock().unwrap().push(request);
        let next = self.script.lock().unwrap().pop_front();
        Reply::Outcome(match next {
            Some(Ok(choice)) => Ok(Outcome::Completion(CompletionResponse::new(
                choice,
                Usage::default(),
                "flaky",
                serde_json::json!({}),
            ))),
            Some(Err(report)) => Err(report),
            None => Err(ErrorReport::new(ErrorKind::Provider, "the script ran out")),
        })
    }
}

fn unavailable(message: &str) -> ErrorReport {
    ErrorReport::new(ErrorKind::ProviderResponse, message)
        .with_http_status(503)
        .with_retryable(true)
}

fn add_call() -> Vec<AssistantContent> {
    vec![call("c1", "add", serde_json::json!({"x": 1, "y": 2}))]
}

fn done() -> Vec<AssistantContent> {
    vec![AssistantContent::text("done")]
}

type Script = Vec<Result<Vec<AssistantContent>, ErrorReport>>;
type Requests = Arc<Mutex<Vec<CompletionRequest>>>;

/// An app recording its effects, a flaky model and the adder granted to
/// one agent, with a witness log.
fn tooling(
    script: Script,
) -> (
    bevy_app::App,
    Entity,
    Requests,
    EffectLogRecorder,
    Arc<ObservationLog>,
) {
    let mut app = run_support::app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    let witness = Arc::new(ObservationLog::default());
    Witnessing::install(app.world_mut(), witness.clone());
    let requests: Requests = Arc::default();
    let model = register(
        &mut app,
        MODEL,
        Flaky {
            script: Mutex::new(script.into()),
            requests: Arc::clone(&requests),
        },
    );
    let tool = register(&mut app, ADD, Adder::new(ADD));
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut().entity_mut(agent).insert(MaxTurns(4));
    app.world_mut().spawn((Grant(tool), ChildOf(agent)));
    (app, agent, requests, recorder, witness)
}

fn failure(world: &World, run: Entity) -> Failure {
    world.get::<Failed>(run).expect("the run failed").0.clone()
}

fn retried(world: &World, run: Entity) -> usize {
    world
        .get::<ProviderRetried>(run)
        .expect("a run counts its retries")
        .0
}

fn retry_facts(log: &ObservationLog) -> Vec<(usize, usize, String)> {
    log.trace()
        .observations
        .iter()
        .filter_map(|o| match &o.action {
            Action::Host { kind, payload } if kind == "rig-ecs/agent/provider_retry" => Some((
                payload["attempt"].as_u64().unwrap() as usize,
                payload["budget"].as_u64().unwrap() as usize,
                payload["reason"]["code"].as_str().unwrap().to_owned(),
            )),
            _ => None,
        })
        .collect()
}

fn bound_entity(world: &mut World, key: &str) -> Entity {
    world
        .query::<(Entity, &Bound)>()
        .iter(world)
        .find(|(_, b)| b.key == HandlerKey::from(key))
        .map(|(e, _)| e)
        .expect("key bound")
}

#[test]
fn a_retryable_failure_after_tool_work_is_reissued_and_the_tool_runs_once() {
    let (mut app, agent, requests, recorder, witness) = tooling(vec![
        Ok(add_call()),
        Err(unavailable("status 503 Service Unavailable")),
        Ok(done()),
    ]);
    let run = app.world_mut().spawn_run(agent, &[], "add", false, None);
    stamp_run(app.world_mut(), run, &recorder).expect("the run stamps its program identity");
    ended(&mut app, run, "the retried run");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.clone()),
        Some("done".into()),
        "{:?}",
        app.world().get::<Failed>(run)
    );

    // Three requests: the call, the lost attempt, the retry. The retry is
    // the lost attempt again: same history, the one tool result in it.
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1].chat_history, requests[2].chat_history);
    let log: EffectLog =
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
    let tools = log
        .iter()
        .filter(|r| matches!(r.kind, EffectKind::ToolCall { .. }))
        .count();
    assert_eq!(tools, 1, "the tool ran once");
    let completions: Vec<bool> = log
        .iter()
        .filter(|r| matches!(r.kind, EffectKind::Completion { .. }))
        .map(|r| r.outcome.is_ok())
        .collect();
    assert_eq!(
        completions,
        [true, false, true],
        "every attempt is a record"
    );

    // One retry spent; the lost attempt was not a model call against the
    // budget: two turns, not three.
    assert_eq!(retried(app.world(), run), 1);
    assert_eq!(app.world().get::<Cursor>(run).unwrap().turn, 2);
    assert_eq!(
        retry_facts(&witness),
        [(1, 3, "provider_response".to_owned())]
    );
    drop(requests);

    // The log replays: the same program over by-id replayers sees the
    // failed attempt answered from its record, retries, and settles.
    let mut replay = run_support::app();
    Replay::default()
        .register(replay.world_mut(), &log)
        .unwrap();
    let model = bound_entity(replay.world_mut(), MODEL);
    let tool = bound_entity(replay.world_mut(), ADD);
    let agent = spawn_agent(replay.world_mut(), "t", model);
    replay.world_mut().entity_mut(agent).insert(MaxTurns(4));
    replay.world_mut().spawn((Grant(tool), ChildOf(agent)));
    let run = replay.world_mut().spawn_run(agent, &[], "add", false, None);
    ended(&mut replay, run, "the replayed run");
    assert_eq!(
        replay.world().get::<RunResult>(run).map(|r| r.0.clone()),
        Some("done".into()),
        "{:?}",
        replay.world().get::<Failed>(run)
    );
    assert_eq!(retried(replay.world(), run), 1);
}

#[test]
fn a_spent_budget_ends_the_run_with_the_last_report() {
    let (mut app, agent, requests, _, witness) = tooling(vec![
        Ok(add_call()),
        Err(unavailable("first")),
        Err(unavailable("second")),
        Err(unavailable("third")),
        Ok(done()),
    ]);
    app.world_mut().entity_mut(agent).insert(ProviderRetries(2));
    let run = app.world_mut().spawn_run(agent, &[], "add", false, None);
    ended(&mut app, run, "the exhausted run");
    let Failure::Provider(report) = failure(app.world(), run) else {
        panic!("{:?}", failure(app.world(), run));
    };
    assert_eq!(report.message, "third");
    assert_eq!(
        requests.lock().unwrap().len(),
        4,
        "the call, then three attempts"
    );
    assert_eq!(retried(app.world(), run), 2);
    assert_eq!(
        retry_facts(&witness)
            .iter()
            .map(|(attempt, budget, _)| (*attempt, *budget))
            .collect::<Vec<_>>(),
        [(1, 2), (2, 2)]
    );
}

#[test]
fn a_non_retryable_failure_after_tool_work_ends_the_run_at_once() {
    let (mut app, agent, requests, _, witness) = tooling(vec![
        Ok(add_call()),
        Err(ErrorReport::new(ErrorKind::Provider, "blocked: SAFETY")),
        Ok(done()),
    ]);
    let run = app.world_mut().spawn_run(agent, &[], "add", false, None);
    ended(&mut app, run, "the refused run");
    let Failure::Provider(report) = failure(app.world(), run) else {
        panic!("{:?}", failure(app.world(), run));
    };
    assert_eq!(report.message, "blocked: SAFETY");
    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(retried(app.world(), run), 0);
    assert!(retry_facts(&witness).is_empty());
}

#[test]
fn a_zero_budget_never_retries() {
    let (mut app, agent, requests, _, _) =
        tooling(vec![Err(unavailable("status 503")), Ok(done())]);
    app.world_mut().entity_mut(agent).insert(ProviderRetries(0));
    let run = app.world_mut().spawn_run(agent, &[], "hi", false, None);
    ended(&mut app, run, "the unretried run");
    assert!(matches!(failure(app.world(), run), Failure::Provider(_)));
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(retried(app.world(), run), 0);
}

/// Whether the host's backoff hold is in force.
#[derive(Resource)]
struct HostBackoff(bool);

/// A host `Gate` system: the re-issued completion of a run that has
/// retried waits under a hold until the host releases it (the backoff).
fn hold_retried_completion(
    fresh: Query<(Entity, &PendingEffect, &ChildOf), Added<PendingEffect>>,
    turns: Query<&ChildOf>,
    runs: Query<&ProviderRetried>,
    backoff: Res<HostBackoff>,
    mut commands: Commands,
) {
    if !backoff.0 {
        return;
    }
    for (entity, effect, turn_of) in &fresh {
        if !matches!(effect.kind, EffectKind::Completion { .. }) {
            continue;
        }
        let Ok(run_of) = turns.get(turn_of.parent()) else {
            continue;
        };
        if runs.get(run_of.parent()).is_ok_and(|retried| retried.0 > 0) {
            commands.entity(entity).insert(Held);
        }
    }
}

fn holding(app: &mut bevy_app::App) -> Vec<Entity> {
    app.world_mut()
        .query_filtered::<Entity, (With<PendingEffect>, With<Held>)>()
        .iter(app.world())
        .collect()
}

#[test]
fn a_host_hold_is_where_a_backoff_goes_and_a_cancel_during_it_ends_the_run() {
    let (mut app, agent, requests, _, _) = tooling(vec![
        Ok(add_call()),
        Err(unavailable("status 503")),
        Ok(done()),
    ]);
    app.insert_resource(HostBackoff(true));
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, hold_retried_completion.in_set(BusSet::Gate));
    let run = app.world_mut().spawn_run(agent, &[], "add", false, None);
    tick_until(&mut app, "the retry is held", |world| {
        world
            .query_filtered::<Entity, (With<PendingEffect>, With<Held>)>()
            .iter(world)
            .next()
            .is_some()
    });
    // Many passes later the hold stands: no third request, no ending.
    for _ in 0..8 {
        app.update();
    }
    assert_eq!(requests.lock().unwrap().len(), 2);
    assert!(app.world().get::<Settled>(run).is_none());
    assert!(app.world().get::<Failed>(run).is_none());
    assert_eq!(retried(app.world(), run), 1);

    // A cancel during the backoff ends the run; the held attempt is never
    // sent.
    app.world_mut()
        .entity_mut(run)
        .insert(Cancelled("the host gave up".into()));
    ended(&mut app, run, "the cancelled run");
    assert!(
        matches!(failure(app.world(), run), Failure::Cancelled(_)),
        "{:?}",
        failure(app.world(), run)
    );
    assert_eq!(requests.lock().unwrap().len(), 2);
    assert!(holding(&mut app).is_empty());
}

#[test]
fn a_checkpoint_saved_during_the_hold_resumes_into_the_retry() {
    let (mut app, agent, requests, recorder, _) = tooling(vec![
        Ok(add_call()),
        Err(unavailable("status 503")),
        Ok(done()),
    ]);
    app.insert_resource(HostBackoff(true));
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, hold_retried_completion.in_set(BusSet::Gate));
    let run = app.world_mut().spawn_run(agent, &[], "add", false, None);
    stamp_run(app.world_mut(), run, &recorder).expect("the run stamps its program identity");
    tick_until(&mut app, "the retry is held", |world| {
        world
            .query_filtered::<Entity, (With<PendingEffect>, With<Held>)>()
            .iter(world)
            .next()
            .is_some()
    });
    assert_eq!(requests.lock().unwrap().len(), 2);
    let saved = save_world(app.world_mut()).expect("every component serializes");
    let json = saved.to_json().expect("serde");
    assert!(
        json.contains(std::any::type_name::<ProviderRetried>()),
        "the spent retry is checkpoint data: {json}"
    );
    drop(app);

    // A fresh world, the same handlers, no hold: the loaded run carries its
    // spent retry and its held attempt, which is released and answered.
    let saved = Checkpoint::from_json(&json).expect("serde");
    let mut app = run_support::app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    let requests: Requests = Arc::default();
    register(
        &mut app,
        MODEL,
        Flaky {
            script: Mutex::new(vec![Ok(done())].into()),
            requests: Arc::clone(&requests),
        },
    );
    register(&mut app, ADD, Adder::new(ADD));
    let loaded = load_world(&saved, app.world_mut(), RestoreMode::Strict, [])
        .expect("the handlers are bound");
    let run = loaded.with::<rig_ecs::agent::Run>(app.world())[0];
    assert_eq!(retried(app.world(), run), 1, "the spent retry is restored");
    for held in holding(&mut app) {
        app.world_mut().entity_mut(held).remove::<Held>();
    }
    ended(&mut app, run, "the resumed run");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.clone()),
        Some("done".into()),
        "{:?}",
        app.world().get::<Failed>(run)
    );
    // One request in the new world: the retry, over the saved history with
    // its tool result; the tool did not run again.
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0].chat_history.iter().any(|m| matches!(m, rig_core::message::Message::User { content } if content.iter().any(|p| matches!(p, rig_core::message::UserContent::ToolResult(_))))),
        "the saved tool result is in the retried request"
    );
    assert_eq!(retried(app.world(), run), 1);
}

/// A model whose first stream closes before its terminal record and whose
/// second call answers.
struct Truncating {
    calls: std::sync::atomic::AtomicUsize,
}

impl Serve for Truncating {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(MODEL),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new(MODEL),
                capabilities: ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            return Reply::written(move |mut writer| async move {
                let id = BlockId::Wire("cut".into());
                writer
                    .event(StreamEvent::BlockStart {
                        id: id.clone(),
                        kind: BlockKind::Text {
                            additional_params: None,
                        },
                    })
                    .await
                    .expect("open stream");
                writer
                    .event(StreamEvent::BlockDelta {
                        id,
                        delta: Delta::Text { text: "par".into() },
                    })
                    .await
                    .expect("open stream");
                // The connection drops here: no terminal record.
            });
        }
        Reply::Outcome(Ok(Outcome::Completion(CompletionResponse::new(
            done(),
            Usage::default(),
            "whole",
            serde_json::json!({}),
        ))))
    }
}

#[test]
fn a_truncated_stream_is_reissued() {
    assert!(
        rig_core::serve::stream_truncated().is_retryable(),
        "a truncation is a transport fault"
    );
    let mut app = run_support::app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    let witness = Arc::new(ObservationLog::default());
    Witnessing::install(app.world_mut(), witness.clone());
    let model = register(
        &mut app,
        MODEL,
        Truncating {
            calls: std::sync::atomic::AtomicUsize::new(0),
        },
    );
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut().entity_mut(agent).insert(MaxTurns(4));
    let run = app.world_mut().spawn_run(agent, &[], "hi", true, None);
    ended(&mut app, run, "the retried stream");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.clone()),
        Some("done".into()),
        "{:?}",
        app.world().get::<Failed>(run)
    );
    assert_eq!(retried(app.world(), run), 1);
    assert_eq!(retry_facts(&witness).len(), 1);
    let log: EffectLog =
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
    let completions: Vec<bool> = log
        .iter()
        .filter(|r| matches!(r.kind, EffectKind::Completion { .. }))
        .map(|r| r.outcome.is_ok())
        .collect();
    assert_eq!(completions, [false, true]);
}

/// `agent::Backoff` on the agent: the retry's completion is held as
/// `rig-ecs/backoff` and released when the world's clock has advanced by
/// the delay — a paused `Time<Virtual>` holds it for as long as the host
/// likes, and advancing the clock by hand releases it.
#[test]
fn a_backoff_on_the_agent_delays_the_retry_on_the_worlds_clock() {
    use bevy_time::{Time, Virtual};
    let (mut app, agent, requests, _, _) = tooling(vec![
        Ok(add_call()),
        Err(unavailable("status 503")),
        Ok(done()),
    ]);
    app.world_mut()
        .entity_mut(agent)
        .insert(rig_ecs::agent::Backoff {
            base: Duration::from_secs(10),
            max: Duration::from_secs(60),
        });
    app.world_mut().resource_mut::<Time<Virtual>>().pause();
    let run = app.world_mut().spawn_run(agent, &[], "add", false, None);
    tick_until(&mut app, "the retry is held by the backoff", |world| {
        world
            .query_filtered::<&rig_ecs::bus::HoldOwners, With<PendingEffect>>()
            .iter(world)
            .any(|owners| {
                owners
                    .owners()
                    .any(|owner| owner.name == rig_ecs::systems::backoff::BACKOFF_OWNER)
            })
    });
    for _ in 0..8 {
        app.update();
    }
    assert_eq!(requests.lock().unwrap().len(), 2, "the clock is paused");
    assert!(app.world().get::<Settled>(run).is_none());
    // Nine seconds is not ten.
    app.world_mut()
        .resource_mut::<Time<Virtual>>()
        .advance_by(Duration::from_secs(9));
    for _ in 0..4 {
        app.update();
    }
    assert_eq!(requests.lock().unwrap().len(), 2);
    app.world_mut()
        .resource_mut::<Time<Virtual>>()
        .advance_by(Duration::from_secs(1));
    ended(&mut app, run, "the retried run");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.clone()),
        Some("done".into())
    );
    assert_eq!(requests.lock().unwrap().len(), 3);
    assert!(holding(&mut app).is_empty());
}
