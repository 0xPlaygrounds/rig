//! The stage rules on their own: plugin effects ([`EffectRequest`]) and a
//! scripted model's tool calls ([`ToolRequest`]) on a headless kernel app,
//! with stage systems as plugins write them. Effects chain within the
//! frame their outcomes arrive in, the first decision wins, Bevy sees two
//! unordered deciders of one kind, holds add up, every Prepare and Gate
//! system sees a late spawn before it starts, reviewers edit an outcome in
//! place in their order and the record keeps the raw one, a failing stage system
//! closes its stage, and every way an effect ends is recorded. A reply's
//! calls start, and their results go back, in the frames they arrive; a
//! gate sees each call when it is about to run; a held call holds back
//! only the later calls that touch what it writes, as Prepare left them; a
//! refusal answers in call order and is recorded as one. A model call that gave up is superseded
//! on another model and the turn goes on; a retry waits on a hold that an
//! interrupt takes away; a refused model call ends its turn and gives the
//! message back; a request too big for the context window fails its turn
//! without a retry and gives the message back; a failed turn gives back
//! the user's messages no model answered; a reply that came as a plugin
//! failed its turn is kept, its calls answered as not run; an image goes
//! to a model that reads none as its name. A side call is built from its
//! agent like a turn's and leaves
//! the conversation alone. A tool's annotations give its footprint, and what it publishes
//! lands on its call. A turn's end is a proposal a reviewer carries on
//! from, holds or replaces, and a new turn call carries on after a length
//! cutoff; a message taken over in Gate never reaches the agent, one
//! accepted carries an answered turn on, and an accepted note or steering
//! message its sender decides is withdrawn, even as the next call is
//! built; a turn a plugin fails starts nothing more; an interrupt ends a
//! turn at once, settling its held and streaming calls.
//!
//! Of the design's twelve rule tests (design.md §5), these land here: 1
//! (single-frame chaining, with a plugin effect and with a reply's calls),
//! 2 (gate when ready), 3 (refusal order), 4 (the conflict rule), 5
//! (admission), 6 (review), 7 (turn endings, and an `Incoming` taken over
//! in Gate is never delivered), 8 (re-entry), 9 (refusal
//! is not failure), 10 (fail closed) and 11 (interrupt). 12 (restore) is
//! in `src/restore/tests.rs`.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy_ecs::error::{FallbackErrorHandler, warn};
use bevy_ecs::schedule::{
    LogLevel, ScheduleBuildError, ScheduleBuildSettings, ScheduleBuildWarning,
};
use bevy_ecs::system::ScheduleSystem;
use bevy_time::TimeUpdateStrategy;
use futures::StreamExt;
use rig_core::ProviderResponseError;
use rig_core::catalog::{Catalog, ModelSpec, Transport};
use rig_core::completion::{FinishReason, ToolAnnotations};
use rig_core::effect::family::Dynamic;
use rig_core::effect::{
    Disposition, EffectKind, EffectRecord, FamilyDescriptor, HandlerDescriptor, HandlerKey,
};
use rig_core::error::ErrorKind;
use rig_core::message::{CallId, Message, StopReason, ToolFunction, ToolName};
use rig_core::operation::{Completion, Finish};
use rig_core::providers::registry::ProviderId;
use rig_core::serve::adapters::{ModelAdapter, ToolAdapter};
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::streaming::{Item, Part, Relayed, StreamEvent};
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_core::tool::{
    ContextValue, DynamicTool, ExitStatus, ToolContext, ToolErrorKind, ToolOutput, ToolResult,
};
use rig_ecs::effects::Handler;
use rig_ecs::prelude::*;
use rig_ecs::turn::RETRY_POLICY;
use serde::{Deserialize, Serialize};

/// How long a test waits for a tool to answer.
const DEADLINE: Duration = Duration::from_secs(10);

/// Adds two numbers.
struct Sum;

#[derive(Deserialize)]
struct AddArgs {
    a: i64,
    b: i64,
}

impl Tool for Sum {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i64;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Adds two integers.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(&self, _: &mut ToolContext, args: AddArgs) -> Result<i64, ToolExecutionError> {
        Ok(args.a + args.b)
    }
}

/// Never answers.
struct Stall;

impl Tool for Stall {
    const NAME: &'static str = "stall";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Waits forever.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        std::future::pending::<()>().await;
        Ok(String::new())
    }
}

/// The effect of calling `tool`, run by the kernel.
fn run<T: Tool + 'static>(tool: T, args: &str) -> EffectRequest {
    EffectRequest {
        kind: EffectKind::ToolCall {
            name: T::NAME.to_owned(),
            args: args.to_owned(),
        },
        handler: Some(Handler(ErasedHandler::new(ToolAdapter::new(tool)))),
    }
}

fn add() -> EffectRequest {
    run(Sum, r#"{"a":2,"b":3}"#)
}

/// An open effect of `kind`, which a plugin answers.
fn open(kind: &str) -> EffectRequest {
    EffectRequest {
        kind: EffectKind::Custom {
            kind: Arc::from(kind),
            payload: serde_json::Value::Null,
        },
        handler: None,
    }
}

/// What the stage systems saw, in order.
#[derive(Resource, Default)]
struct Seen(Vec<String>);

/// Every [`EffectRecorded`] and [`Notice`] so far.
#[derive(Resource, Default)]
struct Written {
    records: Vec<EffectRecord>,
    notices: Vec<String>,
}

/// A headless kernel app, started, that logs stage failures as warnings as
/// rig-harness does.
fn app() -> App {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .insert_resource(FallbackErrorHandler(warn))
        .init_resource::<Seen>()
        .init_resource::<Written>()
        .add_systems(
            Last,
            |mut records: MessageReader<EffectRecorded>,
             mut notices: MessageReader<Notice>,
             mut written: ResMut<Written>| {
                let records = records.read().map(|recorded| recorded.0.clone());
                written.records.extend(records);
                let notices = notices.read().map(|notice| notice.text.clone());
                written.notices.extend(notices);
            },
        );
    app.finish();
    app.update();
    app
}

/// Runs frames until `done`, or the deadline; whether it was done.
fn run_until(app: &mut App, mut done: impl FnMut(&mut World) -> bool) -> bool {
    let started = Instant::now();
    while started.elapsed() < DEADLINE {
        app.update();
        if done(app.world_mut()) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    false
}

fn records(app: &App) -> Vec<EffectRecord> {
    app.world().resource::<Written>().records.clone()
}

fn notices(app: &App) -> Vec<String> {
    app.world().resource::<Written>().notices.clone()
}

fn seen(app: &App) -> Vec<String> {
    app.world().resource::<Seen>().0.clone()
}

fn has<C: Component>(app: &App, entity: Entity) -> bool {
    app.world().get::<C>(entity).is_some()
}

/// An outcome as it is written in a log.
fn output(outcome: &Result<rig_core::effect::Answered, rig_core::error::ErrorReport>) -> String {
    serde_json::to_string(outcome).unwrap_or_default()
}

/// The chain's step.
#[derive(Component)]
struct Step(u32);

/// The frame number.
#[derive(Resource, Default)]
struct Frame(u32);

/// The time step a Gate system saw on each run, by frame.
#[derive(Resource, Default)]
struct Deltas(Vec<(u32, Duration)>);

#[test]
fn effects_chain_in_the_frame_their_outcomes_arrive_and_time_steps_once() {
    let mut app = app();
    let step = Duration::from_millis(10);
    app.insert_resource(TimeUpdateStrategy::ManualDuration(step))
        .init_resource::<Frame>()
        .init_resource::<Deltas>()
        .add_systems(First, |mut frame: ResMut<Frame>| frame.0 += 1)
        .add_systems(
            AgentPass,
            (|time: Res<Time>, frame: Res<Frame>, mut deltas: ResMut<Deltas>| {
                deltas.0.push((frame.0, time.delta()));
            })
            .in_set(Stage::Gate),
        )
        // Each step that settles asks for the next, up to three.
        .add_observer(
            |settled: On<Add<Settled>>, steps: Query<&Step>, mut commands: Commands| {
                if let Ok(Step(n)) = steps.get(settled.entity)
                    && *n < 3
                {
                    commands.spawn((add(), Step(n + 1)));
                }
            },
        );
    app.world_mut().spawn((add(), Step(1)));
    // Whether the next step had started in the frame each step settled.
    let mut chained = Vec::new();
    let done = run_until(&mut app, |world| {
        let mut steps = world.query::<(&Step, Has<Settled>, Has<Started>)>();
        let steps: Vec<(u32, bool, bool)> = steps
            .iter(world)
            .map(|(step, settled, started)| (step.0, settled, started))
            .collect();
        for &(n, settled, _) in steps.iter().filter(|(n, ..)| *n < 3) {
            if settled {
                chained.push(steps.iter().any(|&(m, _, started)| m == n + 1 && started));
            }
        }
        world.resource::<Written>().records.len() == 3
    });
    assert!(done, "three steps ran");
    assert_eq!(chained, [true, true], "each step started the next at once");
    let records = records(&app);
    let ids: Vec<u64> = records.iter().map(|record| record.id.as_u64()).collect();
    assert!(ids.is_sorted() && ids.len() == 3, "{ids:?}");
    for record in &records {
        assert_eq!(record.disposition, Disposition::Ran);
        assert!(output(&record.outcome).contains('5'), "{record:?}");
        assert!(record.reviewed.is_none());
        assert!(record.started_ms.is_some() && record.finished_ms.is_some());
    }
    // A frame's first pass sees its time step; the passes after it, none.
    let deltas = &app.world().resource::<Deltas>().0;
    let mut frames: Vec<u32> = deltas.iter().map(|(frame, _)| *frame).collect();
    frames.dedup();
    let mut passes = Vec::new();
    for frame in frames {
        let seen: Vec<Duration> = deltas
            .iter()
            .filter(|(at, _)| *at == frame)
            .map(|(_, delta)| *delta)
            .collect();
        assert!(
            matches!(seen.split_first(), Some((first, rest)) if *first == step && rest.iter().all(Duration::is_zero)),
            "frame {frame}: {seen:?}"
        );
        passes.push(seen.len());
    }
    assert!(passes.iter().any(|&n| n > 1), "{passes:?}");
    assert_eq!(app.world().resource::<Time>().delta(), step, "restored");
}

#[test]
fn the_first_decision_wins() {
    let mut app = app();
    app.add_systems(
        AgentPass,
        (
            |mut effects: Query<&mut Decision, (ReadyToStart, PluginEffects)>| {
                for mut decision in &mut effects {
                    decision.decide(Decider::plugin("refuser"), Outcome::refused("not now"));
                }
            },
            // Ordered after the refusal, this one decides too: it loses,
            // and it is told so.
            |mut effects: Query<&mut Decision, (ReadyToStart, PluginEffects)>,
             mut seen: ResMut<Seen>| {
                for mut decision in &mut effects {
                    let allowed = rig_core::tool::ToolResult::skipped("allowed");
                    let won = decision.decide(Decider::plugin("allower"), allowed);
                    seen.0.push(format!("won: {won}"));
                }
            },
        )
            .chain()
            .in_set(Stage::Gate),
    );
    let effect = app.world_mut().spawn(open("deploy")).id();
    app.update();
    assert!(has::<Settled>(&app, effect), "settled in its first frame");
    assert!(!has::<Started>(&app, effect), "never started");
    assert_eq!(seen(&app), ["won: false"]);
    let decided = app.world().get::<Decision>(effect);
    assert!(decided.is_some_and(Decision::is_decided), "still decided");
    let records = records(&app);
    assert_eq!(records.len(), 1, "{records:?}");
    let Some(record) = records.first() else {
        return;
    };
    assert_eq!(record.disposition, Disposition::AnsweredBeforeStart);
    assert!(
        matches!(&record.outcome, Err(report) if report.kind == ErrorKind::Denied),
        "{record:?}"
    );
    assert!(record.reviewed.is_none(), "{record:?}");
    assert!(record.started_ms.is_none());
    assert_eq!(record.decided_by, Some(Decider::plugin("refuser")));
}

#[test]
fn an_effect_answered_before_its_decision_was_applied_names_no_decider() {
    let mut app = app();
    let refuse = |mut effects: Query<&mut Decision, (ReadyToStart, PluginEffects)>| {
        for mut decision in &mut effects {
            decision.decide(Decider::plugin("refuser"), Outcome::refused("not now"));
        }
    };
    // Answers `answered` itself, after the refusal and before Start
    // applies it.
    let answer = |effects: Query<(Entity, &EffectRequest), (ReadyToStart, PluginEffects)>,
                  mut commands: Commands| {
        for (entity, request) in &effects {
            if request.kind.name() == "answered" {
                commands.entity(entity).insert(Outcome::custom("answered"));
            }
        }
    };
    app.add_systems(AgentPass, (refuse, answer).chain().in_set(Stage::Gate));
    app.world_mut().spawn(open("refused"));
    app.world_mut().spawn(open("answered"));
    app.update();
    let records = records(&app);
    let decider = |kind: &str| {
        let record = records.iter().find(|record| record.kind.name() == kind);
        record.map(|record| record.decided_by.clone())
    };
    assert_eq!(decider("refused"), Some(Some(Decider::plugin("refuser"))));
    assert_eq!(decider("answered"), Some(None), "{records:?}");
}

/// What the gate in [`after_the_gates_an_answer_sees_their_decisions_and_holds`]
/// does to an effect.
#[derive(Component)]
enum Way {
    Refuse,
    Hold,
    Pass,
}

#[test]
fn after_the_gates_an_answer_sees_their_decisions_and_holds() {
    let mut app = app();
    app.add_systems(
        AgentPass,
        (
            // A stand-in answer, added before the gate but run after it:
            // it answers only what no gate refused or held.
            (|mut effects: Query<
                (&Name, &mut Decision, Has<Holds>),
                (PluginEffects, Without<Outcome>, Without<Settled>),
            >,
              mut seen: ResMut<Seen>| {
                for (name, mut decision, held) in &mut effects {
                    let what = if held {
                        "held"
                    } else if decision.is_decided() {
                        "decided"
                    } else {
                        decision.decide(Decider::plugin("test"), Outcome::custom("cached"));
                        "cached"
                    };
                    let what = format!("{name}: {what}");
                    if !seen.0.contains(&what) {
                        seen.0.push(what);
                    }
                }
            })
            .in_set(GateSystems::After),
            (|mut effects: Query<(Entity, &Way, &mut Decision), (ReadyToStart, PluginEffects)>,
              mut commands: Commands| {
                for (effect, way, mut decision) in &mut effects {
                    match way {
                        Way::Refuse => {
                            decision.decide(Decider::plugin("test"), Outcome::refused("no"));
                        }
                        Way::Hold => {
                            commands.spawn(HoldOf(effect));
                        }
                        Way::Pass => {}
                    }
                }
            })
            .in_set(Stage::Gate),
        ),
    );
    let world = app.world_mut();
    let refused = world
        .spawn((open("job"), Name::new("refused"), Way::Refuse))
        .id();
    let held = world
        .spawn((open("job"), Name::new("held"), Way::Hold))
        .id();
    let passed = world
        .spawn((open("job"), Name::new("passed"), Way::Pass))
        .id();
    app.update();
    let mut seen = seen(&app);
    seen.sort();
    assert_eq!(seen, ["held: held", "passed: cached", "refused: decided"]);
    let outcome = |effect| app.world().get::<Outcome>(effect).map(|outcome| &outcome.0);
    assert!(
        matches!(outcome(refused), Some(Err(report)) if report.kind == ErrorKind::Denied),
        "the gate's refusal stands"
    );
    assert!(
        matches!(outcome(passed), Some(Ok(_))),
        "answered in its place"
    );
    assert!(has::<Settled>(&app, refused) && has::<Settled>(&app, passed));
    assert!(!has::<Started>(&app, refused) && !has::<Started>(&app, passed));
    assert!(
        outcome(held).is_none() && !has::<Started>(&app, held),
        "still held"
    );
}

/// The open tool `wait`'s arguments.
#[derive(Deserialize, schemars::JsonSchema)]
struct Wait {
    signal: String,
}

/// What `wait`'s observer was told of a call, and whether it had started.
#[derive(Debug, PartialEq)]
struct Told {
    tool: Entity,
    call: Entity,
    agent: Entity,
    caller: AgentId,
    effect: rig_core::effect::EffectId,
    parent: Option<rig_core::effect::EffectId>,
    id: String,
    signal: String,
    started: bool,
}

#[derive(Resource, Default)]
struct ToldOf(Vec<Told>);

#[test]
fn an_open_tool_starts_a_call_only_once_its_arguments_parse() {
    let mut app = app();
    app.init_resource::<ToldOf>().add_open_tool(
        "wait",
        "Waits for a signal.",
        ToolOptions::default(),
        |called: On<ToolCalled<Wait>>,
         mut calls: Query<(&mut Decision, Has<Started>)>,
         mut told: ResMut<ToldOf>| {
            let Ok((mut decision, started)) = calls.get_mut(called.call) else {
                return;
            };
            decision.decide(Decider::plugin("test"), Outcome::custom("done"));
            told.0.push(Told {
                tool: called.entity,
                call: called.call,
                agent: called.agent,
                caller: called.agent_id.clone(),
                effect: called.effect,
                parent: called.parent,
                id: called.tool_call.id.to_string(),
                signal: called.args.signal.clone(),
                started,
            });
        },
    );
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    let tool = world
        .query::<(Entity, &ToolDef)>()
        .iter(world)
        .find(|(_, def)| def.0.name.as_str() == "wait")
        .map(|(tool, _)| tool);
    let caller = world.get::<AgentId>(agent).cloned();
    let parent = rig_core::effect::EffectId::from_raw(7);
    let mut call = |id: &str, arguments| {
        let name = ToolName::new("wait").ok()?;
        let call = ToolCall::from_wire(id, ToolFunction::new(name, arguments));
        let spawned = (
            ToolRequest { call },
            ForAgent(agent),
            Effect::with_parent_id(Some(parent)),
        );
        Some(world.spawn(spawned).id())
    };
    let (Some(good), Some(bad)) = (
        call("good", serde_json::json!({ "signal": "go" })),
        call("bad", serde_json::json!({ "signal": 5 })),
    ) else {
        return;
    };
    let done = |world: &mut World| {
        world.get::<Settled>(good).is_some() && world.get::<Settled>(bad).is_some()
    };
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    // The call that fits started, then told its tool every field.
    let effect = app.world().get::<Effect>(good).map(Effect::id);
    let told = &app.world().resource::<ToldOf>().0;
    let (Some(effect), Some(tool), Some(caller)) = (effect, tool, caller) else {
        return;
    };
    let good_told = Told {
        tool,
        call: good,
        agent,
        caller,
        effect,
        parent: Some(parent),
        id: "good".to_owned(),
        signal: "go".to_owned(),
        started: true,
    };
    assert_eq!(told.as_slice(), [good_told]);
    // The call that does not fit was answered before it started, with an
    // error of its kind, the model's mistake and no refusal.
    assert!(!has::<Started>(&app, bad), "never started");
    let records = records(&app);
    let refused = records
        .iter()
        .find(|record| matches!(&record.kind, EffectKind::ToolCall { .. }) && record.id != effect);
    assert!(
        refused.is_some_and(|record| {
            record.disposition == Disposition::AnsweredBeforeStart
                && record.started_ms.is_none()
                && matches!(&record.outcome, Err(report)
                    if report.kind == ErrorKind::Tool(ToolErrorKind::InvalidArgs)
                        && !report.refusal)
        }),
        "{records:?}"
    );
}

/// A gate that refuses every tool call.
fn refuse_calls(mut calls: Query<&mut Decision, (ReadyToStart, ToolCalls)>) {
    for mut decision in &mut calls {
        decision.decide(Decider::plugin("test"), Outcome::refused("no"));
    }
}

/// Another gate that allows every tool call.
fn allow_calls(mut calls: Query<&mut Decision, (ReadyToStart, ToolCalls)>) {
    for mut decision in &mut calls {
        decision.decide(
            Decider::plugin("test"),
            rig_core::tool::ToolResult::skipped("allowed"),
        );
    }
}

/// A gate that takes every message over.
fn take_messages(mut messages: Query<&mut Decision, (ReadyToStart, IncomingMessages)>) {
    for mut decision in &mut messages {
        decision.decide(Decider::plugin("test"), Outcome::custom("taken"));
    }
}

/// Whether the agent loop with `gates` in Gate holds an unordered pair
/// writing the same data, which `None` means it built without.
fn unordered<M>(gates: impl IntoScheduleConfigs<ScheduleSystem, M>) -> Option<bool> {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_systems(AgentPass, gates.in_set(Stage::Gate));
    let built = app
        .world_mut()
        .try_schedule_scope(AgentPass, |world, pass| {
            pass.set_build_settings(ScheduleBuildSettings {
                ambiguity_detection: LogLevel::Error,
                ..Default::default()
            });
            pass.initialize(world).map(drop)
        })
        .ok()?;
    Some(matches!(
        built,
        Err(ScheduleBuildError::Elevated(
            ScheduleBuildWarning::Ambiguity(_)
        ))
    ))
}

#[test]
fn bevy_sees_two_unordered_deciders_of_one_kind_only() {
    assert_eq!(unordered((refuse_calls, allow_calls)), Some(true));
    assert_eq!(unordered((refuse_calls, allow_calls).chain()), Some(false));
    assert_eq!(unordered((refuse_calls, take_messages)), Some(false));
}

#[test]
fn holds_add_up() {
    let mut app = app();
    let world = app.world_mut();
    let effect = world.spawn(open("job")).id();
    let first = world.spawn(HoldOf(effect)).id();
    let second = world.spawn(HoldOf(effect)).id();
    app.update();
    assert!(has::<Admitted>(&app, effect) && !has::<Started>(&app, effect));
    app.world_mut().despawn(first);
    app.update();
    assert!(!has::<Started>(&app, effect), "one hold is left");
    app.world_mut().entity_mut(second).remove::<HoldOf>();
    app.update();
    assert!(
        has::<Started>(&app, effect),
        "started in the frame of the release"
    );
}

/// What a question holds until it is answered.
#[derive(Component)]
struct Asked {
    call: Entity,
    hold: Entity,
}

#[test]
fn an_effect_decided_as_its_hold_goes_never_starts() {
    let mut app = app();
    // A question is answered in Gate; on its answer, an observer refuses
    // the call it held and lets go of it, as Start opens.
    app.add_systems(
        AgentPass,
        (|mut asked: Query<&mut Decision, (ReadyToStart, With<Asked>)>| {
            for mut decision in &mut asked {
                decision.decide(Decider::plugin("test"), Outcome::custom("no"));
            }
        })
        .in_set(Stage::Gate),
    )
    .add_observer(
        |answered: On<Add<Outcome>>,
         asked: Query<&Asked>,
         mut decisions: Query<&mut Decision>,
         mut commands: Commands| {
            let Ok(asked) = asked.get(answered.entity) else {
                return;
            };
            if let Ok(mut decision) = decisions.get_mut(asked.call) {
                decision.decide(
                    Decider::plugin("test"),
                    Outcome::refused("the user said no"),
                );
            }
            commands.entity(asked.hold).despawn();
        },
    );
    let world = app.world_mut();
    let call = world.spawn(add()).id();
    let hold = world.spawn(HoldOf(call)).id();
    world.spawn((open("question"), Asked { call, hold }));
    let settled = run_until(&mut app, |world| world.get::<Settled>(call).is_some());
    assert!(settled, "the refusal settled the call");
    assert!(!has::<Started>(&app, call), "a refused call never starts");
    let records = records(&app);
    let record = records
        .iter()
        .find(|record| matches!(record.kind, EffectKind::ToolCall { .. }));
    assert!(
        record.is_some_and(
            |record| record.disposition == Disposition::AnsweredBeforeStart
                && matches!(&record.outcome, Err(report) if report.kind == ErrorKind::Denied)
        ),
        "{records:?}"
    );
}

/// A mark that moves an effect to another table.
#[derive(Component)]
struct Moved;

#[test]
fn effects_start_in_the_order_they_were_spawned() {
    let mut app = app();
    let order =
        |started: On<Add<Started>>, effects: Query<&EffectRequest>, mut seen: ResMut<Seen>| {
            if let Ok(request) = effects.get(started.entity) {
                seen.0.push(request.kind.name().to_owned());
            }
        };
    // Marking the first effect after it was spawned puts it after the
    // second in Bevy's query order.
    let mark = |new: Query<(Entity, &EffectRequest), Added<Admitted>>, mut commands: Commands| {
        for (effect, request) in &new {
            if request.kind.name() == "first" {
                commands.entity(effect).insert(Moved);
            }
        }
    };
    app.add_observer(order)
        .add_systems(AgentPass, mark.in_set(Stage::Prepare));
    let world = app.world_mut();
    world.spawn(open("first"));
    world.spawn(open("second"));
    app.update();
    assert_eq!(seen(&app), ["first", "second"]);
}

/// Whether the Gate system spawned the late effect.
#[derive(Resource, Default)]
struct SpawnedLate(bool);

fn noted<'a>(seen: &mut Seen, who: &str, names: impl IntoIterator<Item = &'a Name>) {
    for name in names {
        seen.0.push(format!("{who} {}", name.as_str()));
    }
}

#[test]
fn every_prepare_and_gate_system_sees_a_late_spawn_before_it_starts() {
    let mut app = app();
    app.init_resource::<SpawnedLate>()
        .add_systems(
            AgentPass,
            (
                |names: Query<&Name, Added<Admitted>>, mut seen: ResMut<Seen>| {
                    noted(&mut seen, "prepare a", names);
                },
                |names: Query<&Name, Added<Admitted>>, mut seen: ResMut<Seen>| {
                    noted(&mut seen, "prepare b", names);
                },
            )
                .chain()
                .in_set(Stage::Prepare),
        )
        .add_systems(
            AgentPass,
            (
                |names: Query<&Name, ReadyToStart>,
                 mut seen: ResMut<Seen>,
                 mut late: ResMut<SpawnedLate>,
                 mut commands: Commands| {
                    noted(&mut seen, "gate a", names);
                    if !late.0 {
                        late.0 = true;
                        commands.spawn((open("late"), Name::new("late")));
                    }
                },
                |names: Query<&Name, ReadyToStart>, mut seen: ResMut<Seen>| {
                    noted(&mut seen, "gate b", names);
                },
            )
                .chain()
                .in_set(Stage::Gate),
        )
        .add_observer(
            |started: On<Add<Started>>, names: Query<&Name>, mut seen: ResMut<Seen>| {
                noted(&mut seen, "start", names.get(started.entity));
            },
        );
    app.world_mut().spawn((open("first"), Name::new("first")));
    app.update();
    let order = [
        "prepare a first",
        "prepare b first",
        "gate a first",
        "gate b first",
        "start first",
        "prepare a late",
        "prepare b late",
        "gate a late",
        "gate b late",
        "start late",
    ];
    assert_eq!(seen(&app), order);
}

/// Marks an outcome the first reviewer edited.
#[derive(Component)]
struct Redacted;

/// Marks an outcome the second reviewer edited.
#[derive(Component)]
struct Checked;

/// The first reviewer's hold.
#[derive(Resource, Default)]
struct ReviewHold(Option<Entity>);

#[test]
fn reviews_edit_in_place_in_order_and_the_record_keeps_the_raw_outcome() {
    let mut app = app();
    let redact =
        |mut effects: Query<(Entity, &mut Outcome), (ReadyToSettle, Without<Redacted>)>,
         mut hold: ResMut<ReviewHold>,
         mut commands: Commands| {
            for (effect, mut outcome) in &mut effects {
                *outcome = Outcome::from(rig_core::tool::ToolResult::skipped("redacted"));
                commands.entity(effect).insert(Redacted);
                hold.0 = Some(commands.spawn(HoldOf(effect)).id());
            }
        };
    // Ordered after, it sees the first edit as it would any data.
    let check = |mut effects: Query<
        (Entity, &mut Outcome),
        (ReadyToSettle, With<Redacted>, Without<Checked>),
    >,
                 mut seen: ResMut<Seen>,
                 mut commands: Commands| {
        for (effect, mut outcome) in &mut effects {
            let before = outcome.tool_result().output().render();
            seen.0.push(before.clone());
            let checked = format!("{before}, checked");
            *outcome = Outcome::from(rig_core::tool::ToolResult::skipped(checked));
            commands.entity(effect).insert(Checked);
        }
    };
    app.init_resource::<ReviewHold>()
        .add_systems(AgentPass, (redact, check).chain().in_set(Stage::Review));
    let effect = app.world_mut().spawn(add()).id();
    let held = run_until(&mut app, |world| world.resource::<ReviewHold>().0.is_some());
    assert!(held, "the tool answered");
    for _ in 0..3 {
        app.update();
    }
    assert!(!has::<Settled>(&app, effect), "a hold keeps it from Commit");
    assert!(seen(&app).is_empty(), "the later reviewer waits too");
    if let Some(hold) = app.world().resource::<ReviewHold>().0 {
        app.world_mut().despawn(hold);
    }
    app.update();
    assert!(has::<Settled>(&app, effect), "settled once released");
    let seen = seen(&app);
    assert!(
        matches!(seen.as_slice(), [only] if only.contains("redacted")),
        "the later reviewer saw the first edit: {seen:?}"
    );
    let records = records(&app);
    assert_eq!(records.len(), 1, "{records:?}");
    let Some(record) = records.first() else {
        return;
    };
    assert!(
        output(&record.outcome).contains('5'),
        "the raw outcome: {record:?}"
    );
    let reviewed = record.reviewed.as_ref().map(output);
    assert!(
        reviewed.is_some_and(|reviewed| reviewed.contains("redacted, checked")),
        "both edits: {record:?}"
    );
    assert_eq!(record.disposition, Disposition::Ran);
}

/// Which stage's system fails.
#[derive(Resource, Default)]
struct Broken(Option<Stage>);

fn breaks(broken: &Broken, stage: Stage) -> bevy_ecs::error::Result {
    match broken.0 == Some(stage) {
        true => Err(format!("broke in {stage:?}").into()),
        false => Ok(()),
    }
}

fn prepare_breaks(broken: Res<Broken>) -> bevy_ecs::error::Result {
    breaks(&broken, Stage::Prepare)
}

fn gate_breaks(broken: Res<Broken>) -> bevy_ecs::error::Result {
    breaks(&broken, Stage::Gate)
}

fn review_breaks(broken: Res<Broken>) -> bevy_ecs::error::Result {
    breaks(&broken, Stage::Review)
}

#[test]
fn a_failing_stage_system_closes_its_stage() {
    let mut app = app();
    app.init_resource::<Broken>().add_systems(
        AgentPass,
        (
            prepare_breaks.in_set(Stage::Prepare),
            gate_breaks.in_set(Stage::Gate),
            review_breaks.in_set(Stage::Review),
        ),
    );
    let failed = |app: &App, stage: &str| {
        let notices = notices(app);
        notices
            .iter()
            .filter(|notice| notice.contains(&format!("in {stage}")))
            .count()
    };

    // A broken gate starts nothing, and says so once.
    app.world_mut().resource_mut::<Broken>().0 = Some(Stage::Gate);
    let gated = app.world_mut().spawn(open("gated")).id();
    app.update();
    app.update();
    assert!(!has::<Started>(&app, gated));
    assert_eq!(failed(&app, "Gate"), 1, "{:?}", notices(&app));
    // Mended, it starts at the next move.
    app.world_mut().resource_mut::<Broken>().0 = None;
    app.world_mut().spawn(open("next"));
    app.update();
    assert!(has::<Started>(&app, gated));

    // A broken Prepare system has what was admitted answered with an error.
    app.world_mut().resource_mut::<Broken>().0 = Some(Stage::Prepare);
    let prepared = app.world_mut().spawn(open("prepared")).id();
    app.update();
    assert!(has::<Settled>(&app, prepared) && !has::<Started>(&app, prepared));
    let refused = records(&app).into_iter().find(|record| {
        record.disposition == Disposition::AnsweredBeforeStart
            && matches!(&record.outcome, Err(report) if report.kind == ErrorKind::Internal)
    });
    assert!(refused.is_some(), "{:?}", records(&app));
    assert_eq!(failed(&app, "Prepare"), 1, "{:?}", notices(&app));

    // A broken reviewer settles nothing.
    app.world_mut().resource_mut::<Broken>().0 = Some(Stage::Review);
    let reviewed = app
        .world_mut()
        .spawn((open("reviewed"), Outcome::refused("no")))
        .id();
    app.update();
    assert!(!has::<Settled>(&app, reviewed));
    assert_eq!(failed(&app, "Review"), 1, "{:?}", notices(&app));
    app.world_mut().resource_mut::<Broken>().0 = None;
    app.world_mut().spawn(open("next"));
    app.update();
    assert!(has::<Settled>(&app, reviewed));
}

#[test]
fn every_way_an_effect_ends_is_recorded() {
    let mut app = app();
    let world = app.world_mut();
    let running = world.spawn(run(Stall, "{}")).id();
    let replaced = world.spawn(open("replaced")).id();
    world.spawn(HoldOf(replaced));
    let dropped = world.spawn(open("dropped")).id();
    let nested = world.spawn((open("nested"), ChildOf(running))).id();
    app.update();
    assert!(has::<Started>(&app, running) && has::<Running>(&app, running));
    let parent = app.world().get::<Effect>(running).map(Effect::id);
    let child = app
        .world()
        .get::<Effect>(nested)
        .and_then(|effect| effect.parent);
    assert_eq!(child, parent, "a child effect's parent");
    // Decided from outside while it runs: its task is cancelled.
    let answer = rig_core::tool::ToolResult::skipped("answered");
    let world = app.world_mut();
    if let Some(mut decision) = world.get_mut::<Decision>(running) {
        decision.decide(Decider::plugin("test"), answer);
    }
    // Replaced while held: it never starts.
    let successor = world.spawn(open("successor")).id();
    world.entity_mut(replaced).insert(SupersededBy(successor));
    // Despawned before it settles.
    world.despawn(dropped);
    app.update();
    assert!(has::<Settled>(&app, running) && !has::<Running>(&app, running));
    assert!(has::<Settled>(&app, replaced) && !has::<Started>(&app, replaced));
    assert!(has::<Started>(&app, successor));
    let records = records(&app);
    // The handler's record, with when it started.
    let stalled = records
        .iter()
        .find(|record| record.key.as_str() == "tool:stall");
    assert!(
        stalled.is_some_and(|record| record.started_ms.is_some() && Some(record.id) == parent),
        "{stalled:?}"
    );
    let disposition = |key: &str| {
        records
            .iter()
            .find(|record| record.key.as_str() == key)
            .map(|record| (record.disposition, output(&record.outcome)))
    };
    let answered = disposition("tool:stall");
    assert!(
        answered.as_ref().is_some_and(|(how, outcome)| {
            *how == Disposition::AnsweredWhileRunning && outcome.contains("answered")
        }),
        "{answered:?}"
    );
    assert_eq!(
        disposition("replaced").map(|(how, _)| how),
        Some(Disposition::Superseded)
    );
    assert_eq!(
        disposition("dropped").map(|(how, _)| how),
        Some(Disposition::Cancelled)
    );
}

/// Answers at once; changes nothing, so its calls run side by side.
struct Look;

impl Tool for Look {
    const NAME: &'static str = "look";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Looks.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        Ok("seen".to_owned())
    }
}

/// Answers at once; may change anything, so its calls run alone.
struct Touch;

impl Tool for Touch {
    const NAME: &'static str = "touch";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Touches.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        Ok("touched".to_owned())
    }
}

/// A model reply calling `look` or `touch` once for each of `calls`, by
/// call id.
fn reply(calls: &[(&str, &str)]) -> Vec<MockStreamEvent> {
    let calls = calls
        .iter()
        .map(|(tool, id)| MockStreamEvent::tool_call(*id, *tool, serde_json::json!({})));
    calls
        .chain([MockStreamEvent::final_response_with_default_usage()])
        .collect()
}

fn answer(text: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// The app's agent on `model`, with `look` and `touch`, asked to go; frames
/// are counted from 1 in `Frame`.
fn agent_on(app: &mut App, model: &MockCompletionModel) -> Entity {
    let agent = connected(app, model);
    app.world_mut()
        .trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    agent
}

/// The app's agent on `model`, with `look` and `touch`, asked nothing.
fn connected(app: &mut App, model: &MockCompletionModel) -> Entity {
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .map(|spec| spec.shared());
    let read_only = ToolOptions {
        footprint: Some(Touches::Reads(Scope::Anything)),
    };
    app.add_tool_with(Look, read_only)
        .add_tool(Touch)
        .init_resource::<Frame>()
        .add_systems(First, |mut frame: ResMut<Frame>| frame.0 += 1);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    if let Ok(spec) = spec {
        let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
        let handler = Handler(ErasedHandler::new(handler));
        world.entity_mut(agent).insert(Connection { spec, handler });
    }
    agent
}

fn conversation(app: &App, agent: Entity) -> Vec<Message> {
    let conversation = app.world().get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |conversation| {
        conversation
            .messages()
            .iter()
            .map(|message| (**message).clone())
            .collect()
    })
}

/// The calls named by id, with the call ids as the model sent them.
fn call_named(world: &mut World, id: &str) -> Option<Entity> {
    let mut calls = world.query::<(Entity, &ToolRequest)>();
    calls
        .iter(world)
        .find(|(_, request)| request.call.id.to_string() == id)
        .map(|(call, _)| call)
}

/// The tool results of the conversation's message `at`, in order: whether
/// each is an error, and how it reads.
fn results_at(messages: &[Message], at: usize) -> Vec<(bool, String)> {
    match messages.get(at) {
        Some(Message::User { content }) => content
            .iter()
            .filter_map(|item| match item {
                UserContent::ToolResult(result) => Some((result.is_error, format!("{result:?}"))),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

#[test]
fn a_replys_calls_start_and_their_results_go_back_in_the_frames_they_arrive() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([
        reply(&[("look", "l1"), ("look", "l2"), ("look", "l3")]),
        answer("done"),
    ]);
    let agent = agent_on(&mut app, &model);
    // The frame each step was first seen in, after the update. A started
    // model call is seen at the end of its frame even when it settled in
    // it: a settled effect stays until the next frame's loop.
    let (mut replied, mut started, mut answered, mut asked) = (None, None, None, None);
    let mut model_calls = HashSet::new();
    let done = run_until(&mut app, |world| {
        let frame = world.resource::<Frame>().0;
        let said = world
            .get::<Conversation>(agent)
            .map_or(0, |c| c.messages().len());
        let mut calls = world.query_filtered::<(), (With<ToolRequest>, With<Started>)>();
        let running = calls.iter(world).count();
        let mut models = world.query_filtered::<&Effect, (With<TurnCall>, With<Started>)>();
        model_calls.extend(models.iter(world).map(Effect::id));
        replied = replied.or((said >= 2).then_some(frame));
        started = started.or((running == 3).then_some(frame));
        answered = answered.or((said >= 3).then_some(frame));
        asked = asked.or((model_calls.len() == 2).then_some(frame));
        said == 4
    });
    assert!(done, "{:?}", conversation(&app, agent));
    assert!(replied.is_some() && answered.is_some());
    assert_eq!(started, replied, "the calls started in the reply's frame");
    assert_eq!(asked, answered, "the model was asked in the results' frame");
    let results = results_at(&conversation(&app, agent), 2);
    assert_eq!(results.len(), 3, "{results:?}");
}

/// What the gate saw: each call it was asked about, with the calls settled
/// by then.
#[derive(Resource, Default)]
struct Gated {
    seen: Vec<(String, Vec<String>)>,
    hold: Option<Entity>,
    held_once: bool,
}

/// Holds `held` once and refuses `refused`; logs every call it sees.
fn gate(
    mut calls: Query<(Entity, &ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>,
    settled: Query<&ToolRequest, With<Settled>>,
    mut gated: ResMut<Gated>,
    mut commands: Commands,
) {
    for (call, request, mut decision) in &mut calls {
        let id = request.call.id.to_string();
        let done = settled.iter().map(|done| done.call.id.to_string());
        gated.seen.push((id.clone(), done.collect()));
        match id.as_str() {
            "held" if !gated.held_once => {
                gated.held_once = true;
                gated.hold = Some(commands.spawn(HoldOf(call)).id());
            }
            "refused" => {
                decision.decide(Decider::plugin("test"), Outcome::refused("not this one"));
            }
            _ => {}
        }
    }
}

/// The frames the `refused` call was answered and `l2` started in.
#[derive(Resource, Default)]
struct Refusal {
    answered: Option<u32>,
    started: Option<u32>,
}

#[test]
fn a_gate_sees_each_call_when_it_is_about_to_run() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([
        reply(&[("touch", "a"), ("touch", "b")]),
        reply(&[("touch", "held"), ("look", "l1")]),
        reply(&[("touch", "refused"), ("look", "l2")]),
        answer("done"),
    ]);
    app.init_resource::<Gated>()
        .init_resource::<Refusal>()
        .add_systems(AgentPass, gate.in_set(Stage::Gate))
        .add_observer(
            |answered: On<Add<Outcome>>,
             calls: Query<&ToolRequest>,
             frame: Res<Frame>,
             mut refusal: ResMut<Refusal>| {
                if calls
                    .get(answered.entity)
                    .is_ok_and(|call| call.call.id.to_string() == "refused")
                {
                    refusal.answered = Some(frame.0);
                }
            },
        )
        .add_observer(
            |started: On<Add<Started>>,
             calls: Query<&ToolRequest>,
             frame: Res<Frame>,
             mut refusal: ResMut<Refusal>| {
                if calls
                    .get(started.entity)
                    .is_ok_and(|call| call.call.id.to_string() == "l2")
                {
                    refusal.started = Some(frame.0);
                }
            },
        );
    let agent = agent_on(&mut app, &model);
    let held = run_until(&mut app, |world| world.resource::<Gated>().hold.is_some());
    assert!(held, "the gate held a call");
    for _ in 0..3 {
        app.update();
    }
    // The held call holds back the read-only call after it.
    let world = app.world_mut();
    let l1 = call_named(world, "l1").unwrap_or(Entity::PLACEHOLDER);
    assert!(has::<Waiting>(&app, l1) && !has::<Started>(&app, l1));
    if let Some(hold) = app.world().resource::<Gated>().hold {
        app.world_mut().despawn(hold);
    }
    let done = run_until(&mut app, |world| {
        world
            .get::<Conversation>(agent)
            .map_or(0, |c| c.messages().len())
            == 8
    });
    assert!(done, "{:?}", conversation(&app, agent));

    // `b` was gated once `a` had settled, and `l1` once `held` had.
    let seen = &app.world().resource::<Gated>().seen;
    let gated_after = |id: &str, earlier: &str| {
        seen.iter()
            .filter(|(call, _)| call == id)
            .all(|(_, settled)| settled.iter().any(|done| done == earlier))
    };
    assert!(seen.iter().any(|(call, _)| call == "b"), "{seen:?}");
    assert!(
        gated_after("b", "a") && gated_after("l1", "held"),
        "{seen:?}"
    );
    // A refused call holds nothing back: `l2` started in the refusal's frame.
    let refusal = app.world().resource::<Refusal>();
    assert!(refusal.answered.is_some(), "the gate refused");
    assert_eq!(refusal.started, refusal.answered);
}

/// The file a call names.
#[derive(Deserialize)]
struct PathArgs {
    path: PathBuf,
}

/// Reads the file its call names, and touches nothing else.
struct ReadFile;

impl Tool for ReadFile {
    const NAME: &'static str = "read";
    type Args = PathArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Reads a file.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": { "path": { "type": "string" } } })
    }

    fn footprint(&self, args: &PathArgs) -> Touches {
        Touches::Reads(Scope::Paths(vec![args.path.clone()]))
    }

    async fn call(&self, _: &mut ToolContext, _: PathArgs) -> Result<String, ToolExecutionError> {
        Ok("read".to_owned())
    }
}

/// Changes the file its call names, and touches nothing else.
struct EditFile;

impl Tool for EditFile {
    const NAME: &'static str = "edit";
    type Args = PathArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Edits a file.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": { "path": { "type": "string" } } })
    }

    fn footprint(&self, args: &PathArgs) -> Touches {
        Touches::Writes(Scope::Paths(vec![args.path.clone()]))
    }

    async fn call(&self, _: &mut ToolContext, _: PathArgs) -> Result<String, ToolExecutionError> {
        Ok("edited".to_owned())
    }
}

/// Hands work on; registered as independent of the reply's other calls.
struct Task;

impl Tool for Task {
    const NAME: &'static str = "task";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Hands work on.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        Ok("handed on".to_owned())
    }
}

/// In Prepare, points the call `moved` at `/work/a`.
fn move_to_a(mut calls: Query<&mut ToolRequest, Added<Admitted>>) {
    for mut request in &mut calls {
        if request.call.id.to_string() == "moved" {
            let path = serde_json::json!("/work/a");
            request.call.function.arguments.insert("path".into(), path);
        }
    }
}

#[test]
fn a_held_call_holds_back_only_the_calls_that_touch_what_it_writes() {
    let mut app = app();
    let path = |tool: &str, id: &str, path: &str| {
        MockStreamEvent::tool_call(id, tool, serde_json::json!({ "path": path }))
    };
    let model = MockCompletionModel::from_stream_turns([
        vec![
            path("edit", "held", "/work/a"),
            path("read", "rb", "/work/b"),
            path("read", "ra", "/work/a"),
            MockStreamEvent::tool_call("t", "task", serde_json::json!({})),
            path("read", "moved", "/work/b"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        answer("done"),
    ]);
    let independent = ToolOptions {
        footprint: Some(Touches::Independent),
    };
    app.add_tool(ReadFile)
        .add_tool(EditFile)
        .add_tool_with(Task, independent)
        .init_resource::<Gated>()
        .add_systems(
            AgentPass,
            (move_to_a.in_set(Stage::Prepare), gate.in_set(Stage::Gate)),
        );
    let agent = agent_on(&mut app, &model);
    let held = run_until(&mut app, |world| world.resource::<Gated>().hold.is_some());
    assert!(held, "the gate held the edit");
    for _ in 0..3 {
        app.update();
    }
    // The read of another file ran past the held edit; the read of its
    // file, the read Prepare moved onto it and the hand-off wait.
    let world = app.world_mut();
    let [rb, ra, t, moved] =
        ["rb", "ra", "t", "moved"].map(|id| call_named(world, id).unwrap_or(Entity::PLACEHOLDER));
    assert!(has::<Started>(&app, rb));
    for waiting in [ra, t, moved] {
        assert!(has::<Waiting>(&app, waiting) && !has::<Started>(&app, waiting));
    }
    if let Some(hold) = app.world().resource::<Gated>().hold {
        app.world_mut().despawn(hold);
    }
    let done = run_until(&mut app, |world| {
        world
            .get::<Conversation>(agent)
            .is_some_and(|c| c.messages().len() == 4)
    });
    assert!(done, "{:?}", conversation(&app, agent));
    let seen = &app.world().resource::<Gated>().seen;
    let gated_after = |id: &str| {
        seen.iter()
            .filter(|(call, _)| call == id)
            .all(|(_, settled)| settled.iter().any(|done| done == "held"))
    };
    assert!(
        ["ra", "t", "moved"].into_iter().all(gated_after),
        "{seen:?}"
    );
}

/// In Prepare, declares that the call `held` writes only under `/work/src`.
fn declare_src(mut calls: Query<(Entity, &ToolRequest), Added<Admitted>>, mut commands: Commands) {
    for (call, request) in &mut calls {
        if request.call.id.to_string() == "held" {
            let src = Scope::Paths(vec![PathBuf::from("/work/src/../src")]);
            commands
                .entity(call)
                .insert((Footprint(Touches::Writes(src)), DeclaredFootprint));
        }
    }
}

#[test]
fn a_footprint_a_plugin_declared_on_a_call_stands() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("held", "touch", serde_json::json!({})),
            MockStreamEvent::tool_call(
                "docs",
                "read",
                serde_json::json!({ "path": "/work/docs/a.md" }),
            ),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        answer("done"),
    ]);
    app.add_tool(ReadFile).init_resource::<Gated>().add_systems(
        AgentPass,
        (declare_src.in_set(Stage::Prepare), gate.in_set(Stage::Gate)),
    );
    agent_on(&mut app, &model);
    let held = run_until(&mut app, |world| world.resource::<Gated>().hold.is_some());
    assert!(held, "the gate held the shell-like call");
    // Without the declaration, `touch` writes anything and the read waits.
    let docs = run_until(&mut app, |world| {
        call_named(world, "docs").is_some_and(|docs| world.get::<Started>(docs).is_some())
    });
    let world = app.world_mut();
    let held = call_named(world, "held").unwrap_or(Entity::PLACEHOLDER);
    let src = Scope::Paths(vec![PathBuf::from("/work/src")]);
    assert_eq!(
        world.get::<Footprint>(held).map(|f| f.0.clone()),
        Some(Touches::Writes(src))
    );
    assert!(docs, "the read of docs/ ran beside the held call");
}

/// The footprints Gate saw, by call id.
#[derive(Resource, Default)]
struct GatedFootprints(Vec<(String, Touches)>);

fn log_footprints(
    calls: Query<(&ToolRequest, &Footprint), (ReadyToStart, ToolCalls)>,
    mut log: ResMut<GatedFootprints>,
) {
    for (request, footprint) in &calls {
        log.0
            .push((request.call.id.to_string(), footprint.0.clone()));
    }
}

/// A handler's `..` cannot climb past a guard that compares footprint
/// paths: Gate sees each path folded, and a relative one as anything.
#[test]
fn gate_sees_footprint_paths_normalized() {
    let mut app = app();
    let path = |id: &str, path: &str| {
        MockStreamEvent::tool_call(id, "edit", serde_json::json!({ "path": path }))
    };
    let model = MockCompletionModel::from_stream_turns([
        vec![
            path("climbs", "/work/src/../../etc/./passwd"),
            path("relative", "notes.txt"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        answer("done"),
    ]);
    app.add_tool(EditFile)
        .init_resource::<GatedFootprints>()
        .add_systems(AgentPass, log_footprints.in_set(Stage::Gate));
    let agent = agent_on(&mut app, &model);
    let done = run_until(&mut app, |world| {
        world
            .get::<Conversation>(agent)
            .is_some_and(|c| c.messages().len() == 4)
    });
    assert!(done, "{:?}", conversation(&app, agent));
    let seen = &app.world().resource::<GatedFootprints>().0;
    let at_gate = |id: &str| {
        seen.iter()
            .find(|(call, _)| call == id)
            .map(|(_, touches)| touches.clone())
    };
    let etc = Scope::Paths(vec![PathBuf::from("/etc/passwd")]);
    assert_eq!(at_gate("climbs"), Some(Touches::Writes(etc)), "{seen:?}");
    assert_eq!(
        at_gate("relative"),
        Some(Touches::Writes(Scope::Anything)),
        "{seen:?}"
    );
}

/// A tool registered while the app runs, as an MCP server's is, that
/// says it only reads is taken at its word for running beside readers, not
/// for running again after a restart; a local declaration can trust it.
#[test]
fn a_remote_read_only_hint_is_trusted_only_when_declared() {
    let mut app = app();
    let read_only = ToolAnnotations::default().with_read_only(true);
    let trusted = Some(Touches::Reads(Scope::Anything));
    for (name, footprint) in [("hinted", None), ("trusted", trusted.clone())] {
        let Ok(name) = ToolName::new(name) else {
            return;
        };
        let schema = serde_json::json!({ "type": "object" });
        let tool = DynamicTool::new(name, "Looks.", schema, |_| {
            Box::pin(async { Ok(ToolOutput::text("seen")) })
        });
        let tool = tool.with_annotations(read_only.clone());
        let (definition, handler, _) = tool.into_parts();
        let options = ToolOptions { footprint };
        app.world_mut().spawn_tool(definition, handler, options);
    }
    let model = MockCompletionModel::from_stream_turns([
        reply(&[("hinted", "h1"), ("trusted", "t1")]),
        answer("done"),
    ]);
    app.init_resource::<GatedFootprints>()
        .add_systems(AgentPass, log_footprints.in_set(Stage::Gate));
    let agent = agent_on(&mut app, &model);
    let done = run_until(&mut app, |world| {
        world
            .get::<Conversation>(agent)
            .is_some_and(|c| c.messages().len() == 4)
    });
    assert!(done, "{:?}", conversation(&app, agent));
    let seen = &app.world().resource::<GatedFootprints>().0;
    let at_gate = |id: &str| {
        seen.iter()
            .find(|(call, _)| call == id)
            .map(|(_, touches)| touches.clone())
    };
    assert_eq!(at_gate("h1"), Some(Touches::Independent), "{seen:?}");
    assert_eq!(at_gate("t1"), trusted, "{seen:?}");
}

#[test]
fn a_refused_call_answers_in_call_order_and_is_recorded_as_refused() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([
        reply(&[("touch", "refused"), ("look", "l2")]),
        answer("done"),
    ]);
    app.init_resource::<Gated>()
        .add_systems(AgentPass, gate.in_set(Stage::Gate));
    let agent = agent_on(&mut app, &model);
    let done = run_until(&mut app, |world| {
        world
            .get::<Conversation>(agent)
            .map_or(0, |c| c.messages().len())
            == 4
    });
    assert!(done, "{:?}", conversation(&app, agent));
    // One message answers both calls, in the reply's order.
    let results = results_at(&conversation(&app, agent), 2);
    assert!(
        matches!(results.as_slice(), [(true, refused), (false, ran)]
            if refused.contains("not this one") && ran.contains("seen")),
        "{results:?}"
    );
    let records = records(&app);
    let record = |name: &str| {
        records
            .iter()
            .find(|record| matches!(&record.kind, EffectKind::ToolCall { name: called, .. } if called == name))
    };
    let refused = record("touch");
    assert!(
        refused.is_some_and(
            |record| record.disposition == Disposition::AnsweredBeforeStart
                && record.started_ms.is_none()
                && matches!(&record.outcome, Err(report) if report.kind == ErrorKind::Denied)
        ),
        "{refused:?}"
    );
    let ran = record("look");
    assert!(
        ran.is_some_and(|record| record.disposition == Disposition::Ran && record.outcome.is_ok()),
        "{ran:?}"
    );
}

/// How each turn ended, in order.
#[derive(Resource, Default)]
struct Ended(Vec<TurnOutcome>);

fn log_ended(settled: On<Add<Settled>>, turns: Query<&TurnOutcome>, mut log: ResMut<Ended>) {
    log.0.extend(turns.get(settled.entity).cloned());
}

fn ended(app: &App) -> Vec<TurnOutcome> {
    app.world().resource::<Ended>().0.clone()
}

/// `model` as the catalog's DeepSeek model.
fn connection(model: &MockCompletionModel) -> Option<Connection> {
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .ok()?
        .shared();
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    Some(Connection {
        spec,
        handler: Handler(ErasedHandler::new(handler)),
    })
}

/// An overloaded provider's reply, worth retrying.
fn overloaded() -> Vec<MockStreamEvent> {
    let overloaded = ProviderResponseError::without_status("overloaded").with_transient(Some(true));
    vec![MockStreamEvent::Error(MockError::ProviderResponse(
        overloaded,
    ))]
}

/// The records of the model calls, in settle order.
fn model_records(app: &App) -> Vec<EffectRecord> {
    let records = records(app).into_iter();
    let models = records.filter(|record| matches!(record.kind, EffectKind::Completion { .. }));
    models.collect()
}

/// Supersedes each turn call whose retries gave up with the same request
/// on `to`.
fn fall_back(
    to: Option<Connection>,
) -> impl FnMut(
    Query<
        (
            Entity,
            &RetryVerdict,
            &ModelRequest,
            &Effect,
            &CallOf,
            &ForAgent,
        ),
        (ReadyToSettle, With<TurnCall>, Without<SupersededBy>),
    >,
    Commands,
) + Send
+ Sync
+ 'static {
    move |failed, mut commands| {
        for (call, verdict, request, effect, of, agent) in &failed {
            let (Verdict::GaveUp(_), Some(to)) = (&verdict.0, to.clone()) else {
                continue;
            };
            commands
                .spawn((
                    request.clone().on(to),
                    TurnCall,
                    CallOf(of.0),
                    ForAgent(agent.0),
                    Effect::with_parent_id(effect.parent),
                ))
                .add_related::<SupersededBy>(&[call]);
        }
    }
}

#[test]
fn a_call_that_gave_up_is_superseded_on_another_model_and_the_turn_goes_on() {
    let mut app = app();
    let failing = MockCompletionModel::from_stream_turns([overloaded()]);
    let fallback = MockCompletionModel::from_stream_turns([answer("ok")]);
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        // The retries ran out already: the first failure gives up.
        .add_observer(|turn: On<Add<TurnOf>>, mut commands: Commands| {
            let retries = RETRY_POLICY.max_retries;
            commands.entity(turn.entity).insert(Recovery { retries });
        })
        .add_systems(
            AgentPass,
            fall_back(connection(&fallback)).in_set(Stage::Review),
        );
    let agent = agent_on(&mut app, &failing);
    let done = run_until(&mut app, |world| !world.resource::<Ended>().0.is_empty());
    assert!(done, "{:?}", notices(&app));
    let answer = match ended(&app).as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("ok"), "{:?}", ended(&app));
    assert_eq!((failing.request_count(), fallback.request_count()), (1, 1));
    // The failed call never reached the conversation.
    assert_eq!(conversation(&app, agent).len(), 2);
    let models: Vec<_> = model_records(&app)
        .iter()
        .map(|record| (record.disposition, record.outcome.is_ok()))
        .collect();
    assert_eq!(
        models,
        [(Disposition::Superseded, false), (Disposition::Ran, true)]
    );
}

#[test]
fn a_retry_waits_on_a_hold_and_an_interrupt_during_the_wait_sends_nothing() {
    let mut app = app();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    let model = MockCompletionModel::from_stream_turns([overloaded(), answer("late")]);
    let agent = agent_on(&mut app, &model);
    let waits = |world: &mut World| world.query::<&Backoff>().iter(world).count();
    assert!(run_until(&mut app, |world| waits(world) == 1));
    let world = app.world_mut();
    let mut held = world.query_filtered::<(), (With<TurnCall>, With<Holds>, Without<Settled>)>();
    assert_eq!(held.iter(world).count(), 1, "the failed call waits, held");
    world.trigger(StopTurn::new(agent));
    world.flush();
    assert_eq!(waits(world), 0, "the wait went with the call");
    // Long past the wait, nothing was sent again.
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs(120)));
    app.update();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(model.request_count(), 1);
    let world = app.world_mut();
    assert_eq!(world.query::<&TurnCall>().iter(world).count(), 0);
}

/// Whether [`refuse_when_asked`] refuses the turns' model calls.
#[derive(Resource, Default)]
struct Refusing(bool);

fn refuse_when_asked(
    refusing: Res<Refusing>,
    mut calls: Query<&mut Decision, (ReadyToStart, ModelCalls, With<TurnCall>)>,
) {
    for mut decision in calls.iter_mut().filter(|_| refusing.0) {
        decision.decide(Decider::plugin("test"), Outcome::refused("not now"));
    }
}

/// Each turn is recorded in its agent's conversation with how it ended:
/// an answer by its entry, a failure with why, the request it left halted;
/// a note to an idle agent starts no turn and no record.
#[test]
fn each_turn_is_recorded_with_how_it_ended() {
    let mut app = app();
    app.init_resource::<Refusing>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, refuse_when_asked.in_set(Stage::Gate));
    let model = MockCompletionModel::from_stream_turns([answer("hello")]);
    let agent = agent_on(&mut app, &model);
    let ended_once = |world: &mut World| world.resource::<Ended>().0.len() == 1;
    assert!(run_until(&mut app, ended_once));
    let world = app.world_mut();
    let note =
        Deliver::new(agent, "fyi", DeliveryMode::Inform).with_author(Author::plugin("check"));
    world.trigger(note);
    world.resource_mut::<Refusing>().0 = true;
    world.trigger(Deliver::new(agent, "and?", DeliveryMode::Steer));
    let ended_twice = |world: &mut World| world.resource::<Ended>().0.len() == 2;
    assert!(run_until(&mut app, ended_twice));
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return;
    };
    let hello = conversation
        .entries()
        .find(|entry| entry.message.final_answer().as_deref() == Some("hello"));
    let records: Vec<_> = conversation
        .turns()
        .map(|(_, record)| (record.index, record.end))
        .collect();
    let failed = TurnEnd::Failed {
        why: "not now".to_owned(),
    };
    let answered = TurnEnd::Answered {
        answer: hello.map(|entry| entry.id),
    };
    assert_eq!(records, [(0, Some(answered)), (1, Some(failed))]);
    assert_eq!(conversation.halted(), Some(Halt::Kept));
}

#[test]
fn a_refused_turn_call_ends_the_turn_gives_the_message_back_and_is_not_retried() {
    let mut app = app();
    let refuse = |mut calls: Query<&mut Decision, (ReadyToStart, ModelCalls, With<TurnCall>)>| {
        for mut decision in &mut calls {
            decision.decide(Decider::plugin("test"), Outcome::refused("not now"));
        }
    };
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, refuse.in_set(Stage::Gate));
    let model = MockCompletionModel::from_stream_turns([answer("never")]);
    let agent = agent_on(&mut app, &model);
    let done = run_until(&mut app, |world| !world.resource::<Ended>().0.is_empty());
    assert!(done);
    for _ in 0..3 {
        app.update();
    }
    assert!(
        matches!(ended(&app).as_slice(), [TurnOutcome::Failed(why)] if why == "not now"),
        "{:?}",
        ended(&app)
    );
    assert_eq!(model.request_count(), 0);
    assert!(conversation(&app, agent).is_empty(), "the message is back");
    let models: Vec<_> = model_records(&app)
        .iter()
        .map(|record| (record.disposition, record.started_ms.is_none()))
        .collect();
    assert_eq!(models, [(Disposition::AnsweredBeforeStart, true)]);
    let world = app.world_mut();
    assert_eq!(world.query::<&Backoff>().iter(world).count(), 0);
}

/// A request that outgrew the model's context window ends its turn at
/// once: the same request would fail the same way, so it is not retried.
/// The user's message goes back to the input, out of the conversation.
#[test]
fn a_request_too_big_for_the_context_window_fails_its_turn_without_a_retry() {
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    let too_big = ProviderResponseError::without_status("prompt is too long: 1200000 tokens")
        .with_transient(Some(false));
    let too_big = vec![MockStreamEvent::Error(MockError::ProviderResponse(too_big))];
    let model = MockCompletionModel::from_stream_turns([too_big, answer("never")]);
    let agent = agent_on(&mut app, &model);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    for _ in 0..3 {
        app.update();
    }
    assert!(
        matches!(ended(&app).as_slice(), [TurnOutcome::Failed(why)]
            if why.contains("does not fit the model's context window")),
        "{:?}",
        ended(&app)
    );
    assert_eq!(model.request_count(), 1, "not retried");
    assert_eq!(conversation(&app, agent), Vec::<Message>::new());
    let world = app.world_mut();
    assert_eq!(world.query::<&Backoff>().iter(world).count(), 0);
}

/// A model call that failed for good or ran out of retries, or a reply
/// cut off before it answered or with no message, takes the user's message
/// out of the conversation, with the failed reply, and gives it back after
/// an error that says why; sent again, it goes once.
#[test]
fn a_failed_turn_gives_the_message_back_and_sent_again_it_goes_once() {
    let empty = vec![MockStreamEvent::final_response_with_default_usage()];
    for (failed, out_of_retries) in [
        (no_credit(), false),
        (cut_off(), false),
        (empty, false),
        (overloaded(), true),
    ] {
        send_again_after(failed, out_of_retries);
    }
}

fn send_again_after(failed: Vec<MockStreamEvent>, out_of_retries: bool) {
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    collect_recalled(&mut app);
    if out_of_retries {
        // The retries ran out already: a failure gives up at once.
        app.add_observer(|turn: On<Add<TurnOf>>, mut commands: Commands| {
            let retries = RETRY_POLICY.max_retries;
            commands.entity(turn.entity).insert(Recovery { retries });
        });
    }
    let model = MockCompletionModel::from_stream_turns([failed, answer("ok")]);
    let agent = agent_on(&mut app, &model);
    let ended_n = |n| move |world: &mut World| world.resource::<Ended>().0.len() == n;
    assert!(run_until(&mut app, ended_n(1)), "{:?}", notices(&app));
    app.update();
    assert!(matches!(ended(&app).as_slice(), [TurnOutcome::Failed(_)]));
    assert_eq!(
        conversation(&app, agent),
        [],
        "the message and reply are out"
    );
    assert_eq!(app.world().resource::<RecalledTexts>().0, ["go"]);
    let told = notices(&app);
    assert!(
        !told.iter().any(|notice| notice.contains("taken out")),
        "it came back: {told:?}"
    );
    app.world_mut()
        .trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    assert!(run_until(&mut app, ended_n(2)), "{:?}", notices(&app));
    let answer = match ended(&app).as_slice() {
        [_, TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("ok"), "{:?}", ended(&app));
    let sent = model.requests();
    let history = sent.last().map(|request| request.chat_history.as_slice());
    let history: Vec<&Message> = history.into_iter().flatten().map(AsRef::as_ref).collect();
    assert_eq!(history, [&Message::user("go")], "the message went once");
}

/// A failed turn gives back, in one recall, the user's messages no model
/// answered: the steering message its last call carried, then a queued one
/// it never sent. The message a tool-calling reply answered stays, and so
/// does a chat bridge user's queued message, which never reaches the local
/// input.
#[test]
fn a_failed_turn_gives_back_its_unanswered_and_unsent_messages_as_one() {
    let mut app = app();
    app.init_resource::<Ended>()
        .init_resource::<Reviewing>()
        .add_observer(log_ended)
        .add_systems(AgentPass, hold_results.in_set(Stage::Review));
    collect_recalled(&mut app);
    let model = MockCompletionModel::from_stream_turns([reply(&[("look", "1")]), no_credit()]);
    let agent = agent_on(&mut app, &model);
    let held = |world: &mut World| !world.resource::<Reviewing>().0.is_empty();
    assert!(run_until(&mut app, held));
    let world = app.world_mut();
    world.trigger(Deliver::new(agent, "then commit", DeliveryMode::Queue));
    world.trigger(Deliver::new(agent, "use tabs", DeliveryMode::Steer));
    // A chat bridge's user, queued too, is not the local user.
    let remote = Incoming::new("ship it", DeliveryMode::Queue);
    let message = remote.with_author(Author::user().titled("ana"));
    world.trigger(Deliver {
        entity: agent,
        message,
    });
    app.update();
    let world = app.world_mut();
    for hold in std::mem::take(&mut world.resource_mut::<Reviewing>().0) {
        world.despawn(hold);
    }
    let ended_once = |world: &mut World| world.resource::<Ended>().0.len() == 1;
    assert!(run_until(&mut app, ended_once), "{:?}", notices(&app));
    app.update();
    assert_eq!(model.request_count(), 2);
    let sent = serde_json::to_string(&model.requests()).unwrap_or_default();
    assert!(
        sent.contains("use tabs") && !sent.contains("then commit"),
        "{sent}"
    );
    let recalled = app.world().resource::<RecalledTexts>();
    assert_eq!(recalled.0, ["use tabs\n\nthen commit"]);
    let said = conversation(&app, agent);
    assert_eq!(said.len(), 4, "{said:?}");
    assert_eq!(said.first(), Some(&Message::user("go")));
    let last = said.last().map(serde_json::to_string);
    assert!(
        matches!(&last, Some(Ok(text)) if text.contains("ship it")),
        "{last:?}"
    );
}

/// A request the user stopped stays with its note when a later message
/// fails: only the later one comes back.
#[test]
fn a_failed_turn_leaves_a_stopped_request_with_its_note() {
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    collect_recalled(&mut app);
    // The first model call waits in Gate, so the stop always comes first.
    let hold_first = |calls: Query<Entity, (ReadyToStart, With<TurnCall>)>,
                      mut held: Local<bool>,
                      mut commands: Commands| {
        if let Some(call) = calls.iter().next().filter(|_| !*held) {
            commands.spawn(HoldOf(call));
            *held = true;
        }
    };
    app.add_systems(AgentPass, hold_first.in_set(Stage::Gate));
    let model = MockCompletionModel::from_stream_turns([no_credit(), no_credit()]);
    let agent = agent_on(&mut app, &model);
    let running = |world: &mut World| world.get::<ActiveTurn>(agent).is_some();
    assert!(run_until(&mut app, running));
    app.world_mut().trigger(StopTurn::new(agent));
    let ended_n = |n| move |world: &mut World| world.resource::<Ended>().0.len() == n;
    assert!(run_until(&mut app, ended_n(1)), "{:?}", notices(&app));
    app.world_mut()
        .trigger(Deliver::new(agent, "and lint", DeliveryMode::Steer));
    assert!(run_until(&mut app, ended_n(2)), "{:?}", notices(&app));
    app.update();
    assert_eq!(app.world().resource::<RecalledTexts>().0, ["and lint"]);
    let said = conversation(&app, agent);
    assert_eq!(said, [Message::user("go"), Message::user(STOPPED)]);
}

/// A reply that comes in after a plugin failed its turn goes into the
/// conversation, but none of its tool calls runs: each is answered as not
/// run, so every call in the history has a result.
#[test]
fn a_reply_after_its_turn_failed_is_kept_and_its_calls_are_answered_not_run() {
    let mut app = app();
    let budget = |calls: Query<&CallOf, (ReadyToSettle, With<TurnCall>)>,
                  mut commands: Commands| {
        for &CallOf(turn) in &calls {
            let over = TurnOutcome::Failed("over budget".to_owned());
            commands.entity(turn).insert(over);
        }
    };
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, budget.in_set(Stage::Review));
    let model = MockCompletionModel::from_stream_turns([reply(&[("touch", "t1")]), answer("no")]);
    let agent = agent_on(&mut app, &model);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    for _ in 0..3 {
        app.update();
    }
    assert!(
        matches!(ended(&app).as_slice(), [TurnOutcome::Failed(why)] if why == "over budget"),
        "{:?}",
        ended(&app)
    );
    assert_eq!(model.request_count(), 1);
    let said = conversation(&app, agent);
    assert_eq!(said.len(), 3, "{said:?}");
    let results = results_at(&said, 2);
    assert!(
        matches!(results.as_slice(), [(true, text)] if text.contains("not run")),
        "{results:?}"
    );
    let tools = records(&app).into_iter();
    let ran = tools.filter(|record| matches!(record.kind, EffectKind::ToolCall { .. }));
    assert_eq!(ran.count(), 0, "no call started");
}

/// A failed reply a conversation already holds, such as one restored from
/// an older session, goes with the messages a failed turn gives back: one
/// the provider failed, and one cut off at the token limit before it
/// answered. What comes back is the text typed, not a file attached.
#[test]
fn a_failed_turn_takes_a_stored_failed_reply_with_it() {
    use rig_core::message::{AssistantContent, AssistantMessage};
    let error = AssistantMessage::new(vec![AssistantContent::text("half an answer")])
        .with_stop(StopReason::Error("overloaded".to_owned()));
    let cut = AssistantMessage::new(vec![AssistantContent::reasoning("thinking")])
        .with_stop(StopReason::Length);
    for failed in [error, cut] {
        takes_a_stored_failed_reply(failed);
    }
}

fn takes_a_stored_failed_reply(failed: rig_core::message::AssistantMessage) {
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    collect_recalled(&mut app);
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    if let Some(mut conversation) = world.get_mut::<Conversation>(agent) {
        conversation.push(Message::user("hi"));
        conversation.push(Message::assistant("hello"));
        // A file attached to it does not come back as typed text.
        let file = UserContent::text("<file path=\"a.rs\">\n1  fn a() {}\n</file>");
        let content = vec![UserContent::text("and then?"), file];
        conversation.push(Message::User { content });
        conversation.push(Message::Assistant(failed));
    }
    let model = MockCompletionModel::from_stream_turns([no_credit()]);
    agent_on(&mut app, &model);
    let ended = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended), "{:?}", notices(&app));
    app.update();
    let recalled = &app.world().resource::<RecalledTexts>().0;
    assert_eq!(recalled, &["and then?\n\ngo"]);
    let said = conversation(&app, agent);
    assert_eq!(said, [Message::user("hi"), Message::assistant("hello")]);
}

/// The results the last reply that counts asked for stay with it, though
/// a system note came between, as a restored session may hold them; only
/// the user's later message comes back.
#[test]
fn a_failed_turn_keeps_the_results_of_the_reply_before_it() {
    use rig_core::message::{AssistantContent, AssistantMessage, ToolCall};
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    collect_recalled(&mut app);
    let Ok(name) = ToolName::new("look") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name.clone(), serde_json::json!({})));
    let asked = Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        call.clone(),
    )]));
    let held = [
        Message::user("read it"),
        asked,
        Message::system("a note"),
        Message::tool_result(call.id, name, "the file"),
    ];
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    if let Some(mut conversation) = world.get_mut::<Conversation>(agent) {
        for message in held.iter().cloned() {
            conversation.push(message);
        }
        conversation.push(Message::user("now explain it"));
    }
    let model = MockCompletionModel::from_stream_turns([no_credit()]);
    agent_on(&mut app, &model);
    let ended = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended), "{:?}", notices(&app));
    app.update();
    let recalled = &app.world().resource::<RecalledTexts>().0;
    assert_eq!(recalled, &["now explain it\n\ngo"]);
    assert_eq!(conversation(&app, agent), held);
}

/// A failed reply stored after the results of the last reply that counts
/// goes when a call that carries on from those results fails, though no
/// message of the user's comes back.
#[test]
fn a_failed_turn_takes_a_stored_failed_reply_after_results() {
    use rig_core::message::{AssistantContent, AssistantMessage, ToolCall};
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    collect_recalled(&mut app);
    let Ok(name) = ToolName::new("look") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name.clone(), serde_json::json!({})));
    let asked = Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
        call.clone(),
    )]));
    let held = [
        Message::user("read it"),
        asked,
        Message::tool_result(call.id, name, "the file"),
    ];
    let failed = AssistantMessage::new(vec![AssistantContent::text("half")])
        .with_stop(StopReason::Error("overloaded".to_owned()));
    let model = MockCompletionModel::from_stream_turns([no_credit()]);
    let agent = connected(&mut app, &model);
    let world = app.world_mut();
    if let Some(mut conversation) = world.get_mut::<Conversation>(agent) {
        for message in held.iter().cloned() {
            conversation.push(message);
        }
        conversation.push(Message::Assistant(failed));
    }
    // A call that carries on from the results, as a restart's does.
    let turn = world.spawn(TurnOf(agent)).id();
    world.spawn((TurnCall, CallOf(turn), ForAgent(agent)));
    let ended = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended), "{:?}", notices(&app));
    app.update();
    assert!(app.world().resource::<RecalledTexts>().0.is_empty());
    assert_eq!(conversation(&app, agent), held);
}

/// A titled user's message, such as a chat bridge's or a skill's, is not
/// the local user's own: a failed turn takes it out of the conversation,
/// as every user's message no model answered, but hands nothing back to
/// the local input, and says it was taken out.
#[test]
fn a_failed_turn_takes_a_titled_users_message_without_giving_it_back() {
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    collect_recalled(&mut app);
    let model = MockCompletionModel::from_stream_turns([answer("hi"), no_credit()]);
    let agent = agent_on(&mut app, &model);
    let ended_n = |n| move |world: &mut World| world.resource::<Ended>().0.len() == n;
    assert!(run_until(&mut app, ended_n(1)), "{:?}", notices(&app));
    let remote =
        Incoming::new("deploy it", DeliveryMode::Steer).with_author(Author::user().titled("ana"));
    app.world_mut().trigger(Deliver {
        entity: agent,
        message: remote,
    });
    assert!(run_until(&mut app, ended_n(2)), "{:?}", notices(&app));
    app.update();
    assert!(app.world().resource::<RecalledTexts>().0.is_empty());
    assert_eq!(conversation(&app, agent).len(), 2, "only go and hi stay");
    let told = notices(&app);
    assert!(
        told.iter().any(|notice| notice.contains("taken out")),
        "{told:?}"
    );
}

/// An image sent to a model that does not read images goes as its name,
/// and the user is told; the model call still goes.
#[test]
fn an_image_to_a_model_that_reads_none_goes_as_its_name() {
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    let model = MockCompletionModel::from_stream_turns([answer("a cat, I guess")]);
    let spec = Catalog::builtin().resolve("deepseek/deepseek-flash");
    let Ok(mut spec) = spec.map(|found| found.spec.clone()) else {
        return;
    };
    spec.input.image = false;
    let spec = Arc::new(spec);
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    let connection = Connection {
        spec,
        handler: Handler(ErasedHandler::new(handler)),
    };
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    world.entity_mut(agent).insert(connection);
    let image = UserContent::image_base64("iVBORw0KGgo=", None, None);
    let mut message = Incoming::new("", DeliveryMode::Steer);
    message.parts = vec![MessagePart::Attachment(Attachment {
        label: "cat.png".to_owned(),
        content: image,
    })];
    world.trigger(Deliver {
        entity: agent,
        message,
    });
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    let sent = model.requests();
    let last = sent.first().and_then(|request| request.chat_history.last());
    let content = match last.map(|message| &**message) {
        Some(Message::User { content }) => content.to_vec(),
        _ => Vec::new(),
    };
    assert_eq!(content, [UserContent::text("cat.png")]);
    let told = notices(&app);
    assert!(
        told.iter()
            .any(|notice| notice
                .ends_with("does not read images, so cat.png is sent as its name only.")),
        "{told:?}"
    );
}

/// Says it exited cleanly; only reads, as its annotations say.
struct Probe;

impl Tool for Probe {
    const NAME: &'static str = "probe";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Probes.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default().with_read_only(true)
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        context.insert_result(ExitStatus::Exited(0))?;
        Ok("probed".to_owned())
    }
}

/// What settled tool calls published.
#[derive(Resource, Default)]
struct Facts(Vec<ExitStatus>);

#[test]
fn a_tools_annotations_give_its_footprint_and_what_it_publishes_lands_on_its_call() {
    let mut app = app();
    app.add_tool(Probe).init_resource::<Facts>().add_observer(
        |settled: On<Add<Settled>>, calls: Query<&Published>, mut facts: ResMut<Facts>| {
            let published = calls.get(settled.entity).ok();
            let status = published.and_then(|published| published.0.get::<ExitStatus>().ok());
            facts.0.extend(status.flatten());
        },
    );
    let model = MockCompletionModel::from_stream_turns([reply(&[("probe", "p1")]), answer("done")]);
    let agent = agent_on(&mut app, &model);
    let done = run_until(&mut app, |world| {
        world
            .get::<Conversation>(agent)
            .is_some_and(|c| c.messages().len() == 4)
    });
    assert!(done, "{:?}", conversation(&app, agent));
    let world = app.world_mut();
    let mut tools = world.query::<(&ToolDef, &Footprint)>();
    let footprint = tools
        .iter(world)
        .find(|(def, _)| def.0.name.as_str() == "probe")
        .map(|(_, footprint)| footprint.0.clone());
    assert_eq!(footprint, Some(Touches::Reads(Scope::Anything)));
    assert_eq!(app.world().resource::<Facts>().0, [ExitStatus::Exited(0)]);
    let probe = records(&app).into_iter().find(
        |record| matches!(&record.kind, EffectKind::ToolCall { name, .. } if name == "probe"),
    );
    let recorded = probe
        .and_then(|record| record.published)
        .and_then(|output| output.get::<ExitStatus>().ok().flatten());
    assert_eq!(recorded, Some(ExitStatus::Exited(0)));
}

/// The answers of side calls, as they settled.
#[derive(Resource, Default)]
struct SideAnswers(Vec<String>);

#[test]
fn a_side_call_is_built_from_its_agent_and_leaves_the_conversation_alone() {
    let mut app = app();
    let ask = |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<SideCall>)>| {
        for mut call in &mut calls {
            let question = Entry::new(Message::user("And in one word?"));
            if let Some(history) = &mut call.history {
                history.push(question.by(Author::plugin("side")));
            }
        }
    };
    app.init_resource::<SideAnswers>()
        .add_systems(AgentPass, ask.in_set(Stage::Prepare))
        .add_observer(
            |settled: On<Add<Settled>>,
             calls: Query<&Outcome, With<SideCall>>,
             mut answers: ResMut<SideAnswers>| {
                let reply = calls.get(settled.entity).ok().and_then(Outcome::completion);
                let text = reply.and_then(|reply| reply.message()?.final_answer());
                answers.0.extend(text);
            },
        );
    let model = MockCompletionModel::from_stream_turns([answer("hello there"), answer("hi")]);
    let agent = agent_on(&mut app, &model);
    let said = |world: &mut World| world.get::<Conversation>(agent).map(|c| c.messages().len());
    assert!(run_until(&mut app, |world| said(world) == Some(2)));
    app.world_mut().spawn(SideCall(agent));
    let answered = run_until(&mut app, |world| {
        !world.resource::<SideAnswers>().0.is_empty()
    });
    assert!(answered, "{:?}", notices(&app));
    assert_eq!(app.world().resource::<SideAnswers>().0, ["hi"]);
    assert_eq!(
        conversation(&app, agent).len(),
        2,
        "the conversation is left alone"
    );
    let requests = model.requests();
    let (Some(turn), Some(side)) = (requests.first(), requests.get(1)) else {
        return;
    };
    assert_eq!(side.system_instructions(), turn.system_instructions());
    assert_eq!(side.tools.len(), turn.tools.len());
    // The agent has no system prompt here: the conversation and the
    // question.
    assert_eq!(side.chat_history.len(), 3, "{:?}", side.chat_history);
}

/// How each turn that ended was numbered: its index and first message.
#[derive(Resource, Default)]
struct Numbered(Vec<(u64, usize)>);

/// Carries the first turn that would end with an answer on, with a
/// `Queue` nudge from a plugin.
fn nudge(
    turns: Query<(&TurnOf, &TurnOutcome), Added<TurnOutcome>>,
    mut nudged: Local<bool>,
    mut commands: Commands,
) {
    for (&TurnOf(agent), outcome) in &turns {
        if matches!(outcome, TurnOutcome::Answered(_)) && !*nudged {
            *nudged = true;
            let nudge = Deliver::new(agent, "And the tests?", DeliveryMode::Queue);
            commands.trigger(nudge.with_author(Author::plugin("nudge")));
        }
    }
}

#[test]
fn a_reviewer_carries_a_turn_on_with_a_queued_nudge() {
    let mut app = app();
    app.init_resource::<Ended>()
        .init_resource::<Numbered>()
        .add_observer(log_ended)
        .add_observer(
            |settled: On<Add<Settled>>,
             turns: Query<(&Turn, &ForAgent)>,
             agents: Query<&Conversation>,
             mut log: ResMut<Numbered>| {
                let Ok((turn, agent)) = turns.get(settled.entity) else {
                    return;
                };
                let conversation = agents.get(agent.0).ok();
                let first = conversation.map(|conversation| conversation.start_of(turn.start));
                log.0.extend(first.map(|first| (turn.index, first)));
            },
        )
        .add_systems(AgentPass, nudge.in_set(Stage::Review));
    let model = MockCompletionModel::from_stream_turns([
        answer("first"),
        answer("second"),
        answer("third"),
    ]);
    let agent = agent_on(&mut app, &model);
    let ended_once = |world: &mut World| world.resource::<Ended>().0.len() == 1;
    assert!(run_until(&mut app, ended_once), "{:?}", notices(&app));
    // One turn: the nudge carried it on instead of ending it.
    let answer = match ended(&app).as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("second"), "{:?}", ended(&app));
    assert_eq!(model.request_count(), 2);
    let second = model.requests().get(1).map(serde_json::to_string);
    assert!(
        matches!(&second, Some(Ok(sent)) if sent.contains("And the tests?")),
        "{second:?}"
    );
    assert!(!has::<ActiveTurn>(&app, agent));
    // The next turn begins after the four messages of the first.
    app.world_mut()
        .trigger(Deliver::new(agent, "again", DeliveryMode::Steer));
    let ended_twice = |world: &mut World| world.resource::<Ended>().0.len() == 2;
    assert!(run_until(&mut app, ended_twice), "{:?}", notices(&app));
    assert_eq!(app.world().resource::<Numbered>().0, [(0, 0), (1, 4)]);
}

/// The hold a reviewer put on a turn's proposed end.
#[derive(Resource, Default)]
struct Judging(Option<(Entity, Entity)>);

/// Holds each turn's proposed answer while it is judged.
fn hold_answers(
    turns: Query<(Entity, &TurnOutcome), Added<TurnOutcome>>,
    mut judging: ResMut<Judging>,
    mut commands: Commands,
) {
    for (turn, outcome) in &turns {
        if matches!(outcome, TurnOutcome::Answered(_)) {
            let hold = commands.spawn((Name::new("judging"), HoldOf(turn))).id();
            judging.0 = Some((turn, hold));
        }
    }
}

#[test]
fn a_held_proposal_keeps_the_turn_until_a_reviewer_replaces_it() {
    let mut app = app();
    app.init_resource::<Ended>()
        .init_resource::<Judging>()
        .add_observer(log_ended)
        .add_systems(AgentPass, hold_answers.in_set(Stage::Review));
    let model = MockCompletionModel::from_stream_turns([answer("good enough?")]);
    let agent = agent_on(&mut app, &model);
    let held = |world: &mut World| world.resource::<Judging>().0.is_some();
    assert!(run_until(&mut app, held));
    for _ in 0..5 {
        app.update();
    }
    // Every call settled and went; the held proposal keeps the turn.
    assert!(has::<ActiveTurn>(&app, agent));
    assert!(ended(&app).is_empty());
    let world = app.world_mut();
    assert_eq!(world.query::<&Effect>().iter(world).count(), 0);
    let Some((turn, hold)) = world.resource::<Judging>().0 else {
        return;
    };
    let failed = TurnOutcome::Failed("not good enough".to_owned());
    world.entity_mut(turn).insert(failed);
    world.despawn(hold);
    app.update();
    assert!(
        matches!(ended(&app).as_slice(), [TurnOutcome::Failed(why)] if why == "not good enough"),
        "{:?}",
        ended(&app)
    );
    assert!(!has::<ActiveTurn>(&app, agent));
}

#[test]
fn a_hold_on_a_proposed_end_goes_when_the_turn_carries_on() {
    let mut app = app();
    app.init_resource::<Ended>()
        .init_resource::<Judging>()
        .add_observer(log_ended)
        .add_systems(AgentPass, hold_answers.in_set(Stage::Review));
    let model = MockCompletionModel::from_stream_turns([
        answer("first"),
        answer("second"),
        answer("third"),
    ]);
    let agent = agent_on(&mut app, &model);
    let judged = |app: &App| app.world().resource::<Judging>().0;
    let held = |world: &mut World| world.resource::<Judging>().0.is_some();
    assert!(run_until(&mut app, held));
    let Some((turn, first)) = judged(&app) else {
        return;
    };
    // A message the agent accepts carries the turn on: the hold on the
    // answer it took away goes with it.
    let more = Deliver::new(agent, "more", DeliveryMode::Steer);
    app.world_mut().trigger(more);
    let asked = |count| {
        let model = model.clone();
        move |_: &mut World| model.request_count() == count
    };
    assert!(run_until(&mut app, asked(2)));
    assert!(app.world().get_entity(first).is_err(), "the first hold");
    let held_again = |world: &mut World| {
        world
            .resource::<Judging>()
            .0
            .is_some_and(|(_, hold)| hold != first)
    };
    assert!(run_until(&mut app, held_again));
    let Some((_, second)) = judged(&app) else {
        return;
    };
    // So does a new turn call.
    let call = (TurnCall, CallOf(turn), ForAgent(agent));
    app.world_mut().spawn(call);
    assert!(run_until(&mut app, asked(3)));
    assert!(app.world().get_entity(second).is_err(), "the second hold");
    let held_last = |world: &mut World| {
        world
            .resource::<Judging>()
            .0
            .is_some_and(|(_, hold)| hold != second)
    };
    assert!(run_until(&mut app, held_last));
    // Releasing the one hold left on the last answer ends the turn.
    if let Some((_, last)) = judged(&app) {
        app.world_mut().despawn(last);
    }
    let ended_once = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended_once));
    let answer = match ended(&app).as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("third"), "{:?}", ended(&app));
}

#[test]
fn a_messages_parts_go_to_the_model_in_its_authors_order() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([answer("a cat")]);
    let world = app.world_mut();
    let mut agents = world.query_filtered::<Entity, With<Agent>>();
    let agent = agents.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
    let spec = Catalog::builtin().resolve("anthropic/claude-fable-5");
    let Ok(spec) = spec.map(|spec| spec.shared()) else {
        return;
    };
    assert!(spec.input.image);
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    let handler = Handler(ErasedHandler::new(handler));
    world.entity_mut(agent).insert(Connection { spec, handler });
    let image = UserContent::image_base64("iVBORw0KGgo=", None, None);
    let mut message =
        Incoming::new("look", DeliveryMode::Steer).with_author(Author::plugin("eyes"));
    let picture = Attachment {
        label: "cat.png".to_owned(),
        content: image.clone(),
    };
    message.parts.push(MessagePart::Attachment(picture));
    world.spawn((message, ForAgent(agent)));
    let done = run_until(&mut app, |world| {
        world
            .get::<Conversation>(agent)
            .is_some_and(|c| c.messages().len() == 2)
    });
    assert!(done, "{:?}", notices(&app));
    let conversation = app.world().get::<Conversation>(agent);
    let first = conversation.and_then(|c| c.entries().next());
    let header = Author::plugin("eyes").header().unwrap_or_default();
    let said = UserContent::text(format!("{header}\nlook"));
    assert!(
        matches!(first, Some(entry)
            if entry.author == Some(&Author::plugin("eyes"))
                && matches!(entry.message, Message::User { content }
                    if content.iter().eq([&said, &image]))),
        "{first:?}"
    );
}

/// Takes over the user's messages that start with `!`, as a shell would.
fn run_in_shell(mut typed: Query<(&Incoming, &mut Decision), (ReadyToStart, IncomingMessages)>) {
    for (typed, mut decision) in &mut typed {
        if typed.author.kind == AuthorKind::User && typed.text().starts_with('!') {
            decision.decide(Decider::plugin("test"), Outcome::custom("ran in a shell"));
        }
    }
}

/// The record of the message whose text is `text`.
fn message_record(app: &App, text: &str) -> Option<EffectRecord> {
    let sent = |record: &EffectRecord| match &record.kind {
        EffectKind::Custom { kind, payload } => {
            &**kind == "message"
                && payload.get("text").and_then(serde_json::Value::as_str) == Some(text)
        }
        _ => false,
    };
    records(app).into_iter().find(sent)
}

#[test]
fn a_message_taken_over_in_gate_is_never_delivered_and_one_accepted_carries_the_turn_on() {
    let mut app = app();
    app.init_resource::<Ended>()
        .init_resource::<Judging>()
        .add_observer(log_ended)
        .add_systems(
            AgentPass,
            (
                hold_answers.in_set(Stage::Review),
                run_in_shell.in_set(Stage::Gate),
            ),
        );
    let model = MockCompletionModel::from_stream_turns([answer("first"), answer("second")]);
    let agent = agent_on(&mut app, &model);
    let held = |world: &mut World| world.resource::<Judging>().0.is_some();
    assert!(run_until(&mut app, held));
    let typed = |text: &str| Deliver::new(agent, text, DeliveryMode::Steer);
    // Taken over while the turn's answer is held: it neither reaches the
    // model nor carries the turn on.
    app.world_mut().trigger(typed("!ls"));
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(model.request_count(), 1);
    assert!(has::<ActiveTurn>(&app, agent));
    let taken = message_record(&app, "!ls");
    assert!(
        taken.as_ref().is_some_and(|record| {
            record.disposition == Disposition::AnsweredBeforeStart
                && output(&record.outcome).contains("ran in a shell")
        }),
        "{taken:?}"
    );
    // Accepted, a message carries the answered turn on at once.
    let first_hold = app.world().resource::<Judging>().0.map(|(_, hold)| hold);
    app.world_mut().trigger(typed("and the tests?"));
    let asked_again = |_: &mut World| model.request_count() == 2;
    assert!(run_until(&mut app, asked_again), "{:?}", notices(&app));
    let second = model.requests().get(1).map(serde_json::to_string);
    assert!(
        matches!(&second, Some(Ok(sent)) if sent.contains("and the tests?") && !sent.contains("!ls")),
        "{second:?}"
    );
    let held_again =
        |world: &mut World| world.resource::<Judging>().0.map(|(_, hold)| hold) != first_hold;
    assert!(run_until(&mut app, held_again));
    if let Some((_, hold)) = app.world().resource::<Judging>().0 {
        app.world_mut().despawn(hold);
    }
    let ended_once = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended_once));
    let answer = match ended(&app).as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("second"), "{:?}", ended(&app));
    let said = serde_json::to_string(&conversation(&app, agent)).unwrap_or_default();
    assert!(
        !said.contains("!ls") && said.contains("and the tests?"),
        "{said}"
    );
    // An idle agent starts no turn with a message taken over.
    app.world_mut().trigger(typed("!pwd"));
    for _ in 0..3 {
        app.update();
    }
    assert!(!has::<ActiveTurn>(&app, agent));
    assert_eq!(model.request_count(), 2);
    assert!(message_record(&app, "!pwd").is_some());
}

/// A sender takes back a note its busy agent accepted and was not given
/// yet by deciding it: it leaves the inbox and never reaches the
/// conversation, though the turn ends without another model call.
#[test]
fn an_accepted_note_withdrawn_by_deciding_it_is_never_delivered() {
    let mut app = app();
    app.init_resource::<Ended>()
        .init_resource::<Judging>()
        .add_observer(log_ended)
        .add_systems(AgentPass, hold_answers.in_set(Stage::Review));
    let model = MockCompletionModel::from_stream_turns([answer("done")]);
    let agent = agent_on(&mut app, &model);
    let held = |world: &mut World| world.resource::<Judging>().0.is_some();
    assert!(run_until(&mut app, held));
    let note = |text: &str| {
        let note = Incoming::new(text, DeliveryMode::Inform).with_author(Author::plugin("check"));
        (note, ForAgent(agent))
    };
    let kept = app.world_mut().spawn(note("the docs build")).id();
    let taken_back = app.world_mut().spawn(note("the tests fail")).id();
    let accepted = |world: &mut World| {
        world.get::<Started>(kept).is_some() && world.get::<Started>(taken_back).is_some()
    };
    assert!(run_until(&mut app, accepted));
    if let Some(mut decision) = app.world_mut().get_mut::<Decision>(taken_back) {
        decision.decide(Decider::plugin("test"), Outcome::custom("withdrawn"));
    }
    app.update();
    if let Some((_, hold)) = app.world().resource::<Judging>().0 {
        app.world_mut().despawn(hold);
    }
    let ended_once = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, ended_once));
    let said = serde_json::to_string(&conversation(&app, agent)).unwrap_or_default();
    assert!(
        said.contains("the docs build") && !said.contains("the tests fail"),
        "{said}"
    );
    assert_eq!(model.request_count(), 1);
}

/// The holds a reviewer put on tool calls as they answered.
#[derive(Resource, Default)]
struct Reviewing(Vec<Entity>);

/// Holds each tool call's answer while it is reviewed.
fn hold_results(
    calls: Query<Entity, (ReadyToSettle, ToolCalls, Added<Outcome>)>,
    mut reviewing: ResMut<Reviewing>,
    mut commands: Commands,
) {
    for call in &calls {
        reviewing.0.push(commands.spawn(HoldOf(call)).id());
    }
}

/// A steering message its sender takes back by deciding it is never sent,
/// even when the turn's next call is built in the frame it was decided,
/// and the message sent in its place goes instead.
#[test]
fn a_steering_message_taken_back_as_the_next_call_is_built_is_never_sent() {
    let mut app = app();
    app.init_resource::<Reviewing>()
        .add_systems(AgentPass, hold_results.in_set(Stage::Review));
    let turns = [reply(&[("look", "1")]), answer("ok"), answer("done")];
    let model = MockCompletionModel::from_stream_turns(turns);
    let agent = agent_on(&mut app, &model);
    let held = |world: &mut World| !world.resource::<Reviewing>().0.is_empty();
    assert!(run_until(&mut app, held));
    let steer = |text: &str| (Incoming::new(text, DeliveryMode::Steer), ForAgent(agent));
    let old = app.world_mut().spawn(steer("use tabs")).id();
    let accepted = |world: &mut World| world.get::<Started>(old).is_some();
    assert!(run_until(&mut app, accepted));
    // In one frame: taken back, sent again changed, and the result let go.
    let world = app.world_mut();
    if let Some(mut decision) = world.get_mut::<Decision>(old) {
        decision.decide(Decider::plugin("test"), Outcome::custom("taken back"));
    }
    world.spawn(steer("use spaces"));
    for hold in std::mem::take(&mut world.resource_mut::<Reviewing>().0) {
        world.despawn(hold);
    }
    let done =
        |world: &mut World| model.request_count() == 3 && world.get::<ActiveTurn>(agent).is_none();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    let sent = serde_json::to_string(&model.requests()).unwrap_or_default();
    assert!(
        !sent.contains("use tabs") && sent.contains("use spaces"),
        "{sent}"
    );
    let said = serde_json::to_string(&conversation(&app, agent)).unwrap_or_default();
    assert!(
        !said.contains("use tabs") && said.contains("use spaces"),
        "{said}"
    );
}

/// Whether [`keep_on_stop`] keeps what a stopped turn did not send.
#[derive(Resource, Default)]
struct KeepUnsent(bool);

/// The text each [`Recalled`] handed back.
#[derive(Resource, Default)]
struct RecalledTexts(Vec<String>);

/// A model call that fails for good: the account is out of credit.
fn no_credit() -> Vec<MockStreamEvent> {
    let refused =
        ProviderResponseError::without_status("insufficient credit").with_transient(Some(false));
    vec![MockStreamEvent::Error(MockError::ProviderResponse(refused))]
}

/// Keeps the text of each [`Recalled`] in [`RecalledTexts`].
fn collect_recalled(app: &mut App) {
    app.init_resource::<RecalledTexts>().add_systems(
        Last,
        |mut read: MessageReader<Recalled>, mut texts: ResMut<RecalledTexts>| {
            for recalled in read.read() {
                texts.0.push(recalled.text.clone());
            }
        },
    );
}

/// Keeps the messages a turn the user stopped did not send, when
/// [`KeepUnsent`] says so.
fn keep_on_stop(
    unsent: Query<(Entity, &Unsent), Added<Unsent>>,
    keep: Res<KeepUnsent>,
    mut commands: Commands,
) {
    for (message, unsent) in &unsent {
        if keep.0 && matches!(unsent.ended, Some(TurnOutcome::Stopped(None))) {
            commands.entity(message).remove::<Unsent>();
        }
    }
}

/// What a stopped turn did not send is handed back to the user, unless a
/// plugin keeps it in Gate, the pass the turn ended in: then it goes with
/// the next turn's first call.
#[test]
fn a_message_a_stopped_turn_did_not_send_is_handed_back_unless_a_plugin_keeps_it() {
    let mut app = app();
    app.init_resource::<Reviewing>()
        .init_resource::<KeepUnsent>()
        .init_resource::<RecalledTexts>()
        .add_systems(
            AgentPass,
            (
                hold_results.in_set(Stage::Review),
                keep_on_stop.in_set(Stage::Gate),
            ),
        )
        .add_systems(
            Last,
            |mut read: MessageReader<Recalled>, mut texts: ResMut<RecalledTexts>| {
                texts.0.extend(read.read().map(|it| it.text.clone()));
            },
        );
    let turns = [
        reply(&[("look", "1")]),
        reply(&[("look", "2")]),
        answer("done"),
    ];
    let model = MockCompletionModel::from_stream_turns(turns);
    let agent = agent_on(&mut app, &model);
    let stop_with = |app: &mut App, text: &str| {
        let held = |world: &mut World| !world.resource::<Reviewing>().0.is_empty();
        assert!(run_until(app, held));
        let message = (Incoming::new(text, DeliveryMode::Steer), ForAgent(agent));
        let message = app.world_mut().spawn(message).id();
        assert!(run_until(app, |world| world
            .get::<Started>(message)
            .is_some()));
        app.world_mut().trigger(StopTurn::new(agent));
        app.world_mut().resource_mut::<Reviewing>().0.clear();
        app.update();
        message
    };
    stop_with(&mut app, "first steer");
    assert_eq!(app.world().resource::<RecalledTexts>().0, ["first steer"]);
    app.world_mut().resource_mut::<KeepUnsent>().0 = true;
    app.world_mut()
        .trigger(Deliver::new(agent, "again", DeliveryMode::Steer));
    let kept = stop_with(&mut app, "second steer");
    for _ in 0..3 {
        app.update();
    }
    assert!(has::<Started>(&app, kept) && !has::<Outcome>(&app, kept));
    assert_eq!(app.world().resource::<RecalledTexts>().0, ["first steer"]);
    app.world_mut()
        .trigger(Deliver::new(agent, "and now?", DeliveryMode::Steer));
    let done =
        |world: &mut World| model.request_count() == 3 && world.get::<ActiveTurn>(agent).is_none();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    let last = model.requests().last().map(serde_json::to_string);
    assert!(
        matches!(&last, Some(Ok(sent)) if sent.contains("second steer") && !sent.contains("first steer")),
        "{last:?}"
    );
}

/// A reply cut off at the length limit before it answered.
fn cut_off() -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::Reasoning {
            id: "r".to_owned(),
            text: "Let me think about".to_owned(),
        },
        MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::Length),
            ..Finish::default()
        }),
    ]
}

/// Carries on each turn the kernel proposes to fail, once, by calling the
/// model again.
fn continue_once(
    turns: Query<(Entity, &TurnOf, &TurnOutcome), Added<TurnOutcome>>,
    mut continued: Local<bool>,
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent), outcome) in &turns {
        if matches!(outcome, TurnOutcome::Failed(_)) && !*continued {
            *continued = true;
            commands.spawn((TurnCall, CallOf(turn), ForAgent(agent)));
        }
    }
}

#[test]
fn a_length_cutoff_is_a_proposal_that_a_new_turn_call_carries_on() {
    let mut app = app();
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, continue_once.in_set(Stage::Review));
    let model = MockCompletionModel::from_stream_turns([cut_off(), answer("done")]);
    let agent = agent_on(&mut app, &model);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    let answer = match ended(&app).as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("done"), "{:?}", ended(&app));
    assert_eq!(model.request_count(), 2);
    // The proposal was carried on from: no failure shown, the message kept.
    assert!(
        !notices(&app).iter().any(|notice| notice.contains("failed")),
        "{:?}",
        notices(&app)
    );
    let first = conversation(&app, agent).into_iter().next();
    assert!(matches!(first, Some(Message::User { .. })), "{first:?}");
}

#[test]
fn a_turn_a_plugin_fails_starts_nothing_more_and_ends_with_its_reason() {
    let mut app = app();
    let budget = |calls: Query<&CallOf, (ReadyToStart, With<TurnCall>)>, mut commands: Commands| {
        for &CallOf(turn) in &calls {
            let over = TurnOutcome::Failed("over budget".to_owned());
            commands.entity(turn).insert(over);
        }
    };
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, budget.in_set(Stage::Gate));
    let model = MockCompletionModel::from_stream_turns([answer("never")]);
    let agent = agent_on(&mut app, &model);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    assert!(
        matches!(ended(&app).as_slice(), [TurnOutcome::Failed(why)] if why == "over budget"),
        "{:?}",
        ended(&app)
    );
    assert_eq!(model.request_count(), 0, "the turn's call never started");
    assert_eq!(conversation(&app, agent), [], "the message is given back");
    let models: Vec<_> = model_records(&app)
        .iter()
        .map(|record| (record.disposition, record.started_ms.is_none()))
        .collect();
    assert_eq!(models, [(Disposition::AnsweredBeforeStart, true)]);
}

/// A model that streams the start of an answer, then never finishes.
struct Halfway;

impl Serve for Halfway {
    type Family = Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("halfway"),
            family: FamilyDescriptor::Custom {
                kind: "test:halfway".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        let text = Relayed::Item(Item::Event(StreamEvent::Text {
            part: Part::new(0),
            text: "The answer is".to_owned(),
        }));
        let stalls = futures::stream::iter([Ok(text)]).chain(futures::stream::pending());
        Reply::Stream(Box::pin(stalls))
    }
}

#[test]
fn an_interrupt_ends_the_turn_at_once_and_settles_its_held_and_streaming_calls() {
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    let model = MockCompletionModel::from_stream_turns([answer("unused")]);
    let agent = agent_on(&mut app, &model);
    if let Some(mut connection) = app.world_mut().get_mut::<Connection>(agent) {
        connection.handler = Handler(ErasedHandler::new(Halfway));
    }
    let streamed = |world: &mut World| {
        let mut partials = world.query_filtered::<&Partial, With<TurnCall>>();
        partials.iter(world).any(|partial| !partial.text.is_empty())
    };
    assert!(run_until(&mut app, streamed), "{:?}", notices(&app));
    let world = app.world_mut();
    let turn = world.get::<ActiveTurn>(agent).map(ActiveTurn::turn);
    let Some(turn) = turn else {
        return;
    };
    let check = world
        .spawn((open("check"), CallOf(turn), ForAgent(agent)))
        .id();
    let hold = world.spawn(HoldOf(check)).id();
    app.update();
    let world = app.world_mut();
    world.trigger(StopTurn::new(agent));
    world.flush();
    // At once: the hold is gone, the held call answered and settled, the
    // turn ended as stopped, and the streamed text kept.
    assert!(world.get_entity(hold).is_err());
    let answered = world.get::<Outcome>(check).map(|outcome| outcome.0.clone());
    assert!(
        matches!(&answered, Some(Err(report)) if report.message == "the user stopped the turn"),
        "{answered:?}"
    );
    assert!(world.get::<Settled>(check).is_some());
    assert!(world.get::<ActiveTurn>(agent).is_none());
    assert!(matches!(
        world.get::<TurnOutcome>(turn),
        Some(TurnOutcome::Stopped(None))
    ));
    assert!(world.get::<Settled>(turn).is_some());
    assert!(world.get::<Started>(check).is_none());
    let said = conversation(&app, agent);
    let kept = said.last().and_then(|message| message.final_answer());
    assert_eq!(kept.as_deref(), Some("The answer is"), "{said:?}");
    // Kept as stopped, never as an answer: the request is halted, and the
    // next message comes after the stopped note.
    let aborted = matches!(said.last(), Some(Message::Assistant(reply))
        if matches!(reply.stop, Some(StopReason::Aborted(_))));
    assert!(aborted, "{said:?}");
    let world = app.world_mut();
    if let Some(mut conversation) = world.get_mut::<Conversation>(agent) {
        assert_eq!(conversation.halted(), Some(Halt::Stopped));
        let ended = conversation
            .turns()
            .last()
            .and_then(|(_, record)| record.end);
        // The record ends at the kept text; the held call was no call of
        // the reply, so the stop answered none.
        let at = conversation.entries().next_back().map(|entry| entry.id);
        let stopped = TurnEnd::Stopped {
            why: None,
            at,
            calls: Vec::new(),
        };
        assert_eq!(ended, Some(stopped));
        conversation.push(Message::user("never mind"));
        let note = conversation.entries().rev().nth(1);
        assert!(note.is_some_and(|entry| entry.is_stop_marker()));
    }
    // A late release starts nothing.
    for _ in 0..3 {
        app.update();
    }
    assert!(app.world().get_entity(check).is_err(), "settled, then gone");
    assert!(
        matches!(ended(&app).as_slice(), [TurnOutcome::Stopped(None)]),
        "{:?}",
        ended(&app)
    );
    let cancelled: Vec<String> = records(&app)
        .iter()
        .filter(|record| record.disposition == Disposition::Cancelled)
        .map(|record| record.kind.name().to_owned())
        .collect();
    assert_eq!(cancelled, ["completion", "check"], "{:?}", records(&app));
}

/// A model call superseded while it streamed is not the turn's reply: an
/// interrupt before it settles keeps none of its text.
#[test]
fn an_interrupt_keeps_no_text_of_a_superseded_call() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([answer("unused")]);
    let agent = agent_on(&mut app, &model);
    let halfway = Handler(ErasedHandler::new(Halfway));
    if let Some(mut connection) = app.world_mut().get_mut::<Connection>(agent) {
        connection.handler = halfway;
    }
    let streamed = |world: &mut World| {
        let mut partials = world.query_filtered::<&Partial, With<TurnCall>>();
        partials.iter(world).any(|partial| !partial.text.is_empty())
    };
    assert!(run_until(&mut app, streamed), "{:?}", notices(&app));
    let world = app.world_mut();
    let connection = world.get::<Connection>(agent).cloned();
    let mut streaming = world.query_filtered::<(Entity, &ModelRequest, &Effect, &CallOf), (With<TurnCall>, Without<Settled>)>();
    let loser = streaming
        .iter(world)
        .next()
        .map(|(call, request, effect, of)| (call, request.clone(), effect.parent, of.0));
    let (Some((loser, request, parent, turn)), Some(connection)) = (loser, connection) else {
        return;
    };
    world
        .spawn((
            request.on(connection),
            TurnCall,
            CallOf(turn),
            ForAgent(agent),
            Effect::with_parent_id(parent),
        ))
        .add_related::<SupersededBy>(&[loser]);
    world.flush();
    world.trigger(StopTurn::new(agent));
    world.flush();
    let said = conversation(&app, agent);
    assert!(
        !said
            .iter()
            .filter_map(|message| message.final_answer())
            .any(|text| text.contains("The answer is")),
        "{said:?}"
    );
}

/// Two calls of a turn streaming halfway and a third: once both name the
/// third their successor, only its reply is the turn's, and both are
/// recorded as superseded.
#[test]
fn a_race_of_any_size_commits_only_its_winner() {
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    let model = MockCompletionModel::from_stream_turns([answer("unused")]);
    let agent = agent_on(&mut app, &model);
    let halfway = Handler(ErasedHandler::new(Halfway));
    let Some(mut first) = app.world_mut().get_mut::<Connection>(agent) else {
        return;
    };
    first.handler = halfway;
    let streaming = |world: &mut World| {
        let mut partials = world.query_filtered::<&Partial, With<TurnCall>>();
        partials
            .iter(world)
            .filter(|partial| !partial.text.is_empty())
            .count()
    };
    assert!(run_until(&mut app, |world| streaming(world) == 1));
    let world = app.world_mut();
    let racing = world.get::<Connection>(agent).cloned();
    let mut first = world.query_filtered::<(Entity, &ModelRequest, &Effect, &CallOf), (With<TurnCall>, Without<Settled>)>();
    let first = first
        .iter(world)
        .next()
        .map(|(call, request, effect, of)| (call, request.clone(), effect.parent, of.0));
    let winning = MockCompletionModel::from_stream_turns([answer("ok")]);
    let (Some((first, request, parent, turn)), Some(racing), Some(winning)) =
        (first, racing, connection(&winning))
    else {
        return;
    };
    let racer = |request: ModelRequest| {
        (
            request,
            TurnCall,
            CallOf(turn),
            ForAgent(agent),
            Effect::with_parent_id(parent),
        )
    };
    let second = world.spawn(racer(request.clone().on(racing))).id();
    assert!(run_until(&mut app, |world| streaming(world) == 2));
    let world = app.world_mut();
    let winner = world.spawn(racer(request.on(winning))).id();
    world.entity_mut(first).insert(SupersededBy(winner));
    world.entity_mut(second).insert(SupersededBy(winner));
    assert_eq!(world.get::<Supersedes>(winner).map(|of| of.len()), Some(2));
    assert!(run_until(&mut app, |world| !world
        .resource::<Ended>()
        .0
        .is_empty()));
    let answer = match ended(&app).as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("ok"), "{:?}", ended(&app));
    assert_eq!(conversation(&app, agent).len(), 2);
    let mut models: Vec<_> = model_records(&app)
        .iter()
        .map(|record| record.disposition)
        .collect();
    models.sort_by_key(|disposition| format!("{disposition:?}"));
    assert_eq!(
        models,
        [
            Disposition::Ran,
            Disposition::Superseded,
            Disposition::Superseded
        ]
    );
}

/// Retires the agent of each turn that ends, as a plugin whose agents do
/// one job each does.
fn retire_when_done(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<TurnOutcome>>,
    mut commands: Commands,
) {
    if let Ok(&ForAgent(agent)) = turns.get(settled.entity) {
        commands.trigger(Retire { entity: agent });
    }
}

#[test]
fn an_agent_retired_as_its_turn_ends_goes_with_the_turn() {
    let mut app = app();
    // The app's own error handler, which panics on a failed command.
    app.world_mut().remove_resource::<FallbackErrorHandler>();
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        .add_observer(retire_when_done);
    let model = MockCompletionModel::from_stream_turns([answer("done")]);
    let agent = agent_on(&mut app, &model);
    let gone = |world: &mut World| world.get_entity(agent).is_err();
    assert!(run_until(&mut app, gone), "{:?}", notices(&app));
    assert!(
        matches!(ended(&app).as_slice(), [TurnOutcome::Answered(_)]),
        "{:?}",
        ended(&app)
    );
    app.update();
    let world = app.world_mut();
    let turns = world.query_filtered::<(), With<Turn>>().iter(world).count();
    assert_eq!(turns, 0, "the turn went with its agent");
}

/// The agent of a turn whose reply called `stall`, once the call runs and
/// says what its result is if stopped: its output so far.
fn stalled(app: &mut App) -> Option<(Entity, Entity)> {
    app.add_tool(Stall);
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("s1", "stall", serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        answer("unused"),
    ]);
    let agent = agent_on(app, &model);
    let running = |world: &mut World| {
        call_named(world, "s1").is_some_and(|call| world.get::<Started>(call).is_some())
    };
    assert!(run_until(app, running), "{:?}", notices(app));
    let call = call_named(app.world_mut(), "s1")?;
    let so_far = ToolResult::success(ToolOutput::text("partial"));
    app.world_mut()
        .entity_mut(call)
        .insert(IfStopped(Outcome::from(so_far)));
    Some((agent, call))
}

/// The record of the call of the tool `tool`: how it ended and its
/// outcome.
fn record_of(app: &App, tool: &str) -> Option<(Disposition, String)> {
    let records = records(app);
    let record = records
        .iter()
        .find(|record| matches!(&record.kind, EffectKind::ToolCall { name, .. } if name == tool))?;
    Some((record.disposition, output(&record.outcome)))
}

#[test]
fn an_interrupted_call_answers_with_what_it_says_it_is_if_stopped() {
    let mut app = app();
    let Some((agent, _)) = stalled(&mut app) else {
        return;
    };
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    let said = conversation(&app, agent);
    let results = results_at(&said, said.len().saturating_sub(1));
    assert!(
        matches!(results.as_slice(), [(false, result)] if result.contains("partial")),
        "{said:?}"
    );
    let record = record_of(&app, "stall");
    assert!(
        matches!(&record, Some((Disposition::Cancelled, outcome)) if outcome.contains("partial")),
        "{record:?}"
    );
}

/// Never answers, and reads only.
struct Peek;

impl Tool for Peek {
    const NAME: &'static str = "peek";
    type Args = serde_json::Value;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Reads forever.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        _: serde_json::Value,
    ) -> Result<String, ToolExecutionError> {
        std::future::pending::<()>().await;
        Ok(String::new())
    }
}

/// A stopped call that may have changed something, a stopped one that
/// only read, and one that never started each read as rig-core's
/// `not_finished` says, beside a call that finished first. The turn's
/// record names the calls the stop answered, not the finished one.
#[test]
fn a_stopped_call_says_whether_it_ran_and_whether_it_may_have_changed_something() {
    let mut app = app();
    let paths = |path: &str| Scope::Paths(vec![PathBuf::from(path)]);
    let footprint = |touches| ToolOptions {
        footprint: Some(touches),
    };
    app.add_tool_with(Sum, footprint(Touches::Reads(paths("/work/c"))))
        .add_tool_with(Stall, footprint(Touches::Writes(paths("/work/b"))))
        .add_tool_with(Peek, footprint(Touches::Reads(paths("/work/a"))));
    // `touch` may change anything, so it waits behind the others.
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call("a1", "add", serde_json::json!({"a": 1, "b": 2})),
        MockStreamEvent::tool_call("s1", "stall", serde_json::json!({})),
        MockStreamEvent::tool_call("p1", "peek", serde_json::json!({})),
        MockStreamEvent::tool_call("t1", "touch", serde_json::json!({})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let agent = agent_on(&mut app, &model);
    let running = |world: &mut World| {
        let added =
            call_named(world, "a1").is_some_and(|call| world.get::<Outcome>(call).is_some());
        added
            && ["s1", "p1"].iter().all(|id| {
                call_named(world, id).is_some_and(|call| world.get::<Started>(call).is_some())
            })
    };
    assert!(run_until(&mut app, running), "{:?}", notices(&app));
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    let said = conversation(&app, agent);
    let results = results_at(&said, said.len().saturating_sub(1));
    let expected = [
        (false, "3"),
        (
            true,
            "Stopped: the user stopped the turn. It may have partly run; check before running it again.",
        ),
        (
            true,
            "Stopped: the user stopped the turn. It only read; run it again if you still need it.",
        ),
        (true, "Not run: the user stopped the turn."),
    ];
    let read: Vec<bool> = results
        .iter()
        .zip(expected)
        .map(|((error, result), (failed, text))| *error == failed && result.contains(text))
        .collect();
    assert_eq!(read, [true, true, true, true], "{results:?}");
    let ended = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|conversation| {
            let (_, record) = conversation.turns().next_back()?;
            match record.end? {
                TurnEnd::Stopped { at, calls, .. } => Some((
                    at == conversation.entries().next_back().map(|entry| entry.id),
                    calls,
                )),
                _ => None,
            }
        });
    let calls = ["s1", "p1", "t1"].map(CallId::from_wire).to_vec();
    assert_eq!(ended, Some((true, calls)));
}

/// Stopping an idle agent's turn stops the running turns of the agents
/// below it, and only theirs, in one flush.
#[test]
fn stopping_an_idle_agent_stops_the_agents_it_spawned() {
    let mut app = app();
    let Some((child, call)) = stalled(&mut app) else {
        return;
    };
    let world = app.world_mut();
    let parent = world.spawn(Agent).id();
    let other = world.spawn(Agent).id();
    world.entity_mut(child).insert(SpawnedBy(parent));
    world.flush();
    world.trigger(StopTurn::new(other));
    world.flush();
    assert!(world.get::<ActiveTurn>(child).is_some(), "not below it");
    world.trigger(StopTurn::new(parent));
    world.flush();
    assert!(world.get::<ActiveTurn>(child).is_none());
    assert!(world.get::<Settled>(call).is_some());
    let said = conversation(&app, child);
    let results = results_at(&said, said.len().saturating_sub(1));
    assert!(
        matches!(results.as_slice(), [(false, result)] if result.contains("partial")),
        "{said:?}"
    );
}

#[test]
fn a_call_of_a_failed_turn_answers_with_what_it_says_it_is_if_stopped() {
    let mut app = app();
    app.init_resource::<Ended>().add_observer(log_ended);
    let Some((agent, call)) = stalled(&mut app) else {
        return;
    };
    let turn = app.world().get::<CallOf>(call).map(|of| of.0);
    let Some(turn) = turn else {
        return;
    };
    let failed = TurnOutcome::Failed("over budget".to_owned());
    app.world_mut().entity_mut(turn).insert(failed);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    let record = record_of(&app, "stall");
    assert!(
        matches!(&record, Some((Disposition::AnsweredWhileRunning, outcome)) if outcome.contains("partial")),
        "{record:?}"
    );
    let said = conversation(&app, agent);
    let results = said
        .iter()
        .enumerate()
        .flat_map(|(at, _)| results_at(&said, at))
        .collect::<Vec<_>>();
    assert!(
        matches!(results.as_slice(), [(false, result)] if result.contains("partial")),
        "{said:?}"
    );
}

/// What model calls streamed, as a `PostUpdate` reader saw it, and each
/// call's latest stream stamp.
#[derive(Resource, Default)]
struct Streams {
    events: Vec<(Entity, StreamEvent)>,
    stamps: Vec<(Entity, StreamTiming)>,
}

#[test]
fn every_streamed_event_reaches_a_late_reader_in_order() {
    let mut app = app();
    let thought = MockStreamEvent::ReasoningDelta {
        id: "r".to_owned(),
        reasoning: "hm".to_owned(),
    };
    // Two parallel tool calls whose arguments stream interleaved, then an
    // answer.
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("l1", "look"),
            MockStreamEvent::tool_call_name_delta("t1", "touch"),
            MockStreamEvent::tool_call_arguments_delta("l1", "{\"a\":"),
            MockStreamEvent::tool_call_arguments_delta("t1", "{}"),
            MockStreamEvent::tool_call_arguments_delta("l1", "1}"),
            MockStreamEvent::tool_call_end("l1"),
            MockStreamEvent::tool_call_end("t1"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            thought,
            MockStreamEvent::text("Hel"),
            MockStreamEvent::text("lo"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    app.init_resource::<Streams>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(
            PostUpdate,
            |mut deltas: MessageReader<StreamDelta>,
             calls: Query<(Entity, &StreamTiming), (With<TurnCall>, Changed<StreamTiming>)>,
             mut seen: ResMut<Streams>| {
                let read = deltas
                    .read()
                    .map(|delta| (delta.effect, delta.event.clone()));
                seen.events.extend(read);
                for (call, stamp) in &calls {
                    seen.stamps.retain(|(seen, _)| *seen != call);
                    seen.stamps.push((call, *stamp));
                }
            },
        );
    let agent = agent_on(&mut app, &model);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    let seen = app.world().resource::<Streams>();
    // Both calls, the one that only called tools too, were stamped. The
    // tool call's last stamp and the text call's first can change in one
    // frame, which the reader then sees in either order, so the calls go in
    // the order their events came.
    assert_eq!(seen.stamps.len(), 2, "{:?}", seen.stamps);
    let mut stamps = seen.stamps.clone();
    stamps.sort_by_key(|(call, _)| seen.events.iter().position(|(effect, _)| effect == call));
    let [(tools, tools_stamp), (text, text_stamp)] = stamps.as_slice() else {
        return;
    };
    let of = |call: &Entity| -> Vec<&StreamEvent> {
        let of_call = seen.events.iter().filter(|(effect, _)| effect == call);
        of_call.map(|(_, event)| event).collect()
    };
    // Each tool call starts named, streams its arguments, and ends with
    // its id: the parts tell the interleaved calls apart.
    let mut started = Vec::new();
    let mut arguments = Vec::new();
    let mut ended = Vec::new();
    for event in of(tools) {
        match event {
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                name: Some(name),
            } => started.push((*part, name.to_string())),
            StreamEvent::Arguments { part, json } => arguments.push((*part, json.as_str())),
            StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(call),
            } => ended.push((*part, call.id.to_string(), call.function.name.to_string())),
            _ => {}
        }
    }
    assert_eq!(started.len(), 2, "{started:?}");
    let [(look, look_name), (touch, touch_name)] = started.as_slice() else {
        return;
    };
    assert!(look != touch && look_name == "look" && touch_name == "touch");
    assert_eq!(
        arguments,
        [(*look, "{\"a\":"), (*touch, "{}"), (*look, "1}")]
    );
    assert_eq!(
        ended,
        [
            (*look, "l1".to_owned(), "look".to_owned()),
            (*touch, "t1".to_owned(), "touch".to_owned())
        ]
    );
    let fragments: Vec<(&str, &str)> = of(text)
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::Reasoning { text, .. } => Some(("reasoning", text.as_str())),
            StreamEvent::Text { text, .. } => Some(("text", text.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        fragments,
        [("reasoning", "hm"), ("text", "Hel"), ("text", "lo")]
    );
    // Every event was an item, and so was the end.
    for (call, stamp) in [(tools, tools_stamp), (text, text_stamp)] {
        let events = of(call).len() as u64;
        assert!(stamp.items > events, "{stamp:?}");
        assert!(stamp.last >= stamp.first, "{stamp:?}");
    }
    let said = conversation(&app, agent);
    assert_eq!(
        said.last()
            .and_then(|message| message.final_answer())
            .as_deref(),
        Some("Hello")
    );
}

#[test]
fn a_chosen_model_the_catalog_lists_only_later_connects_then() {
    let mut app = app();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    world
        .entity_mut(agent)
        .insert(ModelChoice("later/big".to_owned()));
    app.update();
    assert!(!has::<Connection>(&app, agent), "not in the catalog yet");
    let Ok(vendor) = ProviderId::declare("later") else {
        return;
    };
    let mut models = app.world_mut().resource_mut::<Models>();
    let catalog = models.0.catalog_mut();
    catalog.insert(ModelSpec::new(vendor, "big").with_tools(true));
    models
        .0
        .provide(vendor, |spec: &ModelSpec, _: &Catalog, _: &Transport| {
            let model = MockCompletionModel::from_stream_turns([answer("hi")]);
            let model = ModelAdapter::<Completion>::new(spec.reference(), model);
            Ok(ErasedHandler::new(model))
        });
    app.update();
    let connected = app.world().get::<Connection>(agent);
    let connected = connected.map(|connection| connection.spec.reference());
    assert_eq!(connected.as_deref(), Some("later/big"));
}

/// The items a stream put on its entity, in order.
#[derive(Resource, Default)]
struct Items(Vec<u32>);

#[test]
fn a_streams_items_arrive_in_order_and_wake_the_loop() {
    let mut app = app();
    let woken = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&woken);
    let wake = Wake::new(move || {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    app.insert_resource(wake.clone())
        .init_resource::<Items>()
        .add_observer(
            |item: On<Insert<Done<u32>>>, done: Query<&Done<u32>>, mut items: ResMut<Items>| {
                items.0.extend(done.get(item.entity).map(|done| done.0));
            },
        );
    let pool = bevy_tasks::AsyncComputeTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let items = futures::stream::iter([1_u32, 2, 3]);
    let source = app
        .world_mut()
        .spawn(Running::stream(pool, &wake, items))
        .id();
    assert!(
        app.world()
            .get::<Running>(source)
            .is_some_and(Running::is_stream)
    );
    let ended = |world: &mut World| world.get::<Running>(source).is_none();
    assert!(run_until(&mut app, ended));
    assert_eq!(app.world().resource::<Items>().0, [1, 2, 3]);
    let woken = woken.load(std::sync::atomic::Ordering::Relaxed);
    assert!(woken >= 3, "woken {woken} times");
}

/// `model` as the catalog's DeepSeek model, said to call tools or not.
fn connection_with(model: &MockCompletionModel, tools: bool) -> Option<Connection> {
    let mut connection = connection(model)?;
    let mut spec = (*connection.spec).clone();
    spec.tools = tools;
    connection.spec = Arc::new(spec);
    Some(connection)
}

/// The app's agent on a model that calls no tools, answering with
/// `model`, offered `add` besides `look` and `touch`.
fn plain_agent(app: &mut App, model: &MockCompletionModel) -> Entity {
    let agent = agent_on(app, model);
    app.add_tool(Sum);
    let plain = connection_with(model, false);
    if let Some(plain) = plain {
        app.world_mut().entity_mut(agent).insert(plain);
    }
    agent
}

/// The tools a request offers, by name.
fn offered(request: Option<&rig_core::completion::CompletionRequest>) -> Vec<String> {
    let tools = request.into_iter().flat_map(|request| &request.tools);
    tools.map(|tool| tool.name.to_string()).collect()
}

#[test]
fn an_adapter_applies_to_the_model_a_call_is_finally_routed_to() {
    let mut app = app();
    let own = MockCompletionModel::from_stream_turns([answer("own")]);
    let routed = MockCompletionModel::from_stream_turns([answer("routed")]);
    let adapted = MockCompletionModel::from_stream_turns([answer("adapted")]);
    // What the adapter was given: whether that model calls tools.
    let given = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (seen, wrap) = (Arc::clone(&given), adapted.clone());
    let adapt = move |mut calls: Query<&mut Adapted, (Added<Admitted>, With<TurnCall>)>| {
        for mut adapted in &mut calls {
            if let Ok(mut seen) = seen.lock() {
                seen.push(adapted.0.spec.tools);
            }
            let model = ModelAdapter::<Completion>::new(adapted.0.spec.reference(), wrap.clone());
            adapted.0.handler = Handler(ErasedHandler::new(model));
        }
    };
    let to = connection_with(&routed, true);
    let route = move |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
        for mut call in &mut calls {
            call.connection = to.clone();
        }
    };
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(
            AgentPass,
            (
                adapt.in_set(PrepareSystems::Adapt),
                route.in_set(PrepareSystems::Edit),
            ),
        );
    plain_agent(&mut app, &own);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    let answer = match ended(&app).as_slice() {
        [TurnOutcome::Answered(message)] => message.final_answer(),
        _ => None,
    };
    assert_eq!(answer.as_deref(), Some("adapted"), "{:?}", ended(&app));
    let counts = (own.request_count(), routed.request_count());
    assert_eq!(counts, (0, 0), "only the adapted handler is called");
    assert_eq!(
        given.lock().map(|given| given.clone()).ok(),
        Some(vec![true])
    );
    // The agent's model calls no tools; the model it was routed to does,
    // so the request offers them.
    let tools = offered(adapted.requests().first());
    assert_eq!(tools, ["add", "look", "touch"]);
}

#[test]
fn an_outermost_adapter_added_first_wraps_one_added_later() {
    let mut app = app();
    let own = MockCompletionModel::from_stream_turns([answer("own")]);
    let adapted = MockCompletionModel::from_stream_turns([answer("adapted")]);
    // Each adapter as it was applied, and whether the model it got calls
    // tools.
    let applied = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (seen, wrap) = (Arc::clone(&applied), adapted.clone());
    let outer = Arc::clone(&applied);
    let outermost = move |calls: Query<&Adapted, (Added<Admitted>, With<TurnCall>)>| {
        for adapted in &calls {
            if let Ok(mut applied) = outer.lock() {
                applied.push(("outer", adapted.0.spec.tools));
            }
        }
    };
    let adapt = move |mut calls: Query<&mut Adapted, (Added<Admitted>, With<TurnCall>)>| {
        for mut adapted in &mut calls {
            if let Ok(mut applied) = seen.lock() {
                applied.push(("inner", adapted.0.spec.tools));
            }
            let mut spec = (*adapted.0.spec).clone();
            spec.tools = false;
            let model = ModelAdapter::<Completion>::new(spec.reference(), wrap.clone());
            adapted.0 = Connection {
                spec: Arc::new(spec),
                handler: Handler(ErasedHandler::new(model)),
            };
        }
    };
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, outermost.in_set(PrepareSystems::AdaptOutermost))
        .add_systems(AgentPass, adapt.in_set(PrepareSystems::Adapt));
    agent_on(&mut app, &own);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    // The outer adapter got what the inner one made: a model that calls no
    // tools, on the inner one's handler.
    assert_eq!(
        applied.lock().map(|applied| applied.clone()).ok(),
        Some(vec![("inner", true), ("outer", false)])
    );
    assert_eq!((own.request_count(), adapted.request_count()), (0, 1));
}

/// The system prompts of the model calls Prepare saw, in parts, and the
/// system message Fit measured.
#[derive(Resource, Default)]
struct Prompts(Vec<(Option<rig_core::completion::SystemPrompt>, Option<Message>)>);

#[test]
fn a_model_that_calls_no_tools_is_sent_none() {
    let mut app = app();
    let own = MockCompletionModel::from_stream_turns([answer("plain")]);
    // Each call's prompt in parts, then as Start sends it.
    let read = |calls: Query<(&ModelRequest, &Adapted), Added<Admitted>>,
                mut prompts: ResMut<Prompts>| {
        for (call, adapted) in &calls {
            let sent = adapted
                .sent(call)
                .chat_history
                .first()
                .map(|it| (**it).clone());
            prompts.0.push((call.prompt.clone(), sent));
        }
    };
    app.init_resource::<Ended>()
        .init_resource::<Prompts>()
        .add_observer(log_ended)
        .add_systems(AgentPass, read.in_set(PrepareSystems::Fit));
    let agent = plain_agent(&mut app, &own);
    app.world_mut()
        .entity_mut(agent)
        .insert(SystemPrompt("You add.".into()));
    app.world_mut()
        .spawn(PromptSection::new(300, "environment", "Today is Monday."));
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    // Prepare sees the parts, and Start sends them as they are.
    let prompts = &app.world().resource::<Prompts>().0;
    assert!(matches!(prompts.first(), Some((Some(_), _))), "{prompts:?}");
    let Some((Some(prompt), measured)) = prompts.first() else {
        return;
    };
    assert_eq!(prompt.role, "You add.");
    let tags: Vec<&str> = prompt.sections.iter().map(|s| s.tag.as_str()).collect();
    assert_eq!(tags, ["environment"]);
    let rendered = Message::system(prompt.text());
    assert_eq!(measured.as_ref(), Some(&rendered));
    // The model is sent no tools, and one system message.
    let requests = own.requests();
    let tools = offered(requests.first());
    assert!(tools.is_empty(), "{tools:?}");
    let system: Vec<&Message> = requests
        .iter()
        .flat_map(|request| &request.chat_history)
        .map(|message| &**message)
        .filter(|message| matches!(message, Message::System { .. }))
        .collect();
    assert_eq!(
        system,
        [&Message::system(
            "You add.\n\n<environment>\nToday is Monday.\n</environment>"
        )]
    );
}

/// A model's handler whose wire does not take `seed`.
struct NoSeed(ErasedHandler);

impl Serve for NoSeed {
    type Family = Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.0.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        self.0.serve(kind, dispatch).await
    }

    fn check(&self, _: &EffectKind) -> Vec<rig_core::completion::UnsupportedOption> {
        vec![rig_core::completion::UnsupportedOption::new(
            "seed", "test", "model", "no seed",
        )]
    }
}

/// What `Adapted::dropped` said in Prepare, and `DroppedOptions` after Start.
#[derive(Resource, Default)]
struct LeftOut(Vec<Vec<String>>);

fn options(dropped: &[rig_core::completion::UnsupportedOption]) -> Vec<String> {
    dropped.iter().map(|it| it.option.to_string()).collect()
}

#[test]
fn a_call_knows_the_options_its_model_left_out() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([answer("ok")]);
    let ask_top_p = |mut calls: Query<&mut ModelRequest, Added<Admitted>>| {
        for mut call in &mut calls {
            call.request_mut().options.top_p = Some(0.9);
        }
    };
    let measure = |calls: Query<(&ModelRequest, &Adapted), Added<Admitted>>,
                   mut left: ResMut<LeftOut>| {
        for (call, adapted) in &calls {
            left.0.push(options(&adapted.dropped(call)));
        }
    };
    app.init_resource::<Ended>()
        .init_resource::<LeftOut>()
        .add_observer(log_ended)
        .add_observer(
            |fitted: On<Add<DroppedOptions>>,
             calls: Query<&DroppedOptions>,
             mut left: ResMut<LeftOut>| {
                left.0
                    .extend(calls.get(fitted.entity).map(|it| options(&it.0)));
            },
        )
        .add_systems(
            AgentPass,
            (
                ask_top_p.in_set(PrepareSystems::Edit),
                measure.in_set(PrepareSystems::Fit),
            ),
        );
    let agent = agent_on(&mut app, &model);
    let spec = Catalog::builtin().resolve("anthropic/claude-fable-5");
    let Ok(spec) = spec.map(|spec| spec.shared()) else {
        return;
    };
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    let handler = Handler(ErasedHandler::new(NoSeed(ErasedHandler::new(handler))));
    app.world_mut()
        .entity_mut(agent)
        .insert(Connection { spec, handler });
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    // Fitting to the spec left `top_p` out; the handler's dry run names
    // `seed`. Prepare and Start agree.
    let left = &app.world().resource::<LeftOut>().0;
    assert_eq!(left, &[vec!["top_p", "seed"], vec!["top_p", "seed"]]);
    let sent = model.requests();
    assert_eq!(
        sent.first().map(|request| request.options.top_p),
        Some(None)
    );
}

/// What an adapter's stream rewrite says of a reply.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Rewritten(bool);

impl ContextValue for Rewritten {
    const KEY: &'static str = "rewritten";
}

/// What a reviewer saw on each model call: whether the model it went to
/// calls tools, the tools the request carried, and the adapter's fact.
#[derive(Resource, Default)]
struct SentCalls(Vec<(bool, usize, Option<Rewritten>)>);

#[test]
fn a_call_knows_the_model_it_went_to_and_what_its_adapters_published() {
    let mut app = app();
    let own = MockCompletionModel::from_stream_turns([answer("hi")]);
    // Says the model calls no tools, and tells the call it saw the reply.
    let adapt = |mut calls: Query<&mut Adapted, (Added<Admitted>, With<TurnCall>)>| {
        for mut adapted in &mut calls {
            let mut spec = (*adapted.0.spec).clone();
            spec.tools = false;
            let handler = adapted
                .0
                .handler
                .0
                .clone()
                .map_stream("plain", |published| {
                    move |item: Relayed| {
                        if let Relayed::Done(_) = &item {
                            published.insert(Rewritten(true)).ok();
                        }
                        vec![item]
                    }
                });
            adapted.0 = Connection {
                spec: Arc::new(spec),
                handler: Handler(handler),
            };
        }
    };
    let review = |calls: Query<(&ModelRequest, &SentTo, Option<&Published>), ReadyToSettle>,
                  mut seen: ResMut<SentCalls>| {
        for (call, sent, published) in &calls {
            let fact = published.and_then(|published| published.0.get::<Rewritten>().ok());
            let sent = (sent.0.spec.tools, call.request.tools.len(), fact.flatten());
            seen.0.push(sent);
        }
    };
    app.init_resource::<Ended>()
        .init_resource::<SentCalls>()
        .add_observer(log_ended)
        .add_systems(
            AgentPass,
            (
                adapt.in_set(PrepareSystems::Adapt),
                review.in_set(Stage::Review),
            ),
        );
    agent_on(&mut app, &own);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    // The agent's model calls tools; the adapted one does not, so the
    // request went without them.
    assert_eq!(
        app.world().resource::<SentCalls>().0,
        [(false, 0, Some(Rewritten(true)))]
    );
    let recorded = model_records(&app)
        .into_iter()
        .filter_map(|record| record.published)
        .map(|output| output.get::<Rewritten>().ok().flatten())
        .collect::<Vec<_>>();
    assert_eq!(recorded, [Some(Rewritten(true))]);
}

/// A plugin's mark on the message it sends.
#[derive(Component)]
struct Templated;

/// The turns messages were delivered in, and the parents of the turn
/// calls, in the order they were admitted.
#[derive(Resource, Default)]
struct Delivery {
    into: Vec<(Entity, Entity)>,
    parents: Vec<Option<rig_core::effect::EffectId>>,
}

#[test]
fn a_message_names_the_turn_it_went_with_and_starts_its_first_call() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([reply(&[("look", "l1")]), answer("done")]);
    let read_only = ToolOptions {
        footprint: Some(Touches::Reads(Scope::Anything)),
    };
    let parents = |calls: Query<&Effect, (Added<Admitted>, With<TurnCall>)>,
                   mut delivery: ResMut<Delivery>| {
        delivery
            .parents
            .extend(calls.iter().map(|call| call.parent));
    };
    app.add_tool_with(Look, read_only)
        .init_resource::<Delivery>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_observer(
            |delivered: On<Insert<DeliveredIn>>,
             messages: Query<&DeliveredIn>,
             mut delivery: ResMut<Delivery>| {
                if let Ok(&DeliveredIn(turn)) = messages.get(delivered.entity) {
                    delivery.into.push((delivered.entity, turn));
                }
            },
        )
        // A message that steers the turn while its tool call runs.
        .add_observer(
            |call: On<Add<ToolRequest>>, calls: Query<&ForAgent>, mut commands: Commands| {
                if let Ok(&ForAgent(agent)) = calls.get(call.entity) {
                    commands.trigger(Deliver::new(agent, "and then", DeliveryMode::Steer));
                }
            },
        )
        .add_systems(AgentPass, parents.in_set(Stage::Prepare));
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    let task = rig_core::effect::EffectId::from_raw(1000);
    let mut agent_mut = world.entity_mut(agent);
    agent_mut.insert(EffectParent(task));
    if let Some(connection) = connection(&model) {
        agent_mut.insert(connection);
    }
    // The plugin's own message: spawning it is all it takes.
    let typed =
        Incoming::new("go", DeliveryMode::Steer).with_author(Author::user().titled("review"));
    let sent = world.spawn((typed, ForAgent(agent), Templated)).id();
    let message = world
        .get::<Effect>(sent)
        .map(|effect| (effect.id(), effect.parent));
    assert_eq!(message.map(|(_, parent)| parent), Some(Some(task)));
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    let delivery = app.world().resource::<Delivery>();
    let [(first, turn), (steering, steered)] = delivery.into.as_slice() else {
        return assert_eq!(delivery.into.len(), 2, "{:?}", delivery.into);
    };
    assert_eq!(
        (*first, *turn),
        (sent, *steered),
        "both went with the one turn"
    );
    assert_ne!(steering, first);
    // The turn's first call is the message's work; the next is the agent's.
    let first_call = message.map(|(id, _)| id);
    assert_eq!(delivery.parents, [first_call, Some(task)]);
    // A user's message keeps its title, and goes to the model as typed.
    let conversation = app.world().get::<Conversation>(agent);
    let author = conversation.and_then(|conversation| conversation.author(0));
    assert_eq!(author, Some(&Author::user().titled("review")));
    let said = conversation.map(|conversation| format!("{:?}", conversation.messages()));
    assert!(said.is_some_and(|said| said.contains("\"go\"") && !said.contains("[Output of")));
}

#[test]
fn a_tool_the_agent_may_not_use_is_refused_and_reads_as_unknown() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([
        reply(&[("touch", "forbidden"), ("nope", "unknown")]),
        answer("done"),
    ]);
    app.init_resource::<Ended>().add_observer(log_ended);
    let agent = agent_on(&mut app, &model);
    let only_look = ToolAccess::Only(vec!["look".to_owned()]);
    app.world_mut().entity_mut(agent).insert(only_look);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    // The model reads the same for both.
    let results = results_at(&conversation(&app, agent), 2);
    let texts: Vec<String> = results
        .iter()
        .map(|(_, text)| {
            let text = text.replace("touch", "x").replace("nope", "x");
            text.replace("forbidden", "id").replace("unknown", "id")
        })
        .collect();
    assert_eq!(texts.len(), 2, "{results:?}");
    assert_eq!(texts.first(), texts.get(1), "{results:?}");
    // The record of the forbidden call is a refusal, the other's is not.
    let refusal = |name: &str| {
        let record = records(&app).into_iter().find(
            |record| matches!(&record.kind, EffectKind::ToolCall { name: called, .. } if called == name),
        );
        record.and_then(|record| {
            record
                .outcome
                .err()
                .map(|report| (report.kind, report.refusal))
        })
    };
    assert_eq!(refusal("touch"), Some((ErrorKind::Denied, true)));
    assert!(matches!(
        refusal("nope"),
        Some((ErrorKind::Tool(ToolErrorKind::NotFound), false))
    ));
    // The message's record reads back as one.
    let sent = message_record(&app, "go").and_then(|record| MessageRecord::read(&record.kind));
    assert_eq!(
        sent,
        Some(MessageRecord {
            text: "go".to_owned(),
            author: Author::user(),
            mode: DeliveryMode::Steer,
        })
    );
}

#[test]
fn a_turns_calls_are_recorded_with_its_index() {
    let mut app = app();
    let model = MockCompletionModel::from_stream_turns([
        reply(&[("look", "first")]),
        answer("one"),
        reply(&[("look", "second")]),
        answer("two"),
    ]);
    app.init_resource::<Ended>().add_observer(log_ended);
    let agent = agent_on(&mut app, &model);
    let ended_n = |n: usize| move |world: &mut World| world.resource::<Ended>().0.len() == n;
    assert!(run_until(&mut app, ended_n(1)), "{:?}", notices(&app));
    app.world_mut()
        .trigger(Deliver::new(agent, "again", DeliveryMode::Steer));
    assert!(run_until(&mut app, ended_n(2)), "{:?}", notices(&app));
    let turns = |completions: bool| {
        let records = records(&app).into_iter().filter(|record| {
            completions == matches!(record.kind, EffectKind::Completion { .. })
                && matches!(
                    record.kind,
                    EffectKind::Completion { .. } | EffectKind::ToolCall { .. }
                )
        });
        records.map(|record| record.turn).collect::<Vec<_>>()
    };
    assert_eq!(turns(false), [Some(0), Some(1)]);
    assert_eq!(turns(true), [Some(0), Some(0), Some(1), Some(1)]);
    // A message is not a turn's call.
    assert_eq!(
        message_record(&app, "go").map(|record| record.turn),
        Some(None)
    );
}

/// Spawns a call of `tool`, with no arguments, for the app's agent.
fn spawn_call(world: &mut World, tool: &str, id: &str) -> Entity {
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    ToolName::new(tool).map_or(Entity::PLACEHOLDER, |name| {
        let call = ToolCall::from_wire(id, ToolFunction::new(name, serde_json::json!({})));
        world.spawn((ToolRequest { call }, ForAgent(agent))).id()
    })
}

/// A tool call of a group that starts all together or not at all.
#[derive(Component)]
struct Member;

/// A coordinator after the gates: once a member of the group is answered
/// with an error, or refused, it refuses the members still unanswered.
fn coordinate(
    mut members: Query<(Option<&Outcome>, &mut Decision), (With<Member>, Without<Started>)>,
) {
    let refused = members.iter().any(|(outcome, decision)| {
        outcome
            .or(decision.get())
            .is_some_and(|outcome| outcome.0.is_err())
    });
    if !refused {
        return;
    }
    for (outcome, mut decision) in &mut members {
        if outcome.is_none() {
            decision.decide(
                Decider::plugin("group"),
                Outcome::refused("a member was refused"),
            );
        }
    }
}

#[test]
fn a_coordinator_after_the_gates_sees_a_call_the_kernel_refuses() {
    let mut app = app();
    app.add_tool(Look)
        .add_systems(AgentPass, coordinate.in_set(GateSystems::After));
    let world = app.world_mut();
    let look = spawn_call(world, "look", "look");
    let nope = spawn_call(world, "nope", "nope");
    world.entity_mut(look).insert(Member);
    world.entity_mut(nope).insert(Member);
    let done = |world: &mut World| {
        world.get::<Settled>(look).is_some() && world.get::<Settled>(nope).is_some()
    };
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    // Neither member started: the call of a tool that does not exist was
    // answered before the gates, so the coordinator refused the other.
    assert!(!has::<Started>(&app, look) && !has::<Started>(&app, nope));
    let records = records(&app);
    let outcome = |name: &str| {
        let record = records.iter().find(
            |record| matches!(&record.kind, EffectKind::ToolCall { name: called, .. } if called == name),
        );
        record.and_then(|record| record.outcome.as_ref().err().map(|report| report.kind))
    };
    assert_eq!(outcome("look"), Some(ErrorKind::Denied), "{records:?}");
    assert_eq!(
        outcome("nope"),
        Some(ErrorKind::Tool(ToolErrorKind::NotFound)),
        "{records:?}"
    );
}

/// An answer after the gates that stands in for running every tool call,
/// such as a cached result.
fn stand_in(mut calls: Query<&mut Decision, (ReadyToStart, ToolCalls)>) {
    for mut decision in &mut calls {
        decision.decide(Decider::plugin("cache"), ToolResult::skipped("cached"));
    }
}

#[test]
fn a_call_the_agent_may_not_make_is_refused_not_answered_in_its_place() {
    let mut app = app();
    app.add_tool(Look)
        .add_systems(AgentPass, stand_in.in_set(GateSystems::After));
    let world = app.world_mut();
    let call = spawn_call(world, "look", "look");
    let agent = world.get::<ForAgent>(call).map(|of| of.0);
    if let Some(agent) = agent {
        world.entity_mut(agent).insert(ToolAccess::Only(Vec::new()));
    }
    let done = |world: &mut World| world.get::<Settled>(call).is_some();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    let outcome = app.world().get::<Outcome>(call).map(|outcome| &outcome.0);
    assert!(
        matches!(outcome, Some(Err(report)) if report.kind == ErrorKind::Denied && report.refusal),
        "{outcome:?}"
    );
    let records = records(&app);
    let record = records
        .iter()
        .find(|record| matches!(record.kind, EffectKind::ToolCall { .. }));
    assert!(
        record.is_some_and(|record| record.decided_by.is_none()
            && record.disposition == Disposition::AnsweredBeforeStart),
        "{records:?}"
    );
}

/// A hold a coordinator keeps until every other hold on its effect is gone.
#[derive(Component)]
struct Mine;

/// Releases each of its holds once it is the last one on its effect.
fn release_when_alone(
    mine: Query<(Entity, &HoldOf), With<Mine>>,
    holds: Query<&Holds>,
    mut commands: Commands,
) {
    for (hold, of) in &mine {
        if holds.get(of.0).is_ok_and(|holds| holds.len() == 1) {
            commands.entity(hold).despawn();
        }
    }
}

#[test]
fn releasing_one_of_two_holds_wakes_the_loop() {
    let mut app = app();
    app.add_systems(AgentPass, release_when_alone.in_set(GateSystems::After));
    let world = app.world_mut();
    let effect = world.spawn(open("job")).id();
    world.spawn((HoldOf(effect), Mine));
    let other = world.spawn(HoldOf(effect)).id();
    app.update();
    assert!(!has::<Started>(&app, effect));
    app.world_mut().despawn(other);
    app.update();
    app.update();
    assert!(has::<Started>(&app, effect), "the coordinator let go");
}

/// `model` as the catalog's DeepSeek model, with the cache retentions it
/// honours.
fn connection_caching(
    model: &MockCompletionModel,
    retention: Vec<rig_core::completion::CacheRetention>,
) -> Option<Connection> {
    let mut connection = connection(model)?;
    let caching = rig_core::catalog::CacheSupport::new(retention);
    let spec = (*connection.spec).clone().with_caching(caching);
    connection.spec = Arc::new(spec);
    Some(connection)
}

#[test]
fn a_call_routed_to_another_model_asks_for_that_models_caching() {
    use rig_core::completion::{CacheRetention, CompletionRequest};
    let mut app = app();
    let own = MockCompletionModel::from_stream_turns([answer("own")]);
    let routed = MockCompletionModel::from_stream_turns([answer("routed"), answer("summary")]);
    let to = connection_caching(&routed, vec![CacheRetention::Short]);
    let route = move |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
        for mut call in &mut calls {
            call.connection = to.clone();
        }
    };
    app.init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, route.in_set(PrepareSystems::Edit));
    let agent = agent_on(&mut app, &own);
    if let Some(plain) = connection_caching(&own, Vec::new()) {
        app.world_mut().entity_mut(agent).insert(plain);
    }
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    // A call a plugin makes, leaving its caching to the model.
    let summary = CompletionRequest::from(vec![Message::user("sum it up")]);
    let summary = connection_caching(&routed, vec![CacheRetention::Short])
        .map(|to| ModelRequest::new(summary).on(to));
    if let Some(summary) = summary {
        app.world_mut()
            .spawn((summary, Effect::default(), ForAgent(agent)));
    }
    assert!(run_until(&mut app, |_| routed.request_count() == 2));
    let caching: Vec<_> = routed
        .requests()
        .iter()
        .map(|request| request.options.cache)
        .collect();
    assert_eq!(caching, [Some(CacheRetention::Short); 2]);
    assert_eq!(own.request_count(), 0);
}

/// The layers of the handler each settled model call went to.
#[derive(Resource, Default)]
struct SentLayers(Vec<Vec<String>>);

fn log_sent_layers(
    settled: On<Add<Settled>>,
    calls: Query<&SentTo, With<TurnCall>>,
    mut seen: ResMut<SentLayers>,
) {
    if let Ok(SentTo(connection)) = calls.get(settled.entity) {
        seen.0.push(connection.handler.0.descriptor().layers);
    }
}

#[test]
fn a_retried_call_is_adapted_once() {
    let mut app = app();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs(30)));
    let model = MockCompletionModel::from_stream_turns([overloaded(), answer("ok")]);
    let adapt = |mut calls: Query<&mut Adapted, (Added<Admitted>, With<TurnCall>)>| {
        for mut adapted in &mut calls {
            let handler = adapted.0.handler.0.clone();
            adapted.0.handler = Handler(handler.map_stream("count", |_| |item| vec![item]));
        }
    };
    app.init_resource::<Ended>()
        .init_resource::<SentLayers>()
        .add_observer(log_ended)
        .add_observer(log_sent_layers)
        .add_systems(AgentPass, adapt.in_set(PrepareSystems::Adapt));
    agent_on(&mut app, &model);
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    assert_eq!(model.request_count(), 2);
    // The failed call and its retry each went through the adapter once.
    let layers = &app.world().resource::<SentLayers>().0;
    let counted: Vec<usize> = layers
        .iter()
        .map(|layers| layers.iter().filter(|it| *it == "count").count())
        .collect();
    assert_eq!(counted, [1, 1], "{layers:?}");
}

/// The hold the gate put on the first turn call it saw.
#[derive(Resource, Default)]
struct FirstHold(Option<Entity>);

#[test]
fn a_held_call_goes_to_the_model_it_was_adapted_for() {
    let mut app = app();
    let first = MockCompletionModel::from_stream_turns([answer("first")]);
    let second = MockCompletionModel::from_stream_turns([answer("second")]);
    let hold_once = |calls: Query<Entity, (ReadyToStart, With<TurnCall>)>,
                     mut held: ResMut<FirstHold>,
                     mut commands: Commands| {
        for call in &calls {
            if held.0.is_none() {
                held.0 = Some(commands.spawn(HoldOf(call)).id());
            }
        }
    };
    app.init_resource::<Ended>()
        .init_resource::<FirstHold>()
        .add_observer(log_ended)
        .add_systems(AgentPass, hold_once.in_set(Stage::Gate));
    let agent = agent_on(&mut app, &first);
    let held = |world: &mut World| world.resource::<FirstHold>().0.is_some();
    assert!(run_until(&mut app, held), "{:?}", notices(&app));
    // While the call is held, the agent's model changes.
    let changed = connection(&second);
    assert!(changed.is_some(), "the second model connects");
    if let Some(changed) = changed {
        app.world_mut().entity_mut(agent).insert(changed);
    }
    if let Some(hold) = app.world().resource::<FirstHold>().0 {
        app.world_mut().despawn(hold);
    }
    let done = |world: &mut World| !world.resource::<Ended>().0.is_empty();
    assert!(run_until(&mut app, done), "{:?}", notices(&app));
    // It goes where Fit measured it: the model it was adapted for.
    assert_eq!((first.request_count(), second.request_count()), (1, 0));
}
