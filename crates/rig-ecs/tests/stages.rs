//! The stage rules on their own: plugin effects ([`EffectRequest`]) and a
//! scripted model's tool calls ([`ToolRequest`]) on a headless kernel app,
//! with stage systems as plugins write them. Effects chain within the
//! frame their outcomes arrive in, the first answer wins, holds add up,
//! every Prepare and Gate system sees a late spawn before it starts, a
//! review keeps the raw outcome in the record, a failing stage system
//! closes its stage, and every way an effect ends is recorded. A reply's
//! calls start, and their results go back, in the frames they arrive; a
//! gate sees each call when it is about to run; a refusal answers in call
//! order and is recorded as one.
//!
//! Of the design's twelve rule tests (design.md §5), these land here: 1
//! (single-frame chaining, with a plugin effect and with a reply's calls),
//! 2 (gate when ready), 3 (refusal order), 4 (the conflict rule), 5
//! (admission), 6 (review) and 10 (fail closed). The rest land with the
//! kernel's other effect kinds: 8 (re-entry) and 9 (refusal is not
//! failure) with model calls; 7 (turn endings) and 11 (interrupt) with
//! turn endings; 12 (restore) with the restore commit, in
//! `src/restore/tests.rs`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy_ecs::error::{FallbackErrorHandler, warn};
use bevy_time::TimeUpdateStrategy;
use rig_core::catalog::Catalog;
use rig_core::effect::{Disposition, EffectKind, EffectRecord};
use rig_core::error::ErrorKind;
use rig_core::message::Message;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::{ModelAdapter, ToolAdapter};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::effects::Handler;
use rig_ecs::prelude::*;
use rig_ecs::turn::ModelCall;
use serde::Deserialize;

/// How long a test waits for a tool to answer.
const DEADLINE: Duration = Duration::from_secs(10);

/// Adds two numbers.
struct Sum;

#[derive(Deserialize)]
struct AddArgs {
    a: i64,
    b: i64,
}

impl PortableTool for Sum {
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

    async fn call(&self, args: AddArgs) -> Result<i64, ToolExecutionError> {
        Ok(args.a + args.b)
    }
}

/// Never answers.
struct Stall;

impl PortableTool for Stall {
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

    async fn call(&self, _: serde_json::Value) -> Result<String, ToolExecutionError> {
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

/// Every [`Recorded`] and [`Notice`] so far.
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
            |mut records: MessageReader<Recorded>,
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
fn output(outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>) -> String {
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
fn the_first_answer_wins() {
    let mut app = app();
    app.add_systems(
        AgentPass,
        (
            |effects: Query<Entity, Undecided>, mut commands: Commands| {
                for effect in &effects {
                    commands
                        .entity(effect)
                        .insert_if_new(Outcome::refused("not now"));
                }
            },
            // Ordered after the refusal, and blind to it, this one answers
            // too: it loses.
            (|effects: Query<Entity, (With<EffectRequest>, Without<Settled>)>,
              mut commands: Commands| {
                for effect in &effects {
                    let late = rig_core::tool::ToolResult::skipped("allowed");
                    let late = rig_core::effect::Outcome::ToolResult { result: late };
                    commands.entity(effect).insert_if_new(Outcome(Ok(late)));
                }
            }),
        )
            .chain()
            .in_set(Stage::Gate),
    );
    let effect = app.world_mut().spawn(open("deploy")).id();
    app.update();
    assert!(has::<Settled>(&app, effect), "settled in its first frame");
    assert!(!has::<Started>(&app, effect), "never started");
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
    assert!(record.started_ms.is_none());
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
                |names: Query<&Name, Undecided>,
                 mut seen: ResMut<Seen>,
                 mut late: ResMut<SpawnedLate>,
                 mut commands: Commands| {
                    noted(&mut seen, "gate a", names);
                    if !late.0 {
                        late.0 = true;
                        commands.spawn((open("late"), Name::new("late")));
                    }
                },
                |names: Query<&Name, Undecided>, mut seen: ResMut<Seen>| {
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

/// Marks an outcome the reviewer replaced.
#[derive(Component)]
struct Redacted;

/// The reviewer's hold.
#[derive(Resource, Default)]
struct ReviewHold(Option<Entity>);

#[test]
fn a_review_holds_and_replaces_and_the_record_keeps_the_raw_outcome() {
    let mut app = app();
    let redact = |effects: Query<Entity, (Unreviewed, Without<Redacted>)>,
                  mut hold: ResMut<ReviewHold>,
                  mut commands: Commands| {
        for effect in &effects {
            let redacted = rig_core::tool::ToolResult::skipped("redacted");
            let redacted = rig_core::effect::Outcome::ToolResult { result: redacted };
            commands
                .entity(effect)
                .insert((Outcome(Ok(redacted)), Redacted));
            hold.0 = Some(commands.spawn(HoldOf(effect)).id());
        }
    };
    let read = |outcomes: Query<&Outcome, Unreviewed>, mut seen: ResMut<Seen>| {
        seen.0
            .extend(outcomes.iter().map(|outcome| output(&outcome.0)));
    };
    app.init_resource::<ReviewHold>()
        .add_systems(AgentPass, (redact, read).chain().in_set(Stage::Review));
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
        "the later reviewer saw the replacement only: {seen:?}"
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
        reviewed.is_some_and(|reviewed| reviewed.contains("redacted")),
        "{record:?}"
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
    // An effect in flight has the record it would leave if cancelled now.
    let now = record_of(app.world(), running);
    assert!(
        now.is_some_and(|now| now.disposition == Disposition::Cancelled
            && now.started_ms.is_some()
            && Some(now.id) == parent),
        "{:?}",
        record_of(app.world(), running)
    );

    // Answered from outside while it runs: its task is cancelled.
    let answer = rig_core::tool::ToolResult::skipped("answered");
    let answer = rig_core::effect::Outcome::ToolResult { result: answer };
    let world = app.world_mut();
    world.entity_mut(running).insert(Outcome(Ok(answer)));
    // Replaced while held: it never starts.
    let successor = world.spawn((open("successor"), Supersedes(replaced))).id();
    // Despawned before it settles.
    world.despawn(dropped);
    app.update();
    assert!(has::<Settled>(&app, running) && !has::<Running>(&app, running));
    assert!(has::<Settled>(&app, replaced) && !has::<Started>(&app, replaced));
    assert!(has::<Started>(&app, successor));
    let records = records(&app);
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

impl PortableTool for Look {
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

    async fn call(&self, _: serde_json::Value) -> Result<String, ToolExecutionError> {
        Ok("seen".to_owned())
    }
}

/// Answers at once; may change anything, so its calls run alone.
struct Touch;

impl PortableTool for Touch {
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

    async fn call(&self, _: serde_json::Value) -> Result<String, ToolExecutionError> {
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
    let spec = Catalog::builtin()
        .resolve("deepseek/deepseek-flash")
        .map(|spec| spec.shared());
    let read_only = ToolOptions {
        footprint: Footprint::ReadOnly,
        ..ToolOptions::default()
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
    world.trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    agent
}

fn conversation(app: &App, agent: Entity) -> Vec<Message> {
    let conversation = app.world().get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |conversation| conversation.messages().to_vec())
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
    // The frame each step was first seen in, after the update.
    let (mut replied, mut started, mut answered, mut asked) = (None, None, None, None);
    let done = run_until(&mut app, |world| {
        let frame = world.resource::<Frame>().0;
        let said = world
            .get::<Conversation>(agent)
            .map_or(0, |c| c.messages().len());
        let mut calls = world.query_filtered::<(), (With<ToolRequest>, With<Started>)>();
        let running = calls.iter(world).count();
        let mut models = world.query_filtered::<(), With<ModelCall>>();
        let calling = models.iter(world).count() > 0;
        replied = replied.or((said >= 2).then_some(frame));
        started = started.or((running == 3).then_some(frame));
        answered = answered.or((said >= 3).then_some(frame));
        asked = asked.or((answered.is_some() && calling).then_some(frame));
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
    calls: Query<(Entity, &ToolRequest), Undecided>,
    settled: Query<&ToolRequest, With<Settled>>,
    mut gated: ResMut<Gated>,
    mut commands: Commands,
) {
    for (call, request) in &calls {
        let id = request.call.id.to_string();
        let done = settled.iter().map(|done| done.call.id.to_string());
        gated.seen.push((id.clone(), done.collect()));
        match id.as_str() {
            "held" if !gated.held_once => {
                gated.held_once = true;
                gated.hold = Some(commands.spawn(HoldOf(call)).id());
            }
            "refused" => {
                commands
                    .entity(call)
                    .insert_if_new(Outcome::refused("not this one"));
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
