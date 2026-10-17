use std::sync::mpsc::Receiver;

use rig_basics::subagents::{SubagentsPlugin, TASK};
use rig_core::catalog::{Catalog, Connector};
use rig_core::effect::Answered;
use rig_core::effect::EffectKind;
use rig_core::error::ErrorReport;
use rig_core::serve::{AfterServe, BeforeServe as Pass, Intercept};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_ecs::effects::Handler;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    answer_open, app_on, calls, connect, reply, run_until, scratch_home,
};
use schemars::JsonSchema;
use serde_json::json;

use super::*;

/// Every model request waits this long before the model answers, as a
/// provider takes its time to the first token.
const FIRST_TOKEN: Duration = Duration::from_millis(200);

struct SlowStart;

impl Intercept for SlowStart {
    fn name(&self) -> String {
        "slow-start".to_owned()
    }

    async fn before(&self, _: EffectId, _: &EffectKind) -> Pass {
        blocking(|| {
            std::thread::sleep(FIRST_TOKEN);
            Ok::<_, ToolExecutionError>(())
        })
        .await
        .ok();
        Pass::Proceed
    }

    async fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        AfterServe::Keep
    }
}

/// `sleep`: waits, and touches nothing, so calls of one reply run side by
/// side. (`shell` may change anything, so the kernel runs its calls one
/// after another.)
struct Sleep;

#[derive(Deserialize, JsonSchema)]
struct SleepArgs {
    /// Seconds.
    seconds: f64,
}

impl Tool for Sleep {
    const NAME: &'static str = "sleep";
    type Args = SleepArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Sleep.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<SleepArgs>()
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: SleepArgs,
    ) -> Result<String, ToolExecutionError> {
        blocking(move || {
            std::thread::sleep(Duration::from_secs_f64(args.seconds));
            Ok("slept".to_owned())
        })
        .await
    }
}

/// An app on a session in `dir`, with the plugin, whose one agent talks to
/// `model`, each of its requests slowed by [`FIRST_TOKEN`] when `slow`.
fn agent_app(
    dir: &SessionPaths,
    model: &MockCompletionModel,
    slow: bool,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(SessionPaths(dir.0.clone()))
        .insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((AgentPlugin, SessionLogPlugin, TurnTimelinePlugin));
    let Some(mut connection) = connect(model) else {
        return (app, wakes, Entity::PLACEHOLDER);
    };
    if slow {
        connection.handler = Handler(connection.handler.0.layered(SlowStart));
    }
    let choice = ModelChoice(connection.spec.reference());
    let agent = app.world_mut().spawn((Agent, connection, choice)).id();
    (app, wakes, agent)
}

fn session(name: &str) -> SessionPaths {
    let dir = scratch_home(name).session(&SessionId::generate());
    std::fs::create_dir_all(dir.path()).ok();
    SessionPaths(dir)
}

fn overlap(a: &Span, b: &Span) -> bool {
    a.start_ms < b.end_ms && b.start_ms < a.end_ms
}

#[test]
fn parallel_calls_overlap_and_the_turn_exports_and_survives_a_restart() {
    let both = vec![
        MockStreamEvent::tool_call("c1", "sleep", json!({ "seconds": 0.3 })),
        MockStreamEvent::tool_call("c2", "sleep", json!({ "seconds": 0.1 })),
        MockStreamEvent::final_response_with_default_usage(),
    ];
    let model = MockCompletionModel::from_stream_turns([both, reply("Slept.")]);
    let dir = session("turn-timeline-parallel");
    let (mut app, wakes, agent) = agent_app(&dir, &model, true);
    let reads = ToolOptions {
        footprint: Some(Touches::Reads(Scope::Anything)),
    };
    app.add_tool_with(Sleep, reads);
    app.finish();
    app.update();
    app.world_mut()
        .trigger(Deliver::new(agent, "sleep twice", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Timelines>().turns.is_empty()
    });

    let turns = app.world().resource::<Timelines>().turns.clone();
    assert_eq!(turns.len(), 1, "{turns:#?}");
    let Some(turn) = turns.first() else {
        return;
    };
    let of = |kind| turn.spans.iter().filter(move |span| span.kind == kind);
    let models: Vec<&Span> = of(SpanKind::Model).collect();
    let tools: Vec<&Span> = of(SpanKind::Tool).collect();
    assert_eq!((models.len(), tools.len()), (2, 2), "{turn:#?}");
    for model in &models {
        let waited = model.in_phase_ms(PhaseKind::FirstToken);
        assert!(waited >= 190.0, "first token after {waited} ms: {model:#?}");
    }
    let (Some(long), Some(short)) = (tools.first(), tools.get(1)) else {
        return;
    };
    assert!(overlap(long, short), "{tools:#?}");
    assert!(long.in_phase_ms(PhaseKind::Running) >= 300.0, "{long:#?}");
    let summary = Summary::of(turn);
    assert!(
        (290.0..390.0).contains(&summary.tool_ms),
        "tools ran side by side: {summary:?}"
    );
    assert!(summary.parallel_ms >= 90.0, "{summary:?}");
    let chart = render(app.world().resource::<Timelines>(), turn, false);
    assert!(
        chart.contains("sleep") && chart.contains("2+ at once"),
        "{chart}"
    );

    let path = dir.path().join("trace.json");
    let line = format!("timeline export {}", path.display());
    app.world_mut().trigger(RunCommand::new(agent, line));
    app.update();
    let trace: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    let events = trace
        .get("traceEvents")
        .and_then(|events| events.as_array());
    let field = |event: &serde_json::Value, key: &str| event.get(key).cloned();
    let complete = events.into_iter().flatten().filter(|event| {
        field(event, "ph") == Some(json!("X")) && field(event, "cat") != Some(json!("turn"))
    });
    assert_eq!(complete.count(), turn.spans.len(), "{trace:#}");

    // A resumed session reads the turn back.
    let saved = turn.clone();
    let (mut resumed, _, _) = agent_app(&dir, &model, false);
    resumed.finish();
    resumed.update();
    let turns = &resumed.world().resource::<Timelines>().turns;
    assert_eq!(turns.first(), Some(&saved));
}

/// On a call the user allowed.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Approved;

/// Asks before each `guarded` call runs.
fn ask(
    mut calls: Query<
        (Entity, &ToolRequest, &ForAgent, &Decision),
        (ReadyToStart, ToolCalls, Without<Approved>),
    >,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), decision) in &mut calls {
        if request.call.function.name.as_str() == "guarded" && !decision.is_decided() {
            commands.spawn((Ask::yes_no("Run guarded?"), ForAgent(agent), HoldOf(call)));
        }
    }
}

/// Lets the call run once the user answered.
fn answered(asks: Query<(Entity, &HoldOf), (With<Ask>, Added<Outcome>)>, mut commands: Commands) {
    for (ask, &HoldOf(call)) in &asks {
        commands.entity(call).insert(Approved);
        commands.entity(ask).remove::<HoldOf>();
    }
}

fn guarded(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("fh-turn-timeline"),
            ToolResult::success("done".into()),
        );
    }
}

#[derive(Resource, Default)]
struct Asked(bool);

#[test]
fn a_wait_on_the_user_and_a_subagent_nest_under_their_calls() {
    let task = json!({ "description": "Look", "prompt": "Look around." });
    // After the `task` call every reply is text, so which agent gets which
    // does not matter.
    let model = MockCompletionModel::from_stream_turns([
        calls("g1", "guarded", json!({})),
        calls("t1", TASK, task),
        reply("Started."),
        reply("Nothing here."),
        reply("Done."),
        reply("Done."),
    ]);
    let dir = session("turn-timeline-nested");
    let (mut app, wakes, parent) = agent_app(&dir, &model, false);
    app.add_plugins(SubagentsPlugin)
        .add_open_tool("guarded", "Do it.", ToolOptions::default(), guarded)
        .add_systems(AgentPass, (answered, ask).chain().in_set(Stage::Gate))
        .init_resource::<Asked>()
        .add_observer(|_: On<Add<Ask>>, mut asked: ResMut<Asked>| asked.0 = true);
    app.finish();
    app.update();
    app.world_mut()
        .trigger(Deliver::new(parent, "look around", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| world.resource::<Asked>().0);
    std::thread::sleep(Duration::from_millis(500));
    let yes = Answer::choice("yes");
    assert!(answer_open(app.world_mut(), "Run guarded?", &yes));
    run_until(&mut app, &wakes, |world| {
        let turns = &world.resource::<Timelines>().turns;
        let sub = turns.iter().any(|turn| turn.under.is_some());
        sub && turns.iter().any(|turn| turn.under.is_none())
    });

    let timelines = app.world().resource::<Timelines>();
    let turn = timelines
        .turns
        .iter()
        .find(|turn| turn.under.is_none() && turn.index == 0);
    assert!(turn.is_some(), "no parent turn: {timelines:#?}");
    let Some(turn) = turn else {
        return;
    };
    let named = |prefix: &str| {
        turn.spans
            .iter()
            .find(|span| span.label.starts_with(prefix))
    };
    let found = (named("guarded"), named(TASK), named("you:"));
    assert!(
        found.0.is_some() && found.1.is_some() && found.2.is_some(),
        "{turn:#?}"
    );
    let (Some(call), Some(task), Some(wait)) = found else {
        return;
    };
    assert_eq!(wait.kind, SpanKind::User);
    assert_eq!(wait.parent, Some(call.id));
    let waited = wait.duration_ms();
    assert!((500.0..900.0).contains(&waited), "waited {waited} ms");
    assert!(Summary::of(turn).user_ms >= 500.0);

    let subs: Vec<&TurnTimeline> = timelines.nested_under(turn, task.id).collect();
    assert_eq!(
        subs.len(),
        1,
        "one subagent turn under the task call: {timelines:#?}"
    );
    let Some(sub) = subs.first() else {
        return;
    };
    assert!(
        sub.spans.iter().any(|span| span.kind == SpanKind::Model),
        "{sub:#?}"
    );
    let collapsed = render(timelines, turn, false);
    let expanded = render(timelines, turn, true);
    assert!(collapsed.contains('↳'), "{collapsed}");
    assert!(
        expanded.lines().count() > collapsed.lines().count(),
        "{expanded}"
    );
}
