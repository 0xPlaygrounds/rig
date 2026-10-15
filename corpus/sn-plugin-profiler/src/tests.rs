use std::sync::mpsc::{Receiver, channel};

use rig_core::test_utils::MockCompletionModel;
use rig_harness::clock::ClockPlugin;
use rig_harness::harness_protocol::SessionId;
use rig_harness::load;
use rig_harness_test_support::{
    Collected, Ended, app_in, calls, collect, connect, first_agent, log_ended, reply, run_until,
    scratch_home,
};
use rig_harness_test_support::{press, tui_view};
use rig_tui::{AppKeyBindingsExt, KeyBinding, KeyPress, KeyPressed};

use super::*;

/// How long `slowpoke` holds each model request.
const WAIT: Duration = Duration::from_millis(300);

/// Holds every model request for [`WAIT`] before it is sent.
#[derive(Default)]
struct Slowpoke;

impl Plugin for Slowpoke {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, hold_requests.in_set(Stage::Gate))
            .add_observer(let_go);
    }

    fn name(&self) -> &str {
        "slowpoke"
    }
}

/// On a request `slowpoke` held once.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Waited;

/// On `slowpoke`'s holds.
#[derive(Component)]
struct SlowHold;

fn hold_requests(
    requests: Query<Entity, (ReadyToStart, ModelCalls, Without<Waited>)>,
    me: Res<PluginEntity<Slowpoke>>,
    mut commands: Commands,
) {
    for request in &requests {
        commands.entity(request).insert(Waited);
        commands.spawn((
            Name::new("slowpoke: waiting"),
            HoldOf(request),
            WakeAt::after(WAIT),
            SlowHold,
            ProvidedBy(me.entity()),
        ));
    }
}

fn let_go(due: On<Add<Due>>, holds: Query<(), With<SlowHold>>, mut commands: Commands) {
    if holds.contains(due.entity) {
        commands.entity(due.entity).despawn();
    }
}

/// Only counts the tool calls that settled.
#[derive(Default)]
struct Idle;

impl Plugin for Idle {
    fn build(&self, app: &mut App) {
        app.init_resource::<Finished>().add_observer(count);
    }

    fn name(&self) -> &str {
        "idle"
    }
}

#[derive(Resource, Default)]
struct Finished(usize);

fn count(
    settled: On<Add<Settled>>,
    calls: Query<(), With<ToolRequest>>,
    mut finished: ResMut<Finished>,
) {
    if calls.contains(settled.entity) {
        finished.0 += 1;
    }
}

/// Blocks for 200 ms on Ctrl+K.
#[derive(Default)]
struct Keyhog;

impl Plugin for Keyhog {
    fn build(&self, app: &mut App) {
        let binding = KeyBinding::new(KeyPress::ctrl('k'), "Freeze");
        app.add_key_binding(binding, |_: On<KeyPressed>| {
            std::thread::sleep(Duration::from_millis(200));
        });
    }

    fn name(&self) -> &str {
        "keyhog"
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ReadArgs {
    path: String,
}

fn read(called: On<ToolCalled<ReadArgs>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        let text = format!("contents of {}", called.args.path);
        call.decide(Decider::plugin("read"), ToolResult::success(text.into()));
    }
}

/// An app whose log times every system and observer run, as the agent's
/// does, with the session service (which ends each frame's timings), the
/// profiler, `plugins`, and the terminal view, in the home `home`.
fn start(home: &str, plugins: impl FnOnce(&mut App)) -> (App, Receiver<()>) {
    let (mut app, _) = app_in(&scratch_home(home), &SessionId::generate());
    app.add_plugins(HeadlessPlugins.build().disable::<TaskPoolPlugin>());
    // The loop's wakes, in place of the headless runner's.
    let (sender, wakes) = channel();
    app.insert_resource(Wake::new(move || {
        sender.send(()).ok();
    }))
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ClockPlugin,
        PluginProfilerPlugin,
    ))
    .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    plugins(&mut app);
    tui_view(&mut app);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    (app, wakes)
}

fn notices(app: &App) -> Vec<String> {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices.iter().map(|notice| notice.text.clone()).collect()
}

/// Runs the user's turn `n` to its end.
fn turn(app: &mut App, wakes: &Receiver<()>, agent: Entity, n: usize) {
    let message = Deliver::new(agent, format!("Turn {n}."), DeliveryMode::Queue);
    app.world_mut().trigger(message);
    run_until(app, wakes, |world| world.resource::<Ended>().0.len() >= n);
    assert_eq!(app.world().resource::<Ended>().0.len(), n, "turn {n} ended");
}

/// `/perf --json`, read back.
fn report(app: &mut App, agent: Entity) -> PerfReport {
    let before = notices(app).len();
    app.world_mut()
        .trigger(RunCommand::new(agent, "perf --json"));
    app.update();
    let printed = notices(app).into_iter().skip(before).collect::<Vec<_>>();
    let json = printed.first().cloned().unwrap_or_default();
    let report = serde_json::from_str(&json);
    assert!(report.is_ok(), "{printed:?}");
    report.unwrap_or_default()
}

/// Notices naming a plugin that held up a turn.
fn slow_notices(app: &App) -> Vec<String> {
    let notices = notices(app).into_iter();
    notices
        .filter(|text| text.contains("held up this turn"))
        .collect()
}

/// The package the test plugins come from.
const HERE: &str = "corpus-sn-plugin-profiler";

#[test]
fn names_the_plugin_that_holds_up_turns() {
    let reads = |id: &str| calls(id, "read", serde_json::json!({ "path": "a.txt" }));
    let model = MockCompletionModel::from_stream_turns([
        reads("r1"),
        reply("one"),
        reads("r2"),
        reply("two"),
        reads("r3"),
        reply("three"),
        reads("r4"),
        reply("four"),
    ]);
    let (mut app, wakes) = start("perf-holds", |app| {
        // Two plugins of one package, told apart.
        load::<Slowpoke>(app, HERE, "test");
        load::<Idle>(app, HERE, "test");
    });
    let (agent, connection) = (first_agent(app.world_mut()), connect(&model));
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent on the scripted model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);

    // Two turns of a `read` call and an answer: four requests, each held.
    turn(&mut app, &wakes, agent, 1);
    turn(&mut app, &wakes, agent, 2);
    assert_eq!(app.world().resource::<Finished>().0, 2);
    let session = report(&mut app, agent).session;
    let slowpoke = session.get("slowpoke").cloned().unwrap_or_default();
    assert!(slowpoke.held_up_ms >= 1200, "{slowpoke:?}");
    // Its four holds; its systems ran besides.
    assert_eq!(slowpoke.calls, 4, "{slowpoke:?}");
    assert!(slowpoke.runs > 0, "{slowpoke:?}");
    let slowest = slowpoke.slowest.unwrap_or(Slowest {
        ms: 0,
        handling: String::new(),
    });
    assert!((300..700).contains(&slowest.ms), "{slowest:?}");
    assert!(
        slowest.handling.starts_with("before request #"),
        "{slowest:?}"
    );
    // Only `idle`'s own observer runs are charged to it, and held up little.
    let idle = session.get("idle").cloned().unwrap_or_default();
    assert!(idle.runs >= 2 && idle.calls == 0, "{session:?}");
    assert!(idle.held_up_ms < 50, "{session:?}");
    assert!(slow_notices(&app).is_empty(), "600 ms a turn is under 1 s");

    // Over 500 ms in a turn is named once a session.
    app.world_mut().resource_mut::<PerfSettings>().threshold_ms = 500;
    turn(&mut app, &wakes, agent, 3);
    let named = slow_notices(&app);
    assert_eq!(named.len(), 1, "{named:?}");
    assert!(
        named.iter().all(|text| text.starts_with("slowpoke ")),
        "{named:?}"
    );
    let last_turn = report(&mut app, agent).last_turn;
    let in_turn = last_turn
        .get("slowpoke")
        .map_or(0, |slowpoke| slowpoke.held_up_ms);
    assert!(in_turn >= 600, "{last_turn:?}");
    turn(&mut app, &wakes, agent, 4);
    assert_eq!(slow_notices(&app).len(), 1);
}

#[test]
fn names_the_plugin_whose_key_handler_stalls_the_interface() {
    let (mut app, _) = start("perf-stall", |app| load::<Keyhog>(app, HERE, "test"));
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    press(&mut app, "ctrl+k");
    // The harness hands over a frame's runs in the next one.
    app.update();
    let stalls = report(&mut app, agent).stalls;
    let stall = stalls.iter().find(|stall| stall.plugin == "keyhog");
    assert!(stall.is_some_and(|stall| stall.ms >= 150), "{stalls:?}");
    let handling = stall.and_then(|stall| stall.handling.clone());
    assert_eq!(handling.as_deref(), Some("key ctrl+k"), "{stalls:?}");
}
