use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::message::{ToolCall, ToolFunction, ToolName};
use rig_harness::rig_core::streaming::{Part, PartKind, StreamEvent};

use super::*;

/// The agent's status after a frame.
fn status(app: &mut App, agent: Entity) -> Option<ActivityStatus> {
    app.update();
    let activity = app.world().get::<Activity>(agent);
    activity.map(|activity| activity.status.clone())
}

#[test]
fn a_question_and_a_plugins_work_outside_the_turn_show() {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, ActivityPlugin));
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    let review = EffectRequest {
        kind: EffectKind::Custom {
            kind: "review".into(),
            payload: serde_json::Value::Null,
        },
        handler: None,
    };
    let review = world
        .spawn((review, ForAgent(agent), Name::new("reviewing")))
        .id();
    let reviewing = Some(ActivityStatus::Busy("reviewing".to_owned()));
    assert_eq!(status(&mut app, agent), reviewing);

    let ask = Ask::yes_no("Keep the changes?");
    let ask = app.world_mut().spawn((ask, ForAgent(agent))).id();
    let asking = Some(ActivityStatus::Asking("Keep the changes?".to_owned()));
    assert_eq!(status(&mut app, agent), asking);

    let yes = Answer::choice("yes").outcome();
    app.world_mut().entity_mut(ask).insert(yes);
    assert_eq!(status(&mut app, agent), reviewing);
    app.world_mut()
        .entity_mut(review)
        .insert(Outcome::custom("done"));
    assert_eq!(status(&mut app, agent), Some(ActivityStatus::Idle));
}

#[test]
fn a_tool_calls_activity_names_its_holds_and_its_progress() {
    let mut app = App::new();
    app.add_plugins(ActivityPlugin);
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    let turn = world.spawn(TurnOf(agent)).id();
    let Ok(name) = ToolName::new("edit") else {
        return;
    };
    let call = ToolCall::from_wire("c1", ToolFunction::new(name, serde_json::json!({})));
    let call = world.spawn((ToolRequest { call }, CallOf(turn))).id();
    world.spawn((Name::new("waiting for lease on src/a.rs"), HoldOf(call)));
    world.spawn(HoldOf(call));
    app.update();
    let activity = app.world().get::<Activity>(agent);
    let held_by = activity
        .and_then(|activity| activity.tools.first())
        .map(|tool| &tool.held_by);
    let names = ["waiting for lease on src/a.rs", UNNAMED_HOLD].map(str::to_owned);
    assert_eq!(held_by, Some(&names.to_vec()));

    let progress = ToolProgress("12 of 40 files".to_owned());
    app.world_mut().entity_mut(call).insert(progress);
    app.update();
    let activity = app.world().get::<Activity>(agent);
    let progress = activity
        .and_then(|activity| activity.tools.first())
        .and_then(|tool| tool.progress.as_deref());
    assert_eq!(progress, Some("12 of 40 files"));
}

#[test]
fn an_inline_question_leaves_a_running_turn_its_status() {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, ActivityPlugin));
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    let ask = Ask::yes_no("Try the next step?").inline();
    world.spawn((ask, ForAgent(agent)));
    let asking = Some(ActivityStatus::Asking("Try the next step?".to_owned()));
    assert_eq!(status(&mut app, agent), asking);

    app.world_mut().spawn(TurnOf(agent));
    assert_eq!(status(&mut app, agent), Some(ActivityStatus::Thinking));
}

#[test]
fn a_turn_is_writing_once_its_answer_streams() {
    let mut app = App::new();
    app.add_plugins(ActivityPlugin);
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    let turn = world.spawn(TurnOf(agent)).id();
    let call = world.spawn((Partial::default(), CallOf(turn))).id();
    assert_eq!(status(&mut app, agent), Some(ActivityStatus::Thinking));

    let reasoning = Partial {
        text: " \n".to_owned(),
        reasoning: "Weighing it.".to_owned(),
        calls: Vec::new(),
    };
    app.world_mut().entity_mut(call).insert(reasoning);
    assert_eq!(status(&mut app, agent), Some(ActivityStatus::Thinking));
    let answer = Partial {
        text: "Hello".to_owned(),
        reasoning: String::new(),
        calls: Vec::new(),
    };
    app.world_mut().entity_mut(call).insert(answer);
    assert_eq!(status(&mut app, agent), Some(ActivityStatus::Writing));
    // A tool call streaming its arguments is written too, text or none.
    let Ok(write) = ToolName::new("write") else {
        return;
    };
    let mut writing = Partial::default();
    writing.follow(&StreamEvent::Start {
        part: Part::new(0),
        kind: PartKind::ToolCall,
        name: Some(write),
    });
    app.world_mut().entity_mut(call).insert(writing);
    assert_eq!(status(&mut app, agent), Some(ActivityStatus::Writing));

    // The next model call has written nothing yet.
    let world = app.world_mut();
    world.entity_mut(call).insert(Settled);
    let next = world.spawn((Partial::default(), CallOf(turn))).id();
    assert_eq!(status(&mut app, agent), Some(ActivityStatus::Thinking));

    // A tool running outranks an answer streaming beside it.
    let streaming = Partial {
        text: "Reading it.".to_owned(),
        reasoning: String::new(),
        calls: Vec::new(),
    };
    app.world_mut().entity_mut(next).insert(streaming);
    let Ok(name) = ToolName::new("read") else {
        return;
    };
    let request = ToolCall::from_wire("c1", ToolFunction::new(name, serde_json::json!({})));
    app.world_mut()
        .spawn((ToolRequest { call: request }, CallOf(turn)));
    assert_eq!(status(&mut app, agent), Some(ActivityStatus::RunningTools));
}

#[cfg(feature = "tui")]
#[test]
fn the_status_line_shows_how_long_a_step_has_lasted() {
    let mut app = App::new();
    // Bevy's real clock, stepped by hand.
    app.add_plugins(ActivityPlugin)
        .init_resource::<Time<Real>>()
        .register_required_components::<Agent, rig_tui::StatusItems>();
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    let turn = world.spawn(TurnOf(agent)).id();
    let call = world.spawn((Partial::default(), CallOf(turn))).id();
    let shown = |app: &mut App, after: u64| {
        let world = app.world_mut();
        let mut time = world.resource_mut::<Time<Real>>();
        time.update_with_duration(Duration::from_secs(after));
        app.update();
        let items = app.world().get::<rig_tui::StatusItems>(agent);
        let item = items.and_then(|items| items.iter().find(|item| item.name == "status"));
        // Without its working wave.
        let bars = |c: char| ('▁'..='█').contains(&c) || c == ' ';
        item.map(|item| item.text.trim_start_matches(bars).to_owned())
    };
    // A short step shows no clock, so it does not flicker.
    assert_eq!(shown(&mut app, 0).as_deref(), Some("thinking…"));
    assert_eq!(shown(&mut app, 4).as_deref(), Some("thinking…"));
    assert_eq!(shown(&mut app, 1).as_deref(), Some("thinking… 5s"));
    assert_eq!(shown(&mut app, 65).as_deref(), Some("thinking… 1m10s"));

    // The clock counts from when the status changed.
    let answer = Partial {
        text: "Hello".to_owned(),
        reasoning: String::new(),
        calls: Vec::new(),
    };
    app.world_mut().entity_mut(call).insert(answer);
    assert_eq!(shown(&mut app, 0).as_deref(), Some("writing…"));
    assert_eq!(shown(&mut app, 8).as_deref(), Some("writing… 8s"));
    assert_eq!(shown(&mut app, 235).as_deref(), Some("writing… 4m03s"));
    assert_eq!(shown(&mut app, 3357).as_deref(), Some("writing… 1h00m"));

    app.world_mut().entity_mut(turn).despawn();
    assert_eq!(shown(&mut app, 60).as_deref(), Some("idle"));
}

/// An app with the terminal view's activity, `settings` as its
/// settings.toml, and a thinking shown agent; with the agent and its turn.
#[cfg(feature = "tui")]
fn working(settings: &str) -> (App, Entity, Entity) {
    let mut app = App::new();
    app.insert_resource(rig_harness::settings::Settings::from_toml(settings))
        .add_plugins(ActivityPlugin)
        .init_resource::<rig_harness::main_agent::ShownAgent>()
        .register_required_components::<Agent, rig_tui::StatusItems>();
    animate(&mut app);
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    world
        .resource_mut::<rig_harness::main_agent::ShownAgent>()
        .0 = Some(agent);
    let turn = world.spawn(TurnOf(agent)).id();
    world.spawn((Partial::default(), CallOf(turn)));
    (app, agent, turn)
}

/// Steps Bevy's real clock by `after`, runs a frame, and returns `agent`'s
/// status item and how many [`KeepAwake`]s live.
#[cfg(feature = "tui")]
fn step(app: &mut App, agent: Entity, after: Duration) -> (Option<String>, usize) {
    let world = app.world_mut();
    world
        .resource_mut::<Time<Real>>()
        .update_with_duration(after);
    app.update();
    let items = app.world().get::<rig_tui::StatusItems>(agent);
    let item = items.and_then(|items| items.iter().find(|item| item.name == "status"));
    let text = item.map(|item| item.text.clone());
    let world = app.world_mut();
    let awake = world.query::<&KeepAwake>().iter(world).count();
    (text, awake)
}

#[cfg(feature = "tui")]
#[test]
fn a_working_agents_status_waves_while_the_loop_stays_awake() {
    let (mut app, agent, turn) = working("");
    let (first, awake) = step(&mut app, agent, Duration::ZERO);
    assert_eq!(awake, 1);
    let (second, awake) = step(&mut app, agent, STEP);
    assert_eq!(awake, 1, "one KeepAwake, not one a step");
    assert_ne!(first, second, "the wave moves a step");
    let label = |text: &Option<String>| text.as_deref().is_some_and(|t| t.ends_with(" thinking…"));
    assert!(label(&first) && label(&second), "{first:?} {second:?}");

    // A question stops it; work after the answer starts it again.
    let ask = Ask::yes_no("Keep it?");
    let ask = app.world_mut().spawn((ask, ForAgent(agent))).id();
    let (asking, awake) = step(&mut app, agent, STEP);
    assert_eq!((asking.as_deref(), awake), (Some("asking: Keep it?"), 0));
    app.world_mut().entity_mut(ask).despawn();
    let (_, awake) = step(&mut app, agent, STEP);
    assert_eq!(awake, 1);

    app.world_mut().entity_mut(turn).despawn();
    let (idle, awake) = step(&mut app, agent, STEP);
    assert_eq!((idle.as_deref(), awake), (Some("idle"), 0));
}

#[cfg(feature = "tui")]
#[test]
fn with_animate_off_the_wave_is_still_and_the_loop_sleeps() {
    let (mut app, agent, _) = working("[activity]\nanimate = false\n");
    let (first, awake) = step(&mut app, agent, Duration::ZERO);
    let (second, _) = step(&mut app, agent, STEP);
    assert_eq!(first, second);
    assert_eq!(awake, 0);
    // Still a wave, before the label.
    assert!(first.is_some_and(|text| text.ends_with(" thinking…") && text != "thinking…"));
}

/// A run whose front is not the terminal view, such as `rig -p`, keeps
/// no animation awake.
#[cfg(feature = "tui")]
#[test]
fn another_front_animates_nothing() {
    let mut app = App::new();
    app.insert_resource(rig_harness::settings::Settings::default())
        .add_plugins(ActivityPlugin)
        .init_resource::<rig_harness::main_agent::ShownAgent>();
    app.finish();
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    world
        .resource_mut::<rig_harness::main_agent::ShownAgent>()
        .0 = Some(agent);
    world.spawn(TurnOf(agent));
    let (_, awake) = step(&mut app, agent, STEP);
    assert_eq!(awake, 0);
}
