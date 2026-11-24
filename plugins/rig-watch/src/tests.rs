use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    MemoryStore, app_on, calls, connect, pass_wall_clock, reply, run_until, scratch_dir,
};
use serde_json::json;

use super::*;

fn watch(label: &str, kind: Kind, path: &str, text: &str) -> Watch {
    Watch {
        label: label.to_owned(),
        kind,
        path: path.to_owned(),
        text: text.to_owned(),
    }
}

fn add(add: Vec<Watch>) -> Args {
    Args {
        add,
        every_minutes: Some(10),
        done: false,
    }
}

#[test]
fn a_watch_fires_once_and_nothing_fired_is_said_after_its_minutes() {
    let dir = scratch_dir("rig-watch-fires-once");
    let (title, report) = (dir.join("title"), dir.join("report"));
    let (title, report) = (title.to_string_lossy(), report.to_string_lossy());
    assert!(fs::write(&*title, "rig · working").is_ok());
    let mut watching = Watching::default();
    let watches = vec![
        watch("idle", Kind::Has, &title, "idle"),
        watch("report", Kind::Exists, &report, ""),
    ];
    assert!(apply(&mut watching, add(watches), 1000).is_ok());
    assert_eq!(wake(&mut watching, 1005), None, "nothing holds yet");

    assert!(fs::write(&*title, "rig · idle").is_ok());
    let said = wake(&mut watching, 1010).unwrap_or_default();
    let top = format!(
        "[watch] {}\nFired:\n  idle: has {title} \"idle\"\n",
        local(1010)
    );
    assert!(said.starts_with(&top), "{said}");
    assert!(
        said.contains(&format!("Still watched:\n  report: exists {report}\n")),
        "{said}"
    );
    assert_eq!(wake(&mut watching, 1015), None, "it fires once");

    assert_eq!(
        wake(&mut watching, 1609),
        None,
        "not ten minutes since the last wake"
    );
    let said = wake(&mut watching, 1610).unwrap_or_default();
    assert!(said.contains("\nNothing fired in 10 min"), "{said}");

    // The last wake ends the arming.
    watching.wakes_left = 1;
    assert!(fs::write(&*report, "").is_ok());
    let said = wake(&mut watching, 1700).unwrap_or_default();
    assert!(said.contains("last wake"), "{said}");
    assert_eq!(watching, Watching::default());
    fs::remove_dir_all(dir).ok();
}

#[test]
fn a_call_with_a_watch_that_cannot_be_taken_changes_nothing_and_done_ends_it() {
    let mut watching = Watching::default();
    for bad in [
        watch("x", Kind::Exists, "rel", ""),
        watch("x", Kind::Has, "/a", ""),
    ] {
        let call = add(vec![watch("ok", Kind::Gone, "/a", ""), bad]);
        assert!(apply(&mut watching, call, 0).is_err());
        assert_eq!(watching, Watching::default());
    }
    let missing = watch("x", Kind::Lacks, "/no/such/file", "y");
    assert!(missing.holds(), "a file that is not there lacks every text");
    assert!(apply(&mut watching, add(vec![missing]), 0).is_ok());
    assert!(watching.active);
    let done = Args {
        done: true,
        ..Default::default()
    };
    assert!(apply(&mut watching, done, 0).is_ok());
    assert_eq!(watching, Watching::default());
}

/// The kernel and the plugin on `store`, after the first frame restored the
/// session, with its one agent answered by `model`.
fn start(store: &MemoryStore, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, WatchPlugin));
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Runs frames until `model` was asked `requests` times and `agent` is in
/// no turn.
#[track_caller]
fn settle(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    model: &MockCompletionModel,
    requests: usize,
) {
    run_until(app, wakes, |world| {
        model.request_count() >= requests && !world.entity(agent).contains::<ActiveTurn>()
    });
}

/// The time to the next look passes, and the look is made.
fn later(app: &mut App) {
    app.update();
    pass_wall_clock(app.world_mut(), LOOK);
    app.update();
    app.update();
}

#[test]
fn an_idle_agent_is_woken_by_the_file_it_watches_also_after_a_restart() {
    let dir = scratch_dir("rig-watch-wakes");
    let report = dir.join("report");
    let report = report.to_string_lossy();
    let store = MemoryStore::default();
    let arming = json!({ "add": [{ "label": "report", "kind": "exists", "path": &*report }] });
    let model = MockCompletionModel::from_stream_turns([
        calls("call-1", NAME, arming),
        reply("Armed."),
        reply("Seen."),
    ]);
    let (mut app, wakes, agent) = start(&store, &model);
    let ask = Deliver::new(agent, "Wait for the report.", DeliveryMode::Queue);
    app.world_mut().trigger(ask);
    settle(&mut app, &wakes, agent, &model, 2);
    let watching = app
        .world()
        .get::<Watching>(agent)
        .cloned()
        .unwrap_or_default();
    assert!(watching.active, "{watching:?}");

    // Time passes with no report: the agent is left alone.
    later(&mut app);
    assert_eq!(model.request_count(), 2);

    // The session is started again, the report is written, and the agent
    // is woken with no word from the user.
    drop(app);
    let (mut app, wakes, agent) = start(&store, &model);
    assert_eq!(app.world().get::<Watching>(agent), Some(&watching));
    assert!(fs::write(&*report, "done").is_ok());
    later(&mut app);
    settle(&mut app, &wakes, agent, &model, 3);
    let asked = format!("{:?}", model.requests().last());
    assert!(
        asked.contains("[watch] ") && asked.contains("report: exists"),
        "{asked}"
    );
    assert_eq!(
        app.world().get::<Watching>(agent).map(|w| w.watches.len()),
        Some(0)
    );
    fs::remove_dir_all(dir).ok();
}
