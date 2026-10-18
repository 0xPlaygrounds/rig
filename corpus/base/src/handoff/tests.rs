use std::sync::mpsc::Receiver;

use rig_basics::telemetry::usage::{AgentUsage, UsagePlugin};
use rig_core::completion::Message;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, app_on, commit_message, connect, log_ended, reply, run_until,
};

use super::*;

/// What agent A was asked and answered before the handoff.
const EARLIER: [&str; 6] = [
    "question one",
    "answer one",
    "question two",
    "answer two",
    "question three",
    "answer three",
];

/// The kernel, the handoff and usage plugins on `store`, after the first
/// frame restored the session, with its loop's wakes.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin, HandoffPlugin, UsagePlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    (app, wakes)
}

#[test]
fn handoff_now_starts_a_fresh_agent_from_the_brief_alone() {
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let model = MockCompletionModel::from_stream_turns([
        reply("Brief: files x.rs; goal API"),
        reply("On it."),
    ]);
    let connection = connect(&model);
    assert!(connection.is_some(), "the catalog lists the model");
    let Some(connection) = connection else {
        return;
    };
    let choice = ModelChoice(connection.spec.reference());
    let world = app.world_mut();
    let Ok(a) = world.query_filtered::<Entity, With<Agent>>().single(world) else {
        return;
    };
    world.entity_mut(a).insert((connection, choice.clone()));
    for (at, text) in EARLIER.into_iter().enumerate() {
        let message = match at % 2 {
            0 => Message::user(text),
            _ => Message::assistant(text),
        };
        let added = commit_message(world, a, message);
        assert!(added.is_ok(), "{added:?}");
    }

    let handoff = Deliver::new(a, "/handoff --now implement API", DeliveryMode::Steer);
    app.world_mut().trigger(handoff);
    // The handoff's own turn, then the new agent's first.
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 2
    });

    let world = app.world_mut();
    let mut roots =
        world.query_filtered::<(Entity, Has<ActiveTurn>), (With<Agent>, Without<SpawnedBy>)>();
    let roots: Vec<(Entity, bool)> = roots.iter(world).collect();
    assert_eq!(roots.len(), 2, "{roots:?}");
    assert!(roots.contains(&(a, false)), "A stays, idle: {roots:?}");
    let Some(b) = roots
        .iter()
        .map(|&(agent, _)| agent)
        .find(|&agent| agent != a)
    else {
        return;
    };
    let from = world.get::<HandedOffFrom>(b).map(|from| from.agent.clone());
    assert_eq!(from.as_ref(), world.get::<AgentId>(a));
    assert_eq!(world.get::<ModelChoice>(b), Some(&choice));
    assert_eq!(world.resource::<ShownAgent>().0, Some(b), "B is shown");
    // B's first message is the brief, by A, about the goal.
    let author = world
        .get::<Conversation>(b)
        .and_then(|conversation| conversation.author(0).cloned());
    let by_a = world
        .get::<AgentId>(a)
        .map(|a| Author::agent(a.0.clone(), None));
    assert_eq!(author, by_a.map(|by_a| by_a.titled("implement API")));
    let b_id = world.get::<AgentId>(b).cloned();

    let requests = model.requests();
    let json = |at: usize| {
        let request = requests.get(at).map(|request| &request.chat_history);
        serde_json::to_string(&request).unwrap_or_default()
    };
    assert_eq!(requests.len(), 2, "the brief, then B's first");
    let (brief, first) = (json(0), json(1));
    assert!(
        brief.contains("implement API") && brief.contains("answer three"),
        "{brief}"
    );
    let last = requests
        .get(1)
        .and_then(|request| request.chat_history.last())
        .map(|message| serde_json::to_string(message).unwrap_or_default());
    assert!(
        last.is_some_and(|last| last.contains("Brief: files x.rs")),
        "{first}"
    );
    for earlier in EARLIER {
        assert!(!first.contains(earlier), "{earlier} reached B: {first}");
    }
    // The brief's call counts in A's usage, as a handoff.
    let spent = world
        .get::<AgentUsage>(a)
        .and_then(|spent| spent.purposes.get(PURPOSE));
    assert_eq!(spent.map(|spent| spent.calls), Some(1));
    drop(app);

    // A restart shows B again, named after its goal and A.
    let (mut restarted, _wakes) = start(&store);
    let world = restarted.world_mut();
    let shown = world.resource::<ShownAgent>().0;
    let shown = shown.and_then(|agent| world.get::<AgentId>(agent)).cloned();
    assert!(shown.is_some() && shown == b_id, "{shown:?}");
    let named = world
        .query_filtered::<&Name, With<HandedOffFrom>>()
        .iter(world)
        .map(|name| name.as_str().to_owned())
        .collect::<Vec<_>>();
    assert!(
        named.len() == 1
            && named
                .iter()
                .all(|name| name.starts_with("implement API (from ")),
        "{named:?}"
    );
}

/// Starts an agent handed off from `from`, once, as `take_brief` does in
/// the agent loop.
fn hand_off_once(
    mut done: Local<bool>,
    from: Query<&AgentId, With<Agent>>,
    mut commands: Commands,
) {
    let Some(agent) = from.iter().next().filter(|_| !*done).cloned() else {
        return;
    };
    *done = true;
    let goal = "next".to_owned();
    commands.spawn((Agent, HandedOffFrom { agent, goal }));
}

#[test]
fn a_successor_started_in_the_agent_loop_is_shown_that_frame() {
    let store = MemoryStore::default();
    let (mut app, _wakes) = start(&store);
    app.add_systems(AgentPass, hand_off_once.in_set(Stage::Commit));
    // Anything with an effect runs the loop.
    app.world_mut().spawn(Effect::default());
    app.update();

    let world = app.world_mut();
    let successor = world
        .query_filtered::<Entity, With<HandedOffFrom>>()
        .single(world)
        .ok();
    assert!(successor.is_some());
    assert_eq!(world.resource::<ShownAgent>().0, successor);
}
