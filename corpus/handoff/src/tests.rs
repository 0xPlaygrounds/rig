use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, connect, reply, run_until};
use rig_telemetry::{Spending, UsagePlugin};

use super::*;

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// Delivers `text` to `agent` as the user typed it, then runs frames until
/// `ended` turns ended in all.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, ended: usize) {
    let typed = Deliver::new(agent, text, DeliveryMode::Steer);
    app.world_mut().trigger(typed);
    run_until(app, wakes, |world| world.resource::<Ended>().0 >= ended);
}

#[test]
fn handoff_now_starts_a_fresh_agent_from_the_brief_alone() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin, HandoffPlugin, UsagePlugin))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        reply("answer one"),
        reply("answer two"),
        reply("answer three"),
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
    let a = world.query_filtered::<Entity, With<Agent>>().single(world);
    let Ok(a) = a else {
        return;
    };
    world.entity_mut(a).insert((connection, choice.clone()));

    let asked = ["question one", "question two", "question three"];
    for (turn, question) in asked.into_iter().enumerate() {
        say(&mut app, &wakes, a, question, turn + 1);
    }
    // The handoff's own turn, then the new agent's first.
    say(&mut app, &wakes, a, "/handoff --now implement API", 5);

    let world = app.world_mut();
    let mut roots =
        world.query_filtered::<(Entity, Has<ActiveTurn>), (With<Agent>, Without<SpawnedBy>)>();
    let roots: Vec<(Entity, bool)> = roots.iter(world).collect();
    assert_eq!(roots.len(), 2, "{roots:?}");
    assert!(roots.contains(&(a, false)), "A stays, idle: {roots:?}");
    let b = roots
        .iter()
        .map(|(agent, _)| *agent)
        .find(|&agent| agent != a);
    let Some(b) = b else {
        return;
    };
    let from = world.get::<HandedOffFrom>(b).map(|from| from.agent.clone());
    assert_eq!(from.as_ref(), world.get::<AgentId>(a));
    assert_eq!(world.get::<ModelChoice>(b), Some(&choice));

    let requests = model.requests();
    assert_eq!(requests.len(), 5, "three turns, the brief, B's first");
    let json = |at: usize| {
        let request = requests.get(at);
        serde_json::to_string(&request.map(|request| &request.chat_history)).unwrap_or_default()
    };
    let (brief, first) = (json(3), json(4));
    assert!(
        brief.contains("implement API") && brief.contains("answer three"),
        "{brief}"
    );
    let last = requests
        .get(4)
        .and_then(|request| request.chat_history.last())
        .map(|message| serde_json::to_string(message).unwrap_or_default());
    assert!(
        last.is_some_and(|last| last.contains("Brief: files x.rs")),
        "{first}"
    );
    for earlier in asked.into_iter().chain(["answer one", "answer three"]) {
        assert!(!first.contains(earlier), "{earlier} reached B: {first}");
    }
    // The brief's call counts in A's usage, beside its three turns.
    let spent = world.get::<Spending>(a).map(|spent| spent.0.calls);
    assert_eq!(spent, Some(4));
}
