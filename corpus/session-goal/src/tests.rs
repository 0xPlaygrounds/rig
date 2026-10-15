use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{app_on, connect, reply, run_until};

/// What the commands replied, in order.
#[derive(Resource, Default)]
struct Replies(Vec<String>);

fn record_replies(mut notices: MessageReader<Notice>, mut replies: ResMut<Replies>) {
    replies
        .0
        .extend(notices.read().map(|notice| notice.text.clone()));
}

/// The kernel and the plugin on `store`, after the first frame restored the
/// session, with its one agent answered by `model`.
fn start(store: &MemoryStore, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin, SessionGoalPlugin))
        .init_resource::<Replies>()
        .add_systems(Last, record_replies);
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

/// The goal's text in this session's `/resume` line.
fn resume_label(app: &mut App) -> String {
    let world = app.world_mut();
    let mut labels = world.query::<&ResumeLabel>();
    let texts: Vec<String> = labels
        .iter(world)
        .filter(|label| label.tag == "goal")
        .map(|label| label.text.clone())
        .collect();
    texts.join("|")
}

/// Runs `/line` for `agent` and returns its reply.
fn command(app: &mut App, agent: Entity, line: &str) -> String {
    app.world_mut()
        .trigger(RunCommand::new(agent, line.to_owned()));
    app.update();
    let replies = &mut app.world_mut().resource_mut::<Replies>().0;
    std::mem::take(replies).join("\n")
}

/// Asks `agent` something and returns the request the model got, as JSON.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, model: &MockCompletionModel) -> String {
    let before = model.request_count();
    let ask = Deliver::new(agent, "What next?", DeliveryMode::Queue);
    app.world_mut().trigger(ask);
    run_until(app, wakes, |world| {
        model.request_count() > before && !world.entity(agent).contains::<ActiveTurn>()
    });
    let requests = model.requests();
    let request = requests.last();
    request
        .and_then(|request| serde_json::to_string(request).ok())
        .unwrap_or_default()
}

#[test]
fn the_goal_is_in_every_request_survives_a_restart_and_is_absent_in_a_new_session() {
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([reply("On it.")]);
    let (mut app, wakes, agent) = start(&store, &model);
    let set = command(&mut app, agent, "goal Ship   OAuth");
    assert_eq!(set, "Session goal set: Ship OAuth");
    let preamble = ask(&mut app, &wakes, agent, &model);
    assert!(preamble.contains("Session goal: Ship OAuth"), "{preamble}");
    let status = app.world().resource::<AppStatus>().0.clone();
    assert!(
        status
            .iter()
            .any(|item| item.text == "◎ Ship OAuth" && item.min_width > 0),
        "{status:?}"
    );
    assert_eq!(resume_label(&mut app), "◎ Ship OAuth");
    drop(app);

    // A restart on the same session brings it back.
    let model = MockCompletionModel::from_stream_turns([reply("Still on it.")]);
    let (mut app, wakes, agent) = start(&store, &model);
    assert_eq!(command(&mut app, agent, "goal"), "Session goal: Ship OAuth");
    assert_eq!(resume_label(&mut app), "◎ Ship OAuth");
    let preamble = ask(&mut app, &wakes, agent, &model);
    assert!(preamble.contains("Session goal: Ship OAuth"), "{preamble}");
    // Cleared, the next request has no goal.
    assert_eq!(
        command(&mut app, agent, "goal clear"),
        "Session goal cleared."
    );
    let model_after = MockCompletionModel::from_stream_turns([reply("Fine.")]);
    if let Some(connection) = connect(&model_after) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    let preamble = ask(&mut app, &wakes, agent, &model_after);
    assert!(!preamble.contains("Session goal"), "{preamble}");
    assert_eq!(resume_label(&mut app), "");
    drop(app);

    // `/new` runs a session on a store of its own: no goal there.
    let model = MockCompletionModel::from_stream_turns([reply("Hello.")]);
    let (mut app, wakes, agent) = start(&MemoryStore::default(), &model);
    let shown = command(&mut app, agent, "goal");
    assert_eq!(shown, "No session goal; /goal <text> sets one.");
    let preamble = ask(&mut app, &wakes, agent, &model);
    assert!(!preamble.contains("Session goal"), "{preamble}");
}
