use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until, scratch_home,
};

/// A session on a fresh store in the rig home `home`, its project beside
/// it, whose first agent `turns` answer, after its first frame.
fn session(
    home: &Home,
    turns: Vec<Vec<MockStreamEvent>>,
) -> (App, Receiver<()>, MockCompletionModel) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(RigHome(home.clone()))
        .insert_resource(ProjectRoot(home.root().join("project")))
        .add_plugins((AgentPlugin, SessionLogPlugin, LongTermMemoryPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns(turns);
    let world = app.world_mut();
    if let (Some(agent), Some(connection)) = (first_agent(world), connect(&model)) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, model)
}

/// The user types `text` to the first agent; frames run until the turn
/// ended.
fn ask(app: &mut App, wakes: &Receiver<()>, text: &str) {
    let Some(agent) = first_agent(app.world_mut()) else {
        return;
    };
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| !world.resource::<Ended>().0.is_empty());
}

#[test]
fn a_fact_remembered_in_one_session_is_recalled_in_the_next_but_not_saved() {
    let home = scratch_home("corpus-long-term-memory");

    let fact = serde_json::json!({ "fact": "Run tests with cargo nextest", "scope": "project" });
    let (mut first, wakes, _model) =
        session(&home, vec![calls("1", "remember", fact), reply("Noted.")]);
    ask(
        &mut first,
        &wakes,
        "Remember that tests run with cargo nextest.",
    );
    let stored = fs::read_to_string(home.root().join("project/.rig/memory.md")).unwrap_or_default();
    assert!(
        stored.contains("- Run tests with cargo nextest"),
        "{stored}"
    );
    drop(first);

    let (mut second, wakes, model) = session(&home, vec![reply("Like this.")]);
    ask(&mut second, &wakes, "how do I run the tests?");
    let requests = model.requests();
    // The block is in the turn the cache point ends before.
    let carrying = requests.first().and_then(|request| {
        let turns = request.cache_points.iter().find_map(|point| match point {
            CachePoint::Messages(turns) => Some(*turns),
            _ => None,
        })?;
        let mut conversation = request
            .chat_history
            .iter()
            .filter(|message| !matches!(&***message, message::Message::System { .. }));
        serde_json::to_string(conversation.nth(turns)?).ok()
    });
    assert!(
        carrying
            .as_deref()
            .is_some_and(|message| message.contains("how do I run the tests?")
                && message.contains("<memory>\\nWhat you remember")
                && message.contains("] Run tests with cargo nextest\\n</memory>")),
        "{requests:?}"
    );
    let world = second.world_mut();
    let saved: Vec<String> = world
        .query::<&Conversation>()
        .iter(world)
        .flat_map(|conversation| conversation.messages())
        .map(|message| serde_json::to_string(message).unwrap_or_default())
        .collect();
    assert!(
        saved
            .iter()
            .any(|message| message.contains("how do I run the tests?"))
    );
    assert!(
        !saved.iter().any(|message| message.contains("nextest")),
        "{saved:?}"
    );
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(LongTermMemoryPlugin);
}
