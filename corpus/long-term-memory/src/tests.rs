use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::journal::JournalPlugin;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// A session on a fresh store with the memory plugin keeping its files
/// under `dir`, whose first agent `model` answers, after its first frame.
fn session(
    dir: &Path,
    turns: Vec<Vec<MockStreamEvent>>,
) -> (App, Receiver<()>, MockCompletionModel) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(MemoryFiles::new(&dir.join("home"), &dir.join("project")))
        .add_plugins((AgentPlugin, JournalPlugin, LongTermMemoryPlugin))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns(turns);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    if let (Some(agent), Some(connection)) = (agent, connect(&model)) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, model)
}

/// The user types `text` to the first agent; frames run until the turn
/// ended.
fn ask(app: &mut App, wakes: &Receiver<()>, text: &str) {
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let Some(agent) = agent else {
        return;
    };
    world.trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| world.resource::<Ended>().0 > 0);
}

#[test]
fn a_fact_remembered_in_one_session_is_recalled_in_the_next_but_not_saved() {
    let base = std::env::var_os("CARGO_TARGET_DIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join(format!("corpus-long-term-memory-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();

    let fact = serde_json::json!({ "fact": "Run tests with cargo nextest", "scope": "project" });
    let (mut first, wakes, _model) =
        session(&dir, vec![calls("1", "remember", fact), reply("Noted.")]);
    ask(
        &mut first,
        &wakes,
        "Remember that tests run with cargo nextest.",
    );
    let stored = fs::read_to_string(dir.join("project/.rig/memory.md")).unwrap_or_default();
    assert!(
        stored.contains("- Run tests with cargo nextest"),
        "{stored}"
    );
    drop(first);

    let (mut second, wakes, model) = session(&dir, vec![reply("Like this.")]);
    ask(&mut second, &wakes, "how do I run the tests?");
    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert!(
        sent.first()
            .is_some_and(|request| request.contains("<memory>\\nWhat you remember")
                && request.contains("] Run tests with cargo nextest\\n</memory>")),
        "{sent:?}"
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
    fs::remove_dir_all(&dir).ok();
}
