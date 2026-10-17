use std::path::PathBuf;

use rig_basics::coding_tools::{EditTool, ReadTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, connect, log_ended, reply, run_until,
};
use serde_json::json;

use bevy_ecs::system::RunSystemOnce;

use super::*;
use rig_harness::rig_memory::summary_author;

/// An app on `store` with the file tools and the plugin, its agent talking
/// to `model`, after its first frame.
fn app(
    store: &MemoryStore,
    model: &MockCompletionModel,
) -> (App, std::sync::mpsc::Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((AgentPlugin, SessionLogPlugin, ReadTool, EditTool, WriteTool))
        .add_plugins(ReadBeforeEditPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let agent = agent.unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Sends `text` to `agent` and runs until its turn number `turns` ended.
fn turn(app: &mut App, wakes: &std::sync::mpsc::Receiver<()>, agent: Entity, turns: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, "Go on.", DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

fn edit(id: &str, path: &str, old: &str, new: &str) -> Vec<rig_core::test_utils::MockStreamEvent> {
    let edits = json!([{ "old_text": old, "new_text": new }]);
    calls(id, "edit", json!({ "path": path, "edits": edits }))
}

#[test]
fn edits_need_a_read_of_the_file_as_it_is() {
    let dir: PathBuf = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("ho-read-before-edit-files");
    std::fs::remove_dir_all(&dir).ok();
    assert!(std::fs::create_dir_all(&dir).is_ok());
    // The refusals name the file as the tools resolve it.
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    let file = dir.join("a.txt");
    let new = dir.join("new.txt");
    assert!(std::fs::write(&file, "one").is_ok());
    let a = file.to_string_lossy().into_owned();
    let n = new.to_string_lossy().into_owned();

    let model = MockCompletionModel::from_stream_turns([
        edit("1", &a, "one", "two"),
        calls("2", "read", json!({ "path": a })),
        edit("3", &a, "one", "two"),
        reply("first done"),
        edit("4", &a, "three", "four"),
        calls("5", "write", json!({ "path": n, "content": "new" })),
        calls("6", "read", json!({ "path": a })),
        reply("second done"),
    ]);
    let store = MemoryStore::default();
    let (mut first, wakes, agent) = app(&store, &model);
    turn(&mut first, &wakes, agent, 1);
    let after_three = std::fs::read_to_string(&file).unwrap_or_default();
    // Someone else changes the file between the turns.
    assert!(std::fs::write(&file, "three").is_ok());
    turn(&mut first, &wakes, agent, 2);

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(
        after_three, "two",
        "(3) ran after the read; (1) changed nothing"
    );
    assert_eq!(sent.len(), 8, "{sent:?}");
    let [_, after_1, _, after_3, _, after_4, after_5, _] = sent.as_slice() else {
        return;
    };
    let unread = format!("Read {a} first: it has not been read in this conversation.");
    assert!(after_1.contains(&unread), "{after_1}");
    assert!(after_3.contains("Edited "), "{after_3}");
    let changed = format!("{a} changed since you read it; read it again.");
    assert!(after_4.contains(&changed), "{after_4}");
    assert!(after_5.contains("Created "), "{after_5}");
    assert_eq!(std::fs::read_to_string(&new).unwrap_or_default(), "new");
    assert_eq!(std::fs::read_to_string(&file).unwrap_or_default(), "three");
    drop(first);

    // A restart on the same session remembers the last read of `a.txt`.
    let again = MockCompletionModel::from_stream_turns([
        edit("7", &a, "three", "four"),
        reply("third done"),
    ]);
    let (mut second, wakes, agent) = app(&store, &again);
    turn(&mut second, &wakes, agent, 1);
    let sent = serde_json::to_string(&again.requests()).unwrap_or_default();
    let last = std::fs::read_to_string(&file).unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    // What the model read of call 7, after the restored conversation.
    let result = sent.rsplit("\"toolresult\"").next().unwrap_or_default();
    assert!(result.contains("Edited "), "{result}");
    assert_eq!(last, "four");

    // A conversation compacted into a summary forgets what its agent read.
    let remembered = |world: &World| world.get::<ReadFiles>(agent).map(|read| read.0.len());
    assert_eq!(remembered(second.world()), Some(2), "a.txt and new.txt");
    let compact = move |mut agents: Query<&mut Conversation>| {
        if let Ok(mut conversation) = agents.get_mut(agent) {
            let summary = message::Message::user("The files were read.");
            let summary = Entry::new(summary).by(summary_author());
            conversation.splice(..2, [summary], summary_author());
        }
    };
    assert!(second.world_mut().run_system_once(compact).is_ok());
    second.update();
    assert_eq!(remembered(second.world()), Some(0));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ReadBeforeEditPlugin);
}
