use std::path::PathBuf;

use rig_basics::BasicCommandsPlugin;
use rig_basics::coding_tools::WriteTool;
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, connect, first_agent, log_ended, reply, run_until,
};

use super::*;

/// The summary the scripted model writes.
const SUMMARY: &str = "## Goal\nWrite a.txt.";

/// A project of the test's own, beside its binary, emptied.
fn project(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let root = exe.with_file_name(format!("exit-notes-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).ok();
    root
}

/// The notes written in `root`, as `(file name, text)`.
fn notes_in(root: &Path) -> Vec<(String, String)> {
    let Ok(entries) = std::fs::read_dir(root.join(".rig").join("notes")) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            (
                name,
                std::fs::read_to_string(entry.path()).unwrap_or_default(),
            )
        })
        .collect()
}

/// Runs a session in `root` on `model`: the turns `say` asks for, then
/// `/quit`, until the app exits. Returns the main agent's conversation.
fn session(root: &Path, model: &MockCompletionModel, say: &[&str]) -> Vec<Message> {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(root.to_path_buf()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            BasicCommandsPlugin,
            rig_harness::host::exit::ExitPlugin,
            WriteTool,
            ExitNotesPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let (agent, connection) = (first_agent(world), connect(model));
    assert!(
        agent.is_some() && connection.is_some(),
        "one agent, and the model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return Vec::new();
    };
    world.entity_mut(agent).insert(connection);
    for (turn, text) in say.iter().enumerate() {
        let said = Deliver::new(agent, *text, DeliveryMode::Queue);
        app.world_mut().trigger(said);
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() > turn
        });
    }
    let quit = Deliver::new(agent, "/quit", DeliveryMode::Queue);
    app.world_mut().trigger(quit);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Messages<AppExit>>().is_empty()
    });
    assert!(
        !app.world().resource::<Messages<AppExit>>().is_empty(),
        "the app exits"
    );
    let conversation = app.world().get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |conversation| {
        conversation
            .messages()
            .iter()
            .map(|message| (**message).clone())
            .collect()
    })
}

/// The first turn writes `a.txt` in `root`.
fn write_a(root: &Path) -> Vec<MockStreamEvent> {
    let path = root.join("a.txt").display().to_string();
    calls(
        "c1",
        "write",
        serde_json::json!({ "path": path, "content": "a" }),
    )
}

#[test]
fn quitting_writes_the_summary_and_the_changed_files() {
    let root = project("summary");
    let model = MockCompletionModel::from_stream_turns([
        write_a(&root),
        reply("Wrote it."),
        reply(SUMMARY),
    ]);
    let conversation = session(&root, &model, &["Write a.txt."]);
    let notes = notes_in(&root);
    assert_eq!(notes.len(), 1, "{notes:?}");
    let Some((name, text)) = notes.first() else {
        return;
    };
    assert!(name.ends_with(".md"), "{name}");
    assert!(text.contains(SUMMARY), "{text}");
    assert!(text.contains("- a.txt"), "{text}");
    let summarized = conversation.iter().any(|message| {
        let text = format!("{message:?}");
        text.contains("## Goal") || text.contains("The session ends here")
    });
    assert!(!summarized, "{conversation:#?}");
    assert_eq!(conversation.len(), 4, "{conversation:#?}");
    // The summary call asked for the note last.
    let requests = model.requests();
    let asked = requests
        .last()
        .and_then(|request| request.chat_history.last());
    assert!(
        asked.is_some_and(|asked| format!("{asked:?}").contains("The session ends here")),
        "{asked:?}"
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn a_failed_summary_still_writes_the_files_and_exits() {
    let root = project("failed");
    let failed = vec![MockStreamEvent::Error(MockError::provider("down"))];
    let model =
        MockCompletionModel::from_stream_turns([write_a(&root), reply("Wrote it."), failed]);
    session(&root, &model, &["Write a.txt."]);
    let notes = notes_in(&root);
    assert_eq!(notes.len(), 1, "{notes:?}");
    let Some((_, text)) = notes.first() else {
        return;
    };
    assert!(text.contains(UNAVAILABLE), "{text}");
    assert!(text.contains("- a.txt"), "{text}");
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn a_session_without_turns_exits_without_a_note() {
    let root = project("empty");
    let model = MockCompletionModel::from_stream_turns([reply(SUMMARY)]);
    session(&root, &model, &[]);
    assert_eq!(notes_in(&root), Vec::new());
    assert!(model.requests().is_empty(), "no summary call");
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ExitNotesPlugin);
}

/// Permission modes edit a turn's model requests, the notes their own.
#[test]
fn stages_are_ordered_with_permission_modes() {
    rig_defaults_test_support::assert_stages_ordered((
        ExitNotesPlugin,
        corpus_shared::permission_modes::PermissionModesPlugin,
    ));
}
