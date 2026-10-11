use std::path::{Path, PathBuf};

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{EditTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until, scratch_home,
};

use super::*;

/// The stand-in check: logs each run, takes 0.3 s, fails while `a.rs` says
/// "bad".
const CHECK: &str = "echo run >> runs.log\nsleep 0.3\n\
                     if grep -q bad a.rs; then echo 'error: E1'; exit 1; fi\n";

/// A scratch project with `a.rs` and the stand-in check.
fn project() -> Option<PathBuf> {
    let dir = scratch_home("corpus-ho-background-verify")
        .root()
        .to_path_buf();
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::write(dir.join("a.rs"), "ok").ok()?;
    std::fs::write(dir.join("check.sh"), CHECK).ok()?;
    Some(dir)
}

/// A reply that edits `a.rs` from `old` to `new`.
fn edit(id: &str, root: &Path, old: &str, new: &str) -> Vec<MockStreamEvent> {
    let path = root.join("a.rs").display().to_string();
    let edits = [serde_json::json!({ "old_text": old, "new_text": new })];
    calls(
        id,
        "edit",
        serde_json::json!({ "path": path, "edits": edits }),
    )
}

/// A reply that writes `content` to `a.rs`.
fn write(id: &str, root: &Path, content: &str) -> Vec<MockStreamEvent> {
    let path = root.join("a.rs").display().to_string();
    calls(
        id,
        "write",
        serde_json::json!({ "path": path, "content": content }),
    )
}

/// How many times the check ran.
fn runs(root: &Path) -> usize {
    let log = std::fs::read_to_string(root.join("runs.log")).unwrap_or_default();
    log.lines().count()
}

/// Model request `index`, as JSON.
fn request(model: &MockCompletionModel, index: usize) -> String {
    let requests = model.requests();
    let request = requests.get(index);
    request
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .unwrap_or_default()
}

#[test]
fn edits_are_checked_once_quiet_and_only_the_newest_result_reaches_the_model() {
    let dir = project();
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let model = MockCompletionModel::from_stream_turns([
        // Turn 1: two edits, the second breaks the check.
        edit("1", &dir, "ok", "fine"),
        edit("2", &dir, "fine", "bad"),
        reply("Edited."),
        // Turn 2, "next".
        reply("I see E1."),
        // Turn 3: broken, then fixed before the check runs.
        write("3", &dir, "bad"),
        write("4", &dir, "good"),
        reply("Fixed."),
        // Turn 4, "last".
        reply("Good."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml("[verify]\ncommand = \"sh check.sh\"\n"))
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, JournalPlugin, EditTool, WriteTool))
        .add_plugins(BackgroundVerifyPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let connection = connect(&model);
    assert!(connection.is_some(), "a scripted model");
    let Some(connection) = connection else { return };
    app.world_mut().entity_mut(agent).insert(connection);
    let say = |app: &mut App, text: &str| {
        let ended = app.world().resource::<Ended>().0.len();
        app.world_mut()
            .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
        run_until(app, &wakes, |world| {
            world.resource::<Ended>().0.len() > ended
        });
    };
    let finished =
        |status: CheckStatus| move |world: &World| world.resource::<Verify>().status == status;

    say(&mut app, "Edit a.rs.");
    run_until(&mut app, &wakes, finished(CheckStatus::Failed(1)));
    assert_eq!(runs(&dir), 1, "two edits in a row are checked once");

    say(&mut app, "next");
    let next = request(&model, 3);
    assert!(
        next.contains("error: E1"),
        "the failure goes with the next message: {next}"
    );

    say(&mut app, "Fix it.");
    run_until(&mut app, &wakes, finished(CheckStatus::Passed));
    assert_eq!(
        runs(&dir),
        2,
        "broken then fixed before the check is one run"
    );
    let fixed = request(&model, 6);
    assert_eq!(
        fixed.matches("error: E1").count(),
        1,
        "no new failure note: {fixed}"
    );

    say(&mut app, "last");
    let last = request(&model, 7);
    assert!(
        last.contains("passes again"),
        "the agent saw E1, so it hears of the pass: {last}"
    );
    assert_eq!(model.request_count(), 8, "no note starts a turn");
    std::fs::remove_dir_all(&dir).ok();
}
