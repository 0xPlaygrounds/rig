use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::EditTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, connect, first_agent, log_ended, reply, run_until,
};
use serde_json::json;

use super::*;

/// The header every note starts with.
const HEADER: &str = "The repository changed since your last turn";

/// Runs git in `dir`, as the user would.
fn run_git(dir: &Path, args: &[&str]) -> Option<String> {
    git(&Processes::default(), dir, args)
}

/// A repository with `a.rs` and `b.rs` committed on `main`, and a scratch
/// directory beside it.
fn repository(base: &Path) -> Option<(PathBuf, PathBuf)> {
    let (dir, scratch) = (base.join("repo"), base.join("scratch"));
    fs::remove_dir_all(base).ok();
    fs::create_dir_all(&dir).ok()?;
    fs::create_dir_all(&scratch).ok()?;
    fs::write(dir.join("a.rs"), "fn a() {}\n").ok()?;
    fs::write(dir.join("b.rs"), "fn b() {}\n").ok()?;
    for args in [
        &["init", "-q", "-b", "main"][..],
        &["config", "user.email", "test@example.com"],
        &["config", "user.name", "Test"],
        &["config", "commit.gpgsign", "false"],
        &["add", "a.rs", "b.rs"],
        &["commit", "-q", "-m", "start"],
    ] {
        run_git(&dir, args)?;
    }
    Some((dir, scratch))
}

/// Says `text` to `agent` as the user and runs until `turns` turns ended.
fn say(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, turns: usize) {
    let said = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(said);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

/// Request `index` to `model`, as text.
fn sent(model: &MockCompletionModel, index: usize) -> String {
    let requests = model.requests();
    let request = requests.get(index).map(|request| &request.chat_history);
    format!("{request:?}")
}

#[test]
fn the_next_turn_hears_what_the_user_changed_and_only_that() {
    let base = std::env::current_exe().ok().and_then(|exe| {
        Some(
            exe.parent()?
                .join(format!("oc-git-delta-{}", std::process::id())),
        )
    });
    let made = base.as_deref().and_then(repository);
    assert!(made.is_some(), "a repository in {base:?}");
    let (Some(base), Some((dir, scratch))) = (base, made) else {
        return;
    };
    let a = dir.join("a.rs").display().to_string();
    let edits = json!([{ "old_text": "fn a() {}", "new_text": "fn a() { todo!() }" }]);
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "edit", json!({ "path": a, "edits": edits })),
        reply("Edited a.rs."),
        reply("Seen."),
        reply("Nothing new."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(dir.clone()))
        .insert_resource(Scratch(Some(scratch)))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            EditTool,
            GitDeltaContextPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }

    say(&mut app, &wakes, agent, "Make a() a todo.", 1);
    assert!(!sent(&model, 0).contains(HEADER), "{}", sent(&model, 0));
    assert!(
        fs::read_to_string(dir.join("a.rs")).is_ok_and(|a| a.contains("todo!()")),
        "the agent edited a.rs"
    );

    // The user's own work between turns.
    fs::write(dir.join("b.rs"), "fn b() {}\n// one\n// two\n").ok();
    run_git(&dir, &["checkout", "-q", "-b", "feat"]);
    run_git(&dir, &["commit", "-q", "-am", "fix typo"]);

    say(&mut app, &wakes, agent, "Now the other module.", 2);
    let second = sent(&model, 2);
    assert!(
        second.contains("Branch is now `feat` (was `main`)"),
        "{second}"
    );
    assert!(second.contains("fix typo"), "{second}");
    assert!(second.contains("b.rs +2"), "{second}");
    let note = second
        .split(HEADER)
        .nth(1)
        .and_then(|it| it.split("\", citations").next());
    assert!(
        note.is_some_and(|note| !note.contains("a.rs")),
        "the agent's own edit is not told: {second}"
    );

    say(&mut app, &wakes, agent, "Anything else?", 3);
    let third = sent(&model, 3);
    assert!(third.matches(HEADER).count() <= 1, "{third}");
    assert!(!third.contains(HEADER), "only the turn it was for: {third}");

    let notes: Vec<Note> = app
        .world()
        .get::<Conversation>(agent)
        .map(|conversation| {
            let notes = conversation
                .asides()
                .filter_map(|(_, aside)| Note::of(aside));
            notes.collect()
        })
        .unwrap_or_default();
    assert!(
        matches!(notes.as_slice(), [note] if note.title == NOTE_TITLE && note.text.contains("b.rs +2")),
        "the conversation keeps the one note: {notes:?}"
    );
    fs::remove_dir_all(base).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(GitDeltaContextPlugin);
}
