use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ReadTool};
use rig_basics::subagents::SubagentsPlugin;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Ended, ToolOutputs, answer_open, app_in, calls, first_agent, log_ended, log_tool_outputs,
    reply, run_until, scratch_home, scripted_models,
};
use serde_json::json;

use super::*;

const PARENT: &str = "dirty-guard/parent";
const SUB: &str = "dirty-guard/sub";

/// How many questions the guard asked.
#[derive(Resource, Default)]
struct Asked(usize);

fn count_asks(_: On<Add<Ask>>, mut asked: ResMut<Asked>) {
    asked.0 += 1;
}

fn git_in(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git").current_dir(dir).args(args).output();
    output.map_or_else(
        |failure| failure.to_string(),
        |out| String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

fn text(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

/// A repository named `name` with `a.txt` to `d.txt` committed and `a.txt`
/// modified since.
fn repo(name: &str) -> PathBuf {
    let dir = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name(name);
    std::fs::remove_dir_all(&dir).ok();
    assert!(std::fs::create_dir_all(&dir).is_ok());
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    git_in(&dir, &["init", "-q"]);
    git_in(&dir, &["config", "user.email", "test@example.com"]);
    git_in(&dir, &["config", "user.name", "Test"]);
    git_in(&dir, &["config", "commit.gpgsign", "false"]);
    assert!(std::fs::write(dir.join("a.txt"), "committed a\n").is_ok());
    for file in ["b.txt", "c.txt", "d.txt"] {
        assert!(std::fs::write(dir.join(file), "one\n").is_ok());
    }
    git_in(&dir, &["add", "-A"]);
    git_in(&dir, &["commit", "-qm", "init"]);
    assert!(std::fs::write(dir.join("a.txt"), "modified a\n").is_ok());
    dir
}

fn edit_call(id: &str, path: &Path) -> MockStreamEvent {
    let args = json!({ "path": path, "edits": [{ "old_text": "one", "new_text": "two" }] });
    MockStreamEvent::tool_call(id, "edit", args)
}

/// An app on the project `dir` with the file tools, subagents and the
/// guard; someone answers questions when `answerer`.
fn start(
    dir: &Path,
    parent: &MockCompletionModel,
    sub: &MockCompletionModel,
    answerer: bool,
) -> (App, Receiver<()>) {
    let name = dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned());
    let home = scratch_home(&format!("{}-home", name.unwrap_or_default()));
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(scripted_models([(PARENT, parent), (SUB, sub)]))
        .insert_resource(ProjectRoot(dir.to_path_buf()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ReadTool,
            EditTool,
            SubagentsPlugin,
        ))
        .add_plugins(DirtyTreeGuardPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Asked>()
        .add_observer(count_asks);
    if answerer {
        app.world_mut().spawn(Answerer::default());
    }
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = PARENT.to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    app.world_mut()
        .trigger(Deliver::new(agent, "Go.", DeliveryMode::Steer));
    (app, wakes)
}

fn waiting(world: &World) -> usize {
    world
        .try_query::<&GuardHold>()
        .map_or(0, |mut holds| holds.iter(world).count())
}

fn asked(world: &World) -> usize {
    world.resource::<Asked>().0
}

fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

#[test]
fn the_first_edit_waits_for_the_users_choice() {
    // A read passes; the edit asks; Stash puts the user's change away.
    let dir = repo("fh-dirty-guard-stash");
    let (a, b) = (dir.join("a.txt"), dir.join("b.txt"));
    let parent = MockCompletionModel::from_stream_turns([
        calls("1", "read", json!({ "path": a })),
        vec![
            edit_call("2", &b),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("done"),
    ]);
    let sub = MockCompletionModel::from_stream_turns([reply("unused")]);
    let (mut app, wakes) = start(&dir, &parent, &sub, true);
    run_until(&mut app, &wakes, |world| asked(world) == 1);
    assert!(
        output(&app, "1").contains("modified a"),
        "the read ran unasked"
    );
    let stash = Answer::choice("stash");
    assert!(answer_open(app.world_mut(), &title(1), &stash));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let session = app.world().resource::<SessionPaths>().id();
    let session = session.map(|id| id.to_string()).unwrap_or_default();
    let stashes = git_in(&dir, &["stash", "list"]);
    assert_eq!(stashes.lines().count(), 1, "{stashes}");
    assert!(
        stashes.contains(&format!("rig: before session {session}")),
        "{stashes}"
    );
    assert_eq!(text(&a), "committed a\n");
    assert_eq!(text(&b), "two\n");
    assert_eq!(asked(app.world()), 1);
    assert!(app.world().resource::<DirtyGuard>().settled);

    // Same tree, two edits in one reply and a subagent's edit: one question
    // for the three, all waiting on it.
    let dir = repo("fh-dirty-guard-once");
    let (c, d) = (dir.join("c.txt"), dir.join("d.txt"));
    let b = dir.join("b.txt");
    let task = json!({ "description": "edit d", "prompt": "Edit d.txt.", "model": SUB });
    let parent = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("t", "task", task),
            edit_call("e1", &b),
            edit_call("e2", &c),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("started"),
        reply("all done"),
    ]);
    let sub = MockCompletionModel::from_stream_turns([
        vec![
            edit_call("s1", &d),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("edited d"),
    ]);
    let (mut app, wakes) = start(&dir, &parent, &sub, true);
    run_until(&mut app, &wakes, |world| {
        waiting(world) == 3 && asked(world) == 1
    });
    assert_eq!(waiting(app.world()), 3);
    let go_on = Answer::choice("continue");
    assert!(answer_open(app.world_mut(), &title(1), &go_on));
    let edited = [b, c, d];
    run_until(&mut app, &wakes, |_| {
        edited.iter().all(|file| text(file) == "two\n")
    });
    assert!(edited.iter().all(|file| text(file) == "two\n"));
    assert_eq!(asked(app.world()), 1);
    assert_eq!(text(&dir.join("a.txt")), "modified a\n");
}

#[test]
fn a_run_nobody_answers_refuses_by_default() {
    let dir = repo("fh-dirty-guard-headless");
    let (a, b) = (dir.join("a.txt"), dir.join("b.txt"));
    let parent = MockCompletionModel::from_stream_turns([
        vec![
            edit_call("1", &b),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("explained"),
    ]);
    let sub = MockCompletionModel::from_stream_turns([reply("unused")]);
    let (mut app, wakes) = start(&dir, &parent, &sub, false);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert!(
        output(&app, "1").contains("nobody to ask"),
        "{}",
        output(&app, "1")
    );
    assert_eq!(text(&a), "modified a\n");
    assert_eq!(text(&b), "one\n");
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.first(), Some(TurnOutcome::Answered(_))),
        "{ended:?}"
    );
    assert_eq!(asked(app.world()), 0);
    assert!(!app.world().resource::<DirtyGuard>().settled);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(DirtyTreeGuardPlugin);
}
