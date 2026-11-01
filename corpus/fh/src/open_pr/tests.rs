use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use rig_basics::coding_tools::ShellTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::settings::Settings;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    AnswerAsks, Answers, Collected, Ended, app_on, calls, collect, first_agent, log_ended, reply,
    run_until, scripted_models,
};
use rig_tools::workspace::Local;

use super::*;

/// A fake `gh` that logs its arguments, one per line, and reports an open
/// pull request when `existing` exists beside it.
const FAKE_GH: &str = r#"#!/bin/sh
here=$(dirname "$0")
{ printf '%s\n' "$@"; echo '--end--'; } >> "$here/gh.log"
case "$1 $2" in
  "--version "*) echo "gh fake"; exit 0 ;;
  "pr view")
    if [ -f "$here/existing" ]; then
      echo '{"url":"https://example.com/pr/1","body":"old","state":"OPEN"}'; exit 0
    fi
    echo "no pull requests found" >&2; exit 1 ;;
  "pr create") echo "https://example.com/pr/2"; exit 0 ;;
  "pr edit") exit 0 ;;
esac
exit 1
"#;

/// A repository `work` with one commit pushed to the bare `remote.git`,
/// the user's uncommitted `b.txt`, and the fake `gh`, under the test
/// binary's directory.
fn fixture(name: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe
        .parent()?
        .join(format!("fh-open-pr-{name}-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    let work = dir.join("work");
    fs::create_dir_all(&work).ok()?;
    let runner = local_runner();
    runner
        .git(&dir, &["init", "-q", "--bare", "-b", "main", "remote.git"])
        .ok()?;
    fs::write(work.join("README"), "hello\n").ok()?;
    let remote = dir.join("remote.git").display().to_string();
    for args in [
        &["init", "-q", "-b", "main"][..],
        &["config", "user.email", "test@example.com"],
        &["config", "user.name", "Test"],
        &["config", "commit.gpgsign", "false"],
        &["add", "README"],
        &["commit", "-q", "-m", "Start"],
        &["remote", "add", "origin", &remote],
        &["push", "-q", "-u", "origin", "main"],
        &["remote", "set-head", "origin", "main"],
    ] {
        runner.git(&work, args).ok()?;
    }
    fs::write(work.join("b.txt"), "the user's own work\n").ok()?;
    let gh = dir.join("gh");
    fs::write(&gh, FAKE_GH).ok()?;
    fs::set_permissions(&gh, fs::Permissions::from_mode(0o755)).ok()?;
    Some(dir)
}

/// The processes of this machine.
fn local_runner() -> Runner {
    Runner(Arc::new(Local), Launcher::default())
}

/// A session in `dir/work` where the agent writes `a.txt` with a shell
/// command and runs a test that passes, then `/pr` answered with `answers`, run until a notice
/// says `until`. Returns the writer's prompt and the notices.
fn session(dir: &Path, answers: Vec<Answer>, until: &str) -> (String, Vec<String>) {
    let work = dir.join("work");
    let agent_model = MockCompletionModel::from_stream_turns([
        calls(
            "1",
            "shell",
            serde_json::json!({ "command": format!("cd {} && echo a > a.txt", work.display()) }),
        ),
        calls(
            "2",
            "shell",
            serde_json::json!({ "command": format!("cd {} && test -f a.txt", work.display()) }),
        ),
        reply("Added a.txt."),
    ]);
    let text = serde_json::json!({
        "title": "Add a.txt",
        "commit_message": "Add a.txt\n\nThe file the user asked for.",
        "body": "## Summary\nAdds a.txt.\n\n## Changes\n- a.txt\n\n## How it was tested\n- `test -f a.txt`: passed",
    });
    let writer = MockCompletionModel::from_stream_turns([reply(&text.to_string())]);
    let settings = format!(
        "[open-pr]\nmodel = \"corpus-scripted/writer\"\ngh = \"{}\"",
        dir.join("gh").display()
    );
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml(&settings))
        .insert_resource(scripted_models([
            ("corpus-scripted/agent", &agent_model),
            ("corpus-scripted/writer", &writer),
        ]))
        .insert_resource(ProjectRoot(work))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ShellTool,
            AnswerAsks,
            OpenPrPlugin,
        ))
        .insert_resource(Answers::new(answers))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let model = "corpus-scripted/agent".to_owned();
    world.trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    world.trigger(Deliver::new(agent, "Add a file a.txt", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    app.world_mut()
        .trigger(Deliver::new(agent, "/pr", DeliveryMode::Steer));
    let said = |world: &World| {
        let notices = &world.resource::<Collected<Notice>>().0;
        notices.iter().any(|notice| notice.text.contains(until))
    };
    run_until(&mut app, &wakes, said);
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let notices = notices.iter().map(|notice| notice.text.clone()).collect();
    let prompt = writer
        .requests()
        .first()
        .map(|request| format!("{:?}", request.chat_history));
    (prompt.unwrap_or_default(), notices)
}

/// Runs git on the bare remote of `dir`.
fn remote(dir: &Path, args: &[&str]) -> String {
    let remote = dir.join("remote.git").display().to_string();
    let args = [&["--git-dir", &remote][..], args].concat();
    local_runner().git(dir, &args).unwrap_or_default()
}

/// Runs git in the working repository of `dir`.
fn local(dir: &Path, args: &[&str]) -> String {
    local_runner()
        .git(&dir.join("work"), args)
        .unwrap_or_default()
}

#[test]
fn publish_commits_only_the_sessions_file_and_opens_the_edited_pr() {
    let dir = fixture("publish");
    assert!(dir.is_some(), "a repository fixture");
    let dir = dir.unwrap_or_default();
    let answers = vec![
        Answer::choice("title"),
        Answer::text("Add a.txt for the docs"),
        Answer::choice("publish"),
    ];
    let (prompt, notices) = session(&dir, answers, "Opened");
    assert!(
        prompt.contains("test -f a.txt`: passed"),
        "{prompt} {notices:?}"
    );
    assert!(
        !prompt.contains("the user's own work"),
        "b.txt is not the session's: {prompt}"
    );
    let branch = "rig/add-a-txt";
    assert!(remote(&dir, &["branch"]).contains(branch), "{notices:?}");
    let range = format!("main..{branch}");
    assert_eq!(remote(&dir, &["rev-list", "--count", &range]), "1");
    assert_eq!(
        remote(&dir, &["diff", "--name-only", "main", branch]),
        "a.txt"
    );
    assert_eq!(local(&dir, &["status", "--porcelain"]), "?? b.txt");
    let log = fs::read_to_string(dir.join("gh.log")).unwrap_or_default();
    assert!(log.contains("pr\ncreate\n"), "{log}");
    assert!(
        log.contains("\nAdd a.txt for the docs\n"),
        "the edited title: {log}"
    );
    assert!(log.contains("`test -f a.txt`: passed"), "{log}");
    assert!(
        notices
            .iter()
            .any(|notice| notice.contains("https://example.com/pr/2"))
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn cancel_leaves_no_branch_commit_or_push() {
    let dir = fixture("cancel");
    assert!(dir.is_some(), "a repository fixture");
    let dir = dir.unwrap_or_default();
    session(&dir, vec![Answer::choice("cancel")], "Cancelled");
    assert_eq!(
        remote(&dir, &["branch", "--format=%(refname:short)"]),
        "main"
    );
    assert_eq!(
        local(&dir, &["branch", "--format=%(refname:short)"]),
        "main"
    );
    assert_eq!(local(&dir, &["rev-list", "--count", "HEAD"]), "1");
    assert_eq!(
        local(&dir, &["status", "--porcelain"]),
        "?? a.txt\n?? b.txt"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_open_pr_gets_the_push_and_its_description_not_a_second_pr() {
    let dir = fixture("existing");
    assert!(dir.is_some(), "a repository fixture");
    let dir = dir.unwrap_or_default();
    fs::write(dir.join("existing"), "").ok();
    let answers = vec![Answer::choice("publish"), Answer::choice("yes")];
    session(&dir, answers, "Updated the description");
    let log = fs::read_to_string(dir.join("gh.log")).unwrap_or_default();
    assert!(!log.contains("pr\ncreate\n"), "{log}");
    assert!(log.contains("pr\nedit\n"), "{log}");
    assert!(remote(&dir, &["branch"]).contains("rig/add-a-txt"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(OpenPrPlugin);
}
