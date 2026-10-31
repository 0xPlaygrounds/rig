use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{
    Ended, MemoryStore, ToolOutputs, app_on, first_agent, log_ended, log_tool_outputs, reply,
    run_until, scripted_models,
};

use super::*;

/// Runs `git args` in `dir` as `author`, without touching optional locks.
fn git(dir: &Path, author: &str, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(dir)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .args(["-c", &format!("user.name={author}")])
        .args([
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output();
    output.map_or_else(
        |error| format!("could not run git: {error}"),
        |output| String::from_utf8_lossy(&output.stdout).into_owned(),
    )
}

/// Writes `files` in `dir` and commits them as `author` with `subject`.
fn commit(dir: &Path, author: &str, subject: &str, files: &[(&str, String)]) {
    for (name, text) in files {
        fs::write(dir.join(name), text).ok();
    }
    git(dir, author, &["add", "-A"]);
    git(dir, author, &["commit", "-q", "-m", subject]);
}

/// A repository with three commits by Alice and Bob touching `a.txt` and
/// `b.txt`, the last a 500-line rewrite of `b.txt`.
fn repository() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("ct-git-history-tools-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    fs::create_dir_all(&dir).ok();
    git(&dir, "Alice", &["init", "-q"]);
    let config =
        |limit: u32| format!("name = rig\nretry_limit = {limit}\ntimeout = 30\nverbose = false\n");
    commit(
        &dir,
        "Alice",
        "Add a and b",
        &[("a.txt", config(5)), ("b.txt", "b1\n".to_owned())],
    );
    commit(
        &dir,
        "Bob",
        "Raise retry limit to 7",
        &[("a.txt", config(7)), ("b.txt", "b1\nb2\n".to_owned())],
    );
    let big: String = (0..500).map(|n| format!("line {n}\n")).collect();
    commit(&dir, "Alice", "Rewrite b", &[("b.txt", big)]);
    dir
}

#[test]
fn history_tools_read_compactly_and_change_nothing() {
    let dir = repository();
    // Another git process holds the index lock throughout.
    let lock = dir.join(".git/index.lock");
    fs::write(&lock, "").ok();
    let index_time = || {
        fs::metadata(dir.join(".git/index"))
            .and_then(|meta| meta.modified())
            .ok()
    };
    let (status, mtime) = (git(&dir, "Alice", &["status", "--porcelain"]), index_time());

    let asks = [
        ("log", "git_log", serde_json::json!({ "path": "a.txt" })),
        (
            "blame",
            "git_blame",
            serde_json::json!({ "path": "a.txt", "start_line": 1, "end_line": 4 }),
        ),
        (
            "show",
            "git_show",
            serde_json::json!({ "rev": "HEAD~1", "path": "b.txt" }),
        ),
        (
            "diff",
            "git_diff",
            serde_json::json!({ "from": "HEAD~1", "to": "HEAD" }),
        ),
        ("nope", "git_show", serde_json::json!({ "rev": "nope" })),
        (
            "outside",
            "git_log",
            serde_json::json!({ "path": "../elsewhere" }),
        ),
    ];
    let mut turn: Vec<MockStreamEvent> = asks
        .iter()
        .map(|(id, tool, args)| MockStreamEvent::tool_call(*id, *tool, args.clone()))
        .collect();
    turn.push(MockStreamEvent::final_response_with_default_usage());
    let model = MockCompletionModel::from_stream_turns([turn, reply("It was raised to 7.")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, GitHistoryToolsPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    world.trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    world.trigger(Deliver::new(
        agent,
        "Why does retry_limit default to 7?",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();

    let log = output("log");
    assert_eq!(log.lines().count(), 2, "{log}");
    assert!(
        log.contains("Bob: Raise retry limit to 7") && log.contains("Alice: Add a and b"),
        "{log}"
    );
    assert!(!log.contains("Rewrite b"), "{log}");

    let blame = output("blame");
    assert_eq!(blame.matches("Add a and b").count(), 1, "{blame}");
    assert_eq!(
        blame.matches("Bob: Raise retry limit to 7").count(),
        1,
        "{blame}"
    );
    assert!(
        blame.contains("retry_limit = 7") && blame.contains("(see above)"),
        "{blame}"
    );

    let show = output("show");
    assert!(
        show.contains("Raise retry limit to 7") && show.contains("+b2"),
        "{show}"
    );
    assert!(!show.contains("a.txt"), "only b.txt's change: {show}");

    let diff = output("diff");
    assert!(
        diff.contains("more lines left out. Narrow with `path`"),
        "{diff}"
    );
    assert_eq!(diff.lines().count(), MAX_LINES + 1, "{diff}");

    let nope = output("nope");
    assert!(
        nope.contains("no commit or object `nope`") && !nope.contains("fatal"),
        "{nope}"
    );
    let outside = output("outside");
    assert!(outside.contains("outside the repository"), "{outside}");

    // The tools say they only read, for approval plugins.
    let mut tools = app.world_mut().query::<(&ToolDef, &Footprint)>();
    let reads = tools
        .iter(app.world())
        .filter(|(def, footprint)| {
            def.0.name.as_str().starts_with("git_") && footprint.0.reads_only()
        })
        .count();
    assert_eq!(reads, 4);

    assert!(lock.exists(), "the other process's lock stays");
    assert_eq!(git(&dir, "Alice", &["status", "--porcelain"]), status);
    assert_eq!(index_time(), mtime);
    fs::remove_dir_all(&dir).ok();
}
