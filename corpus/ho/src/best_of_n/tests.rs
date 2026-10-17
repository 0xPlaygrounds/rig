use std::path::PathBuf;
use std::process::Command;

use rig_basics::coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Collected, MemoryStore, app_on, calls, collect, reply, run_until, scripted_models,
};

/// What `file.txt` in the project held as the apply started.
#[derive(Resource, Default)]
struct BeforeApply(Option<String>);

fn note_before_apply(
    started: On<Add<Started>>,
    jobs: Query<&Job>,
    root: Res<ProjectRoot>,
    mut before: ResMut<BeforeApply>,
) {
    if let Ok(Job {
        work: Work::Apply { .. },
        ..
    }) = jobs.get(started.entity)
    {
        before.0 = std::fs::read_to_string(root.0.join("file.txt")).ok();
    }
}

fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git").arg("-C").arg(dir).args(args).output();
    output.map_or_else(
        |error| error.to_string(),
        |output| String::from_utf8_lossy(&output.stdout).into_owned(),
    )
}

#[test]
fn the_passing_candidate_is_applied_and_the_worktrees_go() {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")), PathBuf::from);
    let repo = target.join(format!("corpus-best-of-{}", std::process::id()));
    std::fs::remove_dir_all(&repo).ok();
    std::fs::create_dir_all(&repo).ok();
    std::fs::write(repo.join("file.txt"), "old\n").ok();
    for args in [
        &["init", "-q"][..],
        &["config", "user.email", "t@t"],
        &["config", "user.name", "t"],
        &["add", "file.txt"],
        &["commit", "-qm", "start"],
    ] {
        git(&repo, args);
    }

    let main = MockCompletionModel::from_stream_turns([reply("unused")]);
    let write =
        |path: &str, text: &str| calls("w1", "write", json!({ "path": path, "content": text }));
    // Candidate 1 first tries the project's own file, from its worktree
    // `.rig/best-of/<id>/1`.
    let one = MockCompletionModel::from_stream_turns([
        write("../../../../file.txt", "escaped\n"),
        write("file.txt", "wrong\n"),
        reply("wrote wrong"),
    ]);
    let two = MockCompletionModel::from_stream_turns([write("file.txt", "right\n"), reply("ok")]);
    let models = scripted_models([
        ("best-main/m", &main),
        ("best-one/m", &one),
        ("best-two/m", &two),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(models)
        .insert_resource(ProjectRoot(repo.clone()))
        .init_resource::<BeforeApply>()
        .add_plugins((AgentPlugin, SessionLogPlugin, WriteTool, BestOfPlugin))
        .add_observer(note_before_apply);
    collect::<Notice>(&mut app);
    let host = app
        .world_mut()
        .spawn((Agent, ModelChoice("best-main/m".to_owned())))
        .id();
    app.finish();
    app.update();

    let line = "best-of 2 --auto --verify \"grep -q right file.txt\" \
                --models best-one/m,best-two/m Fix file.txt";
    app.world_mut().trigger(RunCommand::new(host, line));
    let said = |world: &World| {
        let conversation = world.get::<Conversation>(host);
        format!(
            "{:?}",
            conversation.map(|conversation| conversation.messages())
        )
    };
    let runs = repo.join(DIR);
    let left = |runs: &Path| {
        let entries = std::fs::read_dir(runs).into_iter().flatten().flatten();
        entries.filter(|entry| entry.path().is_dir()).count()
    };
    run_until(&mut app, &wakes, |world| {
        said(world).contains("candidate 2 won") && left(&runs) == 0
    });

    let before = app.world().resource::<BeforeApply>().0.clone();
    let after = std::fs::read_to_string(repo.join("file.txt")).unwrap_or_default();
    let worktrees = git(&repo, &["worktree", "list"]);
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let said = format!("{}\n{notices:?}", said(app.world()));
    std::fs::remove_dir_all(&repo).ok();

    assert_eq!(before.as_deref(), Some("old\n"), "{said}");
    assert_eq!(after, "right\n", "{said}");
    assert_eq!(left(&runs), 0, "{said}");
    assert_eq!(worktrees.lines().count(), 1, "{worktrees}");
    assert!(said.contains("candidate 2 won"), "{said}");
    assert!(said.contains("candidate 1 failed verification"), "{said}");
    assert_eq!(main.request_count(), 0, "a note starts no turn");
    let heard = format!("{:?}", one.requests());
    assert!(heard.contains("file.txt is outside"), "{heard}");
}
