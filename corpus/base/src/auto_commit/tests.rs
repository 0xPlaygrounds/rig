use std::fs;
use std::path::{Path, PathBuf};

use rig_basics::coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::settings::Settings;
use rig_harness_test_support::MemoryStore;

use super::*;
use rig_harness_test_support::{
    Ended, app_on, calls, first_agent, log_ended, reply, run_until, scripted_models,
};

/// A repository with one commit, and the user's uncommitted `b.txt`.
fn repository(dir: &Path) -> Option<()> {
    fs::remove_dir_all(dir).ok();
    fs::create_dir_all(dir).ok()?;
    fs::write(dir.join("README"), "hello\n").ok()?;
    for args in [
        &["init", "-q"][..],
        &["config", "user.email", "test@example.com"],
        &["config", "user.name", "Test"],
        &["config", "commit.gpgsign", "false"],
        &["add", "README"],
        &["commit", "-q", "-m", "Start"],
    ] {
        git(&Processes::default(), dir, args).ok()?;
    }
    fs::write(dir.join("b.txt"), "the user's own work\n").ok()
}

#[test]
fn a_turn_that_writes_a_file_is_committed_with_the_model_subject_and_undone() {
    let base = std::env::var_os("CARGO_TARGET_DIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join(format!("corpus-auto-commit-{}", std::process::id()));
    assert!(
        repository(&dir).is_some(),
        "a repository in {}",
        dir.display()
    );
    let a = dir.join("a.txt").display().to_string();
    let processes = Processes::default();
    let git = |args: &[&str]| git(&processes, &dir, args).unwrap_or_default();

    let model = MockCompletionModel::from_stream_turns([
        calls(
            "1",
            "write",
            serde_json::json!({ "path": a, "content": "a\n" }),
        ),
        reply("added a"),
    ]);
    let cheap = MockCompletionModel::from_stream_turns([reply("Add a.txt")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(Settings::from_toml(
            "[auto-commit]\nmessage_model = \"corpus-scripted/cheap\"",
        ))
        .insert_resource(scripted_models([
            ("corpus-scripted/agent", &model),
            ("corpus-scripted/cheap", &cheap),
        ]))
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, WriteTool, AutoCommitPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let model = "corpus-scripted/agent".to_owned();
    world.trigger(SetModel {
        entity: agent,
        model,
    });
    world.trigger(Deliver::new(agent, "Add a file a.txt", DeliveryMode::Queue));

    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(git(&["log", "-1", "--format=%s"]), "Add a.txt");
    let body = git(&["log", "-1", "--format=%b"]);
    assert!(body.contains("Turn: 1"), "{body}");
    assert_eq!(git(&["ls-tree", "--name-only", "HEAD"]), "README\na.txt");
    let status = git(&["status", "--porcelain"]);
    assert_eq!(status, "?? b.txt", "the user's file stays uncommitted");

    app.world_mut()
        .trigger(Deliver::new(agent, "/undo", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |_| {
        git(&["ls-tree", "--name-only", "HEAD"]) == "README"
    });
    assert_eq!(git(&["ls-tree", "--name-only", "HEAD"]), "README");
    assert!(!dir.join("a.txt").exists());
    assert!(dir.join("b.txt").exists());
    let told = app
        .world()
        .get::<Conversation>(agent)
        .is_some_and(|conversation| {
            format!("{:?}", conversation.messages().last()).contains("The user undid your changes")
        });
    assert!(told, "the agent is told its changes were undone");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(AutoCommitPlugin);
}
