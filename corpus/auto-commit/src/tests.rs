use std::fs;
use std::path::{Path, PathBuf};

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use rig_ecs::journal::JournalPlugin;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

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
        git(dir, args).ok()?;
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

    let model = MockCompletionModel::from_stream_turns([
        calls(
            "1",
            "write",
            serde_json::json!({ "path": a, "content": "a\n" }),
        ),
        reply("added a"),
    ]);
    let cheap = MockCompletionModel::from_stream_turns([reply("Add a.txt")]);
    let connections = (connect(&model), connect(&cheap));
    assert!(
        connections.0.is_some() && connections.1.is_some(),
        "the catalog lists the model"
    );
    let (Some(connection), Some(cheap_connection)) = connections else {
        return;
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(AutoCommitConfig {
            dir: Some(dir.clone()),
            ..AutoCommitConfig::default()
        })
        .insert_resource(MessageModel(cheap_connection))
        .add_plugins((AgentPlugin, JournalPlugin, WriteTool, AutoCommitPlugin))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    assert!(agent.is_ok(), "one agent");
    let Ok(agent) = agent else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Add a file a.txt", DeliveryMode::Queue));

    let subject = |dir: &Path| git(dir, &["log", "-1", "--format=%s"]).unwrap_or_default();
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0 > 0 && subject(&dir) == "Add a.txt"
    });
    assert_eq!(subject(&dir), "Add a.txt");
    let body = git(&dir, &["log", "-1", "--format=%b"]).unwrap_or_default();
    assert!(body.contains("Turn: 1"), "{body}");
    let tree = |dir: &Path| git(dir, &["ls-tree", "--name-only", "HEAD"]).unwrap_or_default();
    assert_eq!(tree(&dir), "README\na.txt");
    let status = git(&dir, &["status", "--porcelain"]).unwrap_or_default();
    assert_eq!(status, "?? b.txt", "the user's file stays uncommitted");

    app.world_mut()
        .trigger(Deliver::new(agent, "/undo", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |_| tree(&dir) == "README");
    assert_eq!(tree(&dir), "README");
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
