use std::path::PathBuf;
use std::process::Command;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{ReadTool, WriteTool};
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use rig_telemetry::{Spending, UsagePlugin};
use serde_json::json;

use super::*;

/// A new git repository beside the test binary, in the build directory.
fn repository() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!("review-agent-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    let init = Command::new("git")
        .arg("-C")
        .arg(&dir)
        .args(["init", "-q"])
        .status();
    assert!(init.is_ok_and(|status| status.success()), "git init failed");
    dir
}

#[test]
fn a_turn_that_changed_a_file_is_reviewed_once_and_the_agent_fixes_the_findings() {
    let dir = repository();
    let file = dir.join("a.txt").to_string_lossy().into_owned();
    let main = MockCompletionModel::from_stream_turns([
        calls("w1", "write", json!({ "path": file, "content": "x" })),
        reply("done"),
        calls("w2", "write", json!({ "path": file, "content": "x\n" })),
        reply("fixed"),
    ]);
    let reviewer = MockCompletionModel::from_stream_turns([reply("a.txt lacks a newline")]);

    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(ReviewConfig {
            dir: dir.clone(),
            ..ReviewConfig::default()
        })
        .insert_resource(ReviewerModel(connect(&reviewer)))
        .add_plugins((AgentPlugin, JournalPlugin, ReadTool, WriteTool))
        .add_plugins((UsagePlugin, ReviewAgentPlugin));
    let mut agent = app.world_mut().spawn(Agent);
    if let Some(connection) = connect(&main) {
        agent.insert((ModelChoice(connection.spec.reference()), connection));
    }
    let agent = agent.id();
    app.finish();
    app.update();
    app.world_mut()
        .trigger(Deliver::new(agent, "write a.txt", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        main.request_count() >= 4 && world.get::<ActiveTurn>(agent).is_none()
    });

    // The reviewer read the user's request and the diff of a.txt.
    let asked = format!("{:?}", reviewer.requests());
    assert!(asked.contains("write a.txt"), "{asked}");
    assert!(asked.contains("b/a.txt"), "{asked}");
    assert!(asked.contains("No newline at end of file"), "{asked}");
    // The agent's follow-up turn started from the findings.
    let followed = format!("{:?}", main.requests().get(2));
    assert!(followed.contains("Reviewer ("), "{followed}");
    assert!(followed.contains("lacks a newline"), "{followed}");
    let written = std::fs::read_to_string(&file).unwrap_or_default();
    assert_eq!(written, "x\n");
    // The fix was not reviewed again; the one review counts under its own
    // agent.
    assert_eq!(reviewer.request_count(), 1);
    let world = app.world_mut();
    let mut reviewers = world.query_filtered::<(&SpawnedBy, &Spending), With<Reviewer>>();
    let spent: Vec<(Entity, u64)> = reviewers
        .iter(world)
        .map(|(by, spent)| (by.0, spent.0.calls))
        .collect();
    assert_eq!(spent, [(agent, 1)]);
    assert!(world.get::<Reviewing>(agent).is_none());
    std::fs::remove_dir_all(&dir).ok();
}
