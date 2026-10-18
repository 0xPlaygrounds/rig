use std::process::Command;

use rig_basics::coding_tools::{ReadTool, WriteTool};
use rig_basics::telemetry::usage::{AgentUsage, SessionUsage, UsagePlugin};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, app_on, calls, log_ended, reply, run_until, scripted_models,
};

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
    let models = [("corpus/main", &main), ("corpus-review/gpt-x", &reviewer)];
    let settings = "[review]\nmodel = \"corpus-review/gpt-x\"";
    app.insert_resource(scripted_models(models))
        .insert_resource(Settings::from_toml(settings))
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, ReadTool, WriteTool))
        .add_plugins((UsagePlugin, ReviewAgentPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    let agent = app.world_mut().spawn(Agent).id();
    app.finish();
    app.update();
    let model = "corpus/main".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });
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
    // The review held the end of the agent's turn, which carried on from
    // the findings: two turns ended, the reviewer's and that one.
    assert_eq!(app.world().resource::<Ended>().0.len(), 2);
    let followed = format!("{:?}", main.requests().get(2));
    assert!(followed.contains("Reviewer (gpt-x) found"), "{followed}");
    assert!(followed.contains("lacks a newline"), "{followed}");
    let written = std::fs::read_to_string(&file).unwrap_or_default();
    assert_eq!(written, "x\n");
    // The fix was not reviewed again. The reviewer did its one job and
    // retired; its call counts in the session's usage under its model, not
    // in the agent's.
    assert_eq!(reviewer.request_count(), 1);
    let world = app.world_mut();
    let reviewers = world
        .query_filtered::<(), With<Reviewer>>()
        .iter(world)
        .count();
    assert_eq!(reviewers, 0);
    let session = &world.resource::<SessionUsage>().0;
    let reviewed = session.models.get("corpus-review/gpt-x");
    assert_eq!(reviewed.map(|spent| spent.calls), Some(1));
    let own = world
        .get::<AgentUsage>(agent)
        .map(|spent| spent.total.calls);
    assert_eq!(own, Some(4));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ReviewAgentPlugin);
}
