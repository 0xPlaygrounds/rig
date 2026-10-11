use std::process::Command;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{ReadTool, WriteTool};
use rig_core::catalog::{Catalog, Connector, ModelSpec};
use rig_core::operation::Completion;
use rig_core::providers::registry::ProviderId;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};
use rig_telemetry::{Spending, UsagePlugin};

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

/// The built-in catalog, with `corpus-review/gpt-x` answered by `reviewer`.
fn models(reviewer: &MockCompletionModel) -> Option<Models> {
    let vendor = ProviderId::declare("corpus-review").ok()?;
    let mut connector = Connector::new(Catalog::builtin().clone());
    let spec = ModelSpec::new(vendor, "gpt-x").with_tools(true);
    connector.catalog_mut().insert(spec);
    let reviewer = reviewer.clone();
    connector.provide(vendor, move |spec: &ModelSpec, _: &Catalog| {
        let model = ModelAdapter::<Completion>::new(spec.reference(), reviewer.clone());
        Ok(ErasedHandler::new(model))
    });
    Some(Models(connector))
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
    // Without the catalog, the reviewer is never asked and the test fails.
    if let Some(models) = models(&reviewer) {
        app.insert_resource(models);
    }
    let settings = "[review]\nmodel = \"corpus-review/gpt-x\"";
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, JournalPlugin, ReadTool, WriteTool))
        .add_plugins((UsagePlugin, ReviewAgentPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
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
    // The review held the end of the agent's turn, which carried on from
    // the findings: two turns ended, the reviewer's and that one.
    assert_eq!(app.world().resource::<Ended>().0.len(), 2);
    let followed = format!("{:?}", main.requests().get(2));
    assert!(followed.contains("Reviewer (gpt-x) found"), "{followed}");
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
        .map(|(by, spent)| (by.0, spent.total.calls))
        .collect();
    assert_eq!(spent, [(agent, 1)]);
    std::fs::remove_dir_all(&dir).ok();
}
