use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::WriteTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_on, calls, collect, connect, first_agent, log_ended, reply, run_until,
    scratch_home,
};

use super::*;

/// The notices written, in order.
fn notices(app: &App) -> Vec<&str> {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices.iter().map(|notice| notice.text.as_str()).collect()
}

/// A fresh project named `name`, a scratch directory beside the test's
/// binary, with `checks` as its `.rig/checks.toml`.
fn project(name: &str, checks: &str) -> Option<PathBuf> {
    let dir = scratch_home(&format!("corpus-check-after-turn-{name}"))
        .root()
        .to_path_buf();
    std::fs::create_dir_all(dir.join(".rig")).ok()?;
    std::fs::write(dir.join(".rig/checks.toml"), checks).ok()?;
    Some(dir)
}

/// A turn that writes `content` to `root/out.txt`, then says it is done.
fn writes(id: &str, root: &Path, content: &str) -> [Vec<MockStreamEvent>; 2] {
    let path = root.join("out.txt").display().to_string();
    let args = serde_json::json!({ "path": path, "content": content });
    [calls(id, "write", args), reply("Done.")]
}

/// Runs a user message to an agent in the project `root`, answered by
/// `model`, until a notice has `until`, and a few frames more, in case
/// anything else would start; returns the app.
fn run(store: &MemoryStore, root: &Path, model: &MockCompletionModel, until: &str) -> Option<App> {
    let (mut app, wakes): (App, Receiver<()>) = app_on(store);
    app.insert_resource(ProjectRoot(root.to_path_buf()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            WriteTool,
            CheckAfterTurnPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world)?;
    world.entity_mut(agent).insert(connect(model)?);
    world.trigger(Deliver::new(agent, "Make out.txt.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        let notices = &world.resource::<Collected<Notice>>().0;
        notices.iter().any(|notice| notice.text.contains(until))
    });
    for _ in 0..20 {
        app.update();
    }
    Some(app)
}

#[test]
fn a_failing_check_starts_a_fix_turn_until_it_passes() {
    let toml = "[[check]]\nname = \"has-ok\"\ncommand = \"grep -q ok out.txt\"\nwhen = \"*.txt\"\n";
    let dir = project("fix", toml);
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let [bad, done] = writes("1", &dir, "bad");
    let [good, fixed] = writes("2", &dir, "ok");
    let model = MockCompletionModel::from_stream_turns([bad, done, good, fixed]);
    let app = run(&MemoryStore::default(), &dir, &model, "checks passed");
    assert!(app.is_some(), "the app starts");
    let Some(app) = app else { return };

    assert_eq!(
        app.world().resource::<Ended>().0.len(),
        2,
        "the user's turn and one fix turn"
    );
    let notices = notices(&app);
    assert!(notices.contains(&"checks passed"), "{notices:?}");
    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(sent.len(), 4, "two model calls per turn");
    let fix = sent.get(2).map(String::as_str).unwrap_or_default();
    assert!(
        fix.contains("has-ok") && fix.contains("failed") && fix.contains("exit code 1"),
        "the fix turn's first request names the check and its failure: {fix}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_check_that_keeps_failing_stops_after_max_rounds() {
    let toml =
        "max_rounds = 2\n[[check]]\nname = \"never\"\ncommand = \"false\"\nwhen = \"*.txt\"\n";
    let dir = project("rounds", toml);
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let script = (1..=3).flat_map(|turn| writes(&turn.to_string(), &dir, "x"));
    let model = MockCompletionModel::from_stream_turns(script);
    let app = run(&MemoryStore::default(), &dir, &model, "still failing");
    assert!(app.is_some(), "the app starts");
    let Some(app) = app else { return };

    assert_eq!(
        app.world().resource::<Ended>().0.len(),
        3,
        "the user's turn and two fix rounds"
    );
    assert_eq!(model.request_count(), 6);
    let notices = notices(&app);
    assert!(
        notices.contains(&"checks still failing after 2 rounds"),
        "{notices:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
