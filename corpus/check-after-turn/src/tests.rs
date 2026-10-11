use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{ShellTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

use super::*;

/// The turns that ended and the notices written, in order.
#[derive(Resource, Default)]
struct Seen {
    turns: usize,
    notices: Vec<String>,
}

fn collect(mut notices: MessageReader<Notice>, mut seen: ResMut<Seen>) {
    seen.notices
        .extend(notices.read().map(|notice| notice.text.clone()));
}

/// A fresh directory for `name`, under the build's target directory.
fn scratch(name: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?.join(format!(
        "corpus-check-after-turn-{name}-{}",
        std::process::id()
    ));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The kernel with `write`, `shell` and the checks of `toml` for `root`,
/// its agent answered by `model`, after the first frame.
fn start(
    store: &MemoryStore,
    root: &Path,
    toml: &str,
    model: &MockCompletionModel,
) -> Option<(App, Receiver<()>, Entity)> {
    let checks = Checks::parse(root.to_path_buf(), toml).ok()?;
    let (mut app, wakes) = app_on(store);
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(checks)
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            WriteTool,
            ShellTool,
            CheckAfterTurnPlugin,
        ))
        .init_resource::<Seen>()
        .add_observer(|_: On<TurnEnded>, mut seen: ResMut<Seen>| seen.turns += 1)
        .add_systems(Last, collect);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(connect(model)?);
    Some((app, wakes, agent))
}

/// A turn that writes `content` to `path`, then says it is done.
fn writes(id: &str, path: &Path, content: &str) -> [Vec<MockStreamEvent>; 2] {
    let args = serde_json::json!({ "path": path.display().to_string(), "content": content });
    [calls(id, "write", args), reply("Done.")]
}

/// Asks `agent` for the file, then runs until a notice has `until`, and a
/// few frames more, in case anything else would start.
fn run(app: &mut App, wakes: &Receiver<()>, agent: Entity, until: &str) {
    let ask = Deliver::new(agent, "Make out.txt.", DeliveryMode::Steer);
    app.world_mut().trigger(ask);
    run_until(app, wakes, |world| {
        let seen = world.resource::<Seen>();
        seen.notices.iter().any(|notice| notice.contains(until))
    });
    for _ in 0..20 {
        app.update();
    }
}

#[test]
fn a_failing_check_starts_a_fix_turn_until_it_passes() {
    let store = MemoryStore::default();
    let dir = scratch("fix");
    assert!(dir.is_some(), "a scratch directory");
    let Some(dir) = dir else { return };
    let out = dir.join("out.txt");
    let toml = format!(
        "[[check]]\nname = \"has-ok\"\ncommand = \"grep -q ok '{}'\"\nwhen = \"*.txt\"\n",
        out.display()
    );
    let [bad, done] = writes("1", &out, "bad");
    let [good, fixed] = writes("2", &out, "ok");
    let model = MockCompletionModel::from_stream_turns([bad, done, good, fixed]);
    let started = start(&store, &dir, &toml, &model);
    assert!(started.is_some(), "the app starts");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    run(&mut app, &wakes, agent, "checks passed");

    let seen = app.world().resource::<Seen>();
    assert_eq!(seen.turns, 2, "the user's turn and one fix turn");
    assert!(
        seen.notices.iter().any(|notice| notice == "checks passed"),
        "{:?}",
        seen.notices
    );
    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    assert_eq!(sent.len(), 4, "two model calls per turn");
    let fix = sent.get(2).map(String::as_str).unwrap_or_default();
    assert!(
        fix.contains("has-ok") && fix.contains("failed") && fix.contains("[exit code 1]"),
        "the fix turn's first request names the check and its failure: {fix}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_check_that_keeps_failing_stops_after_max_rounds() {
    let store = MemoryStore::default();
    let dir = scratch("rounds");
    assert!(dir.is_some(), "a scratch directory");
    let Some(dir) = dir else { return };
    let out = dir.join("out.txt");
    let toml =
        "max_rounds = 2\n[[check]]\nname = \"never\"\ncommand = \"false\"\nwhen = \"*.txt\"\n";
    let script = (1..=3).flat_map(|turn| writes(&turn.to_string(), &out, "x"));
    let model = MockCompletionModel::from_stream_turns(script);
    let started = start(&store, &dir, toml, &model);
    assert!(started.is_some(), "the app starts");
    let Some((mut app, wakes, agent)) = started else {
        return;
    };
    run(&mut app, &wakes, agent, "still failing");

    let seen = app.world().resource::<Seen>();
    assert_eq!(seen.turns, 3, "the user's turn and two fix rounds");
    assert_eq!(model.request_count(), 6);
    assert!(
        seen.notices
            .iter()
            .any(|notice| notice == "checks still failing after 2 rounds"),
        "{:?}",
        seen.notices
    );
    std::fs::remove_dir_all(&dir).ok();
}
