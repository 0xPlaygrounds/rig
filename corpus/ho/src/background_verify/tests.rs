use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ShellTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::clock::ClockPlugin;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, connect, first_agent, log_ended, reply, run_until,
    scratch_home,
};

use super::*;

/// The stand-in check: logs each run, takes 0.3 s, fails while `a.rs` says
/// "bad".
const CHECK: &str = "echo run >> runs.log\nsleep 0.3\n\
                     if grep -q bad a.rs; then echo 'error: E1'; exit 1; fi\n";

/// A scratch project `name` with `a.rs` and the stand-in check.
fn project(name: &str) -> Option<PathBuf> {
    let dir = scratch_home(name).root().to_path_buf();
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::write(dir.join("a.rs"), "ok").ok()?;
    std::fs::write(dir.join("check.sh"), CHECK).ok()?;
    Some(dir)
}

/// A reply that edits `a.rs` from `old` to `new`.
fn edit(id: &str, root: &Path, old: &str, new: &str) -> Vec<MockStreamEvent> {
    let path = root.join("a.rs").display().to_string();
    let edits = [serde_json::json!({ "old_text": old, "new_text": new })];
    calls(
        id,
        "edit",
        serde_json::json!({ "path": path, "edits": edits }),
    )
}

/// A reply that writes `content` to `a.rs`.
fn write(id: &str, root: &Path, content: &str) -> Vec<MockStreamEvent> {
    let path = root.join("a.rs").display().to_string();
    calls(
        id,
        "write",
        serde_json::json!({ "path": path, "content": content }),
    )
}

/// A reply whose shell call runs until the file `go` exists, so the agent
/// is busy until the test lets it go on.
fn busy_until(id: &str, go: &Path) -> Vec<MockStreamEvent> {
    let go = go.display();
    let command = format!("while [ ! -f '{go}' ]; do sleep 0.05; done");
    calls(id, "shell", serde_json::json!({ "command": command }))
}

/// How many times the check ran.
fn runs(root: &Path) -> usize {
    let log = std::fs::read_to_string(root.join("runs.log")).unwrap_or_default();
    log.lines().count()
}

/// Model request `index`, as JSON.
fn request(model: &MockCompletionModel, index: usize) -> String {
    let requests = model.requests();
    let request = requests.get(index);
    request
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .unwrap_or_default()
}

/// An app checking `dir` with the stand-in, its loop's wakes, and its
/// agent on `model`.
fn app(dir: &Path, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml(
        "[verify]\ncommand = \"sh check.sh\"\nquiet_ms = 100\n",
    ))
    .insert_resource(ProjectRoot(dir.to_path_buf()))
    .add_plugins((AgentPlugin, SessionLogPlugin, ClockPlugin))
    .add_plugins((EditTool, WriteTool, ShellTool))
    .add_plugins(BackgroundVerifyPlugin)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Sends `text` to `agent`.
fn say(app: &mut App, agent: Entity, text: &str) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
}

/// Whether the agent's turns ended `count` times.
fn ended(count: usize) -> impl Fn(&World) -> bool {
    move |world| world.resource::<Ended>().0.len() >= count
}

/// Whether the check ran `count` times and its status is `status`.
fn checked(dir: &Path, count: usize, status: CheckStatus) -> impl Fn(&World) -> bool {
    let dir = dir.to_path_buf();
    move |world| runs(&dir) == count && world.resource::<Verify>().status == status
}

#[test]
fn edits_are_checked_once_quiet_and_only_the_newest_result_reaches_an_idle_agent() {
    let dir = project("corpus-ho-background-verify-idle");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let model = MockCompletionModel::from_stream_turns([
        // Turn 1: two edits, the second breaks the check.
        edit("1", &dir, "ok", "fine"),
        edit("2", &dir, "fine", "bad"),
        reply("Edited."),
        // Turn 2, "next".
        reply("I see E1."),
        // Turn 3: broken, then fixed before the check runs.
        write("3", &dir, "bad"),
        write("4", &dir, "good"),
        reply("Fixed."),
        // Turn 4, "last".
        reply("Good."),
    ]);
    let (mut app, wakes, agent) = app(&dir, &model);

    say(&mut app, agent, "Edit a.rs.");
    run_until(&mut app, &wakes, ended(1));
    run_until(&mut app, &wakes, checked(&dir, 1, CheckStatus::Failed(1)));
    assert_eq!(runs(&dir), 1, "two edits in a row are checked once");

    say(&mut app, agent, "next");
    run_until(&mut app, &wakes, ended(2));
    let next = request(&model, 3);
    assert!(
        next.contains("error: E1"),
        "the failure goes with the idle agent's next message: {next}"
    );

    say(&mut app, agent, "Fix it.");
    run_until(&mut app, &wakes, ended(3));
    run_until(&mut app, &wakes, checked(&dir, 2, CheckStatus::Passed));
    let fixed = request(&model, 6);
    assert_eq!(
        fixed.matches("error: E1").count(),
        1,
        "no new failure note: {fixed}"
    );

    say(&mut app, agent, "last");
    run_until(&mut app, &wakes, ended(4));
    let last = request(&model, 7);
    assert!(
        last.contains("passes again"),
        "the agent saw E1, so it hears of the pass: {last}"
    );
    assert_eq!(model.request_count(), 8, "no note starts a turn");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_busy_agent_gets_its_note_with_its_next_call_unless_a_pass_overtakes_it() {
    let dir = project("corpus-ho-background-verify-busy");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let model = MockCompletionModel::from_stream_turns([
        write("1", &dir, "bad"),
        busy_until("2", &dir.join("go1")),
        write("3", &dir, "bad again"),
        busy_until("4", &dir.join("go2")),
        reply("Done."),
    ]);
    let (mut app, wakes, agent) = app(&dir, &model);

    say(&mut app, agent, "Break a.rs, twice.");
    run_until(&mut app, &wakes, checked(&dir, 1, CheckStatus::Failed(1)));
    std::fs::write(dir.join("go1"), "").ok();
    run_until(&mut app, &wakes, checked(&dir, 2, CheckStatus::Failed(1)));
    let after_first = request(&model, 2);
    assert!(
        after_first.contains("error: E1"),
        "the failure goes with the busy agent's next call: {after_first}"
    );

    // The second failure waits for the agent's next call; a pass first.
    std::fs::write(dir.join("a.rs"), "good").ok();
    say(&mut app, agent, "/verify now");
    run_until(&mut app, &wakes, checked(&dir, 3, CheckStatus::Passed));
    std::fs::write(dir.join("go2"), "").ok();
    run_until(&mut app, &wakes, ended(1));
    let last = request(&model, 4);
    assert_eq!(
        last.matches("error: E1").count(),
        1,
        "the second failure was withdrawn: {last}"
    );
    assert!(
        last.contains("passes again"),
        "the agent saw the first failure, so it hears of the pass: {last}"
    );
    assert_eq!(model.request_count(), 5, "no note starts a turn");
    std::fs::remove_dir_all(&dir).ok();
}

/// Holds the first turn's proposed end, as a slow reviewer would, until
/// the test despawns the hold.
#[derive(Resource, Default)]
struct SlowReview(Vec<Entity>);

fn hold_ends(
    proposed: Query<(Entity, &Turn), (With<TurnOf>, Added<TurnOutcome>)>,
    mut slow: ResMut<SlowReview>,
    mut commands: Commands,
) {
    for (turn, _) in proposed.iter().filter(|(_, turn)| turn.index == 0) {
        slow.0.push(commands.spawn(HoldOf(turn)).id());
    }
}

#[test]
fn a_note_left_when_a_busy_agents_turn_ends_is_still_withdrawn_by_a_pass() {
    let dir = project("corpus-ho-background-verify-turn-end");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let model = MockCompletionModel::from_stream_turns([
        write("1", &dir, "bad"),
        reply("Done."),
        reply("Next."),
    ]);
    let (mut app, wakes, agent) = app(&dir, &model);
    app.init_resource::<SlowReview>()
        .add_systems(AgentPass, hold_ends.in_set(Stage::Review));

    // The failure reaches the agent while its turn's end is held, after
    // its last model call.
    say(&mut app, agent, "Break a.rs.");
    run_until(&mut app, &wakes, checked(&dir, 1, CheckStatus::Failed(1)));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Verify>().waiting.is_empty()
    });
    let holds = std::mem::take(&mut app.world_mut().resource_mut::<SlowReview>().0);
    for hold in holds {
        app.world_mut().despawn(hold);
    }
    run_until(&mut app, &wakes, ended(1));

    // The turn ended without sending it, so a pass still takes it back.
    std::fs::write(dir.join("a.rs"), "good").ok();
    say(&mut app, agent, "/verify now");
    run_until(&mut app, &wakes, checked(&dir, 2, CheckStatus::Passed));
    say(&mut app, agent, "next");
    run_until(&mut app, &wakes, ended(2));
    let next = request(&model, 2);
    assert!(
        !next.contains("E1") && !next.contains("passes again"),
        "a withdrawn failure, and no pass after it: {next}"
    );
    assert_eq!(model.request_count(), 3, "no note starts a turn");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(BackgroundVerifyPlugin);
}
