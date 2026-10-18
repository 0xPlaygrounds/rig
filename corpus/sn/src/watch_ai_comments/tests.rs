use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ShellTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_on, calls, collect, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until, scratch_home,
};
use serde_json::json;

use super::*;
use rig_basics::checkpoint::CheckpointPlugin;

/// How many turns started.
#[derive(Resource, Default)]
struct Turns(usize);

fn count_turns(_: On<Add<TurnOf>>, mut turns: ResMut<Turns>) {
    turns.0 += 1;
}

/// An app with the real `edit` and checkpointed `shell` tools and this
/// plugin on the project `dir`, answered by `model`, and its agent.
fn start(model: &MockCompletionModel, dir: &Path) -> (App, Receiver<()>, Entity) {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(ProjectRoot(dir.to_path_buf()))
        .add_plugins((AgentPlugin, SessionLogPlugin, CheckpointPlugin))
        .add_plugins((EditTool, ShellTool, WatchAiCommentsPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Turns>()
        .add_observer(count_turns);
    collect::<FileChanges>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Runs frames until the watch reported `path` `times` times in all and
/// its saves were read.
fn settle(app: &mut App, wakes: &Receiver<()>, path: &Path, times: usize) {
    let reported = |world: &World| {
        let changes = &world.resource::<Collected<FileChanges>>().0;
        let seen = changes
            .iter()
            .filter(|change| change.paths.iter().any(|p| p == path));
        seen.count() >= times
    };
    run_until(app, wakes, reported);
    app.update();
    run_until(app, wakes, |world| {
        let watching = world.resource::<Watching>();
        watching.saved.is_empty() && watching.settling.is_none()
    });
}

/// How many times the watch reported `path` so far.
fn reports(app: &App, path: &Path) -> usize {
    let changes = &app.world().resource::<Collected<FileChanges>>().0;
    changes
        .iter()
        .filter(|change| change.paths.iter().any(|p| p == path))
        .count()
}

/// The messages this plugin sent `agent`, as JSON text.
fn asked(app: &App, agent: Entity) -> Vec<String> {
    let conversation = app.world().get::<Conversation>(agent);
    let messages = conversation.map(|conversation| conversation.messages().iter());
    messages
        .into_iter()
        .flatten()
        .filter_map(|message| serde_json::to_string(message).ok())
        .filter(|text| text.contains("The AI comments in the project"))
        .collect()
}

fn text(path: &PathBuf) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

#[test]
fn saved_ai_comments_start_turns_once_and_are_taken_out() {
    let home = scratch_home("watch-ai-comments");
    let dir = home.root().join("project");
    std::fs::create_dir_all(&dir).ok();
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    let file = |name: &str| dir.join(name);
    std::fs::write(file("ctx.py"), "x = 1  # AI\n").ok();
    let rename = json!([{ "old_text": "def foo()", "new_text": "def bar()" }]);
    let touch = json!([{ "old_text": "x = [1]", "new_text": "x = {1}" }]);
    // A `shell` call that runs until the test lets it end.
    let (ready, go) = (home.root().join("ready"), home.root().join("go"));
    let wait = format!(
        "touch '{}'; while [ ! -e '{}' ]; do sleep 0.05; done",
        ready.display(),
        go.display()
    );
    let model = MockCompletionModel::from_stream_turns([
        calls("a1", "edit", json!({ "path": "a.py", "edits": rename })),
        reply("Renamed."),
        calls("b1", "edit", json!({ "path": "b.py", "edits": touch })),
        reply("It is slow because a list is searched."),
        calls("s1", "shell", json!({ "command": wait })),
        reply("Hi."),
        reply("Done."),
    ]);
    let (mut app, wakes, agent) = start(&model, &dir);
    app.world_mut()
        .trigger(Deliver::new(agent, "/watch on", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Watching>().primed
    });
    // The watch keeps the worker loaded.
    let kept = app
        .world_mut()
        .query::<&rig_harness::worker::KeepLoaded>()
        .iter(app.world())
        .count();
    assert_eq!(kept, 1);

    // `AI!`: one turn, given the trigger and the context comment.
    let a = file("a.py");
    std::fs::write(&a, "def foo(): pass  # rename foo to bar AI!\n").ok();
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    assert_eq!(app.world().resource::<Turns>().0, 1);
    let first = asked(&app, agent);
    assert_eq!(first.len(), 1, "{first:?}");
    let message = first.first().cloned().unwrap_or_default();
    for said in ["a.py", "rename foo to bar AI!", "ctx.py", "x = 1  # AI"] {
        assert!(message.contains(said), "{said}: {message}");
    }
    assert_eq!(text(&a), "def bar(): pass\n");
    assert_eq!(text(&file("ctx.py")), "x = 1\n");
    // Neither the agent's edit nor the clean-up starts another.
    settle(&mut app, &wakes, &a, 1);
    assert_eq!(app.world().resource::<Turns>().0, 1);

    // `AI?`: the edit is refused, the answer ends the turn, the comment goes.
    let b = file("b.py");
    std::fs::write(&b, "# why is this slow AI?\nx = [1]\n").ok();
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let refused = outputs.get("b1").cloned().unwrap_or_default();
    assert!(refused.contains("question only"), "{refused}");
    assert_eq!(text(&b), "x = [1]\n");

    // Context alone starts nothing.
    let c = file("c.py");
    std::fs::write(&c, "# AI\ny = 2  # AI\n").ok();
    settle(&mut app, &wakes, &c, 1);
    assert_eq!(app.world().resource::<Turns>().0, 2);

    // Two saves while a turn's `shell` call runs: one queued turn, with the
    // newest comment, though the call's checkpoint lists the user's save.
    app.world_mut()
        .trigger(Deliver::new(agent, "hello", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |_| ready.exists());
    let d = file("d.py");
    std::fs::write(&d, "# first AI!\n").ok();
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Watching>().queue.is_empty()
    });
    let seen = reports(&app, &d);
    std::fs::write(&d, "# the second one AI!\n").ok();
    settle(&mut app, &wakes, &d, seen + 1);
    assert_eq!(
        app.world().resource::<Watching>().queue,
        std::slice::from_ref(&d)
    );
    std::fs::write(&go, "").ok();
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 4
    });
    assert_eq!(app.world().resource::<Turns>().0, 4);
    let last = asked(&app, agent).last().cloned().unwrap_or_default();
    assert!(
        last.contains("the second one AI!") && !last.contains("first AI!"),
        "{last}"
    );
    assert_eq!(text(&d), "");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(WatchAiCommentsPlugin);
}
