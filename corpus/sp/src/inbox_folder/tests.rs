use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::Instant;

use rig_basics::subagents::Helper;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::ask::AskPlugin;
use rig_harness_test_support::{
    MemoryStore, app_on, calls, first_agent, pass_wall_clock, reply, scratch_home, scripted_models,
    tui_view,
};
use rig_tui::TuiTestScreen;
use serde_json::json;

use super::*;

/// What happened to the items, in order: `start <file>` and `end <file>`.
#[derive(Resource, Default)]
struct Order(Vec<String>);

fn started(added: On<Add<InboxTurn>>, turns: Query<&InboxTurn>, mut order: ResMut<Order>) {
    if let Ok(turn) = turns.get(added.entity) {
        order.0.push(format!("start {}", turn.item.file));
    }
}

fn ended(settled: On<Add<Settled>>, turns: Query<&InboxTurn>, mut order: ResMut<Order>) {
    if let Ok(turn) = turns.get(settled.entity) {
        order.0.push(format!("end {}", turn.item.file));
    }
}

/// The folder `in` of a fresh home, with `files` in it.
fn folder(name: &str, files: &[(&str, &str)]) -> (PathBuf, PathBuf) {
    let home = scratch_home(name);
    let root = home.root().to_path_buf();
    let dir = root.join("in");
    fs::create_dir_all(&dir).ok();
    for (file, text) in files {
        fs::write(dir.join(file), text).ok();
    }
    (fs::canonicalize(&root).unwrap_or(root), dir)
}

/// An app answering the folder `in` of `root` with the scripted models.
fn start(root: &Path, models: Models) -> (App, Receiver<()>) {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(ProjectRoot(root.to_path_buf()))
        .insert_resource(Settings::from_toml("[inbox]\ndir = \"in\"\n"))
        .insert_resource(models)
        .add_plugins((AgentPlugin, SessionLogPlugin, InboxFolderPlugin))
        .init_resource::<Order>()
        .add_observer(started)
        .add_observer(ended);
    (app, wakes)
}

/// Runs frames until `done`, each as if [`SETTLE`] passed, for 10 s at most.
fn run(app: &mut App, wakes: &Receiver<()>, done: impl Fn(&World) -> bool) {
    let begun = Instant::now();
    while begun.elapsed() < Duration::from_secs(10) && !done(app.world()) {
        app.update();
        pass_wall_clock(app.world_mut(), SETTLE);
        wakes.recv_timeout(Duration::from_millis(20)).ok();
    }
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_default()
}

/// Whether the answer file says `line` on a line of its own.
fn says(dir: &Path, file: &str, line: &str) -> bool {
    read(&dir.join(file)).lines().any(|it| it == line)
}

/// Holds the turn answering `a.md` until `b.md` is queued.
#[derive(Component)]
struct UntilBQueued;

fn hold_a(proposed: Query<(Entity, &InboxTurn), Added<TurnOutcome>>, mut commands: Commands) {
    for (turn, answering) in &proposed {
        if answering.item.file == "a.md" {
            commands.spawn((UntilBQueued, HoldOf(turn)));
        }
    }
}

fn release_a(inbox: Res<Inbox>, holds: Query<Entity, With<UntilBQueued>>, mut commands: Commands) {
    if inbox.queued().any(|file| file == "b.md") {
        for hold in &holds {
            commands.entity(hold).despawn();
        }
    }
}

/// Writes `b.md` and `c.md` once `a.md`'s turn runs.
fn drop_more(added: On<Add<InboxTurn>>, turns: Query<&InboxTurn>, inbox: Res<Inbox>) {
    let Some(dir) = inbox.dir.as_ref() else {
        return;
    };
    if turns
        .get(added.entity)
        .is_ok_and(|turn| turn.item.file == "a.md")
    {
        fs::write(dir.join("b.md"), "say B").ok();
        fs::write(dir.join("c.md"), "---\nagent: docs\n---\nsay C").ok();
    }
}

/// The `docs` helper answers on its own scripted model.
fn docs_model(added: On<Add<Helper>>, mut commands: Commands) {
    let model = "test/docs".to_owned();
    commands.trigger(SetModel {
        entity: added.entity,
        model,
    });
}

#[test]
fn items_queue_per_agent_and_are_answered_beside_their_requests() {
    let (root, dir) = folder("sp-inbox-folder-queue", &[("a.md", "say A")]);
    let main = MockCompletionModel::from_stream_turns([reply("A"), reply("B")]);
    let docs = MockCompletionModel::from_stream_turns([reply("C")]);
    let models = scripted_models([("test/main", &main), ("test/docs", &docs)]);
    let (mut app, wakes) = start(&root, models);
    // The agent shown is the primary one, as in the agent.
    app.add_plugins(rig_harness::pick::PickPlugin)
        .insert_resource(TuiTestScreen::new(80, 40))
        .add_observer(drop_more)
        .add_observer(docs_model)
        .add_systems(AgentPass, hold_a.in_set(Stage::Review))
        .add_systems(Update, release_a.before(AgentLoop));
    tui_view(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/main".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });

    let done = dir.join(DONE);
    let all_done = |_: &World| {
        ["a.md", "b.md", "c.md"]
            .iter()
            .all(|f| done.join(f).exists())
    };
    run(&mut app, &wakes, all_done);

    assert!(
        says(&dir, "a.answer.md", "A"),
        "{}",
        read(&dir.join("a.answer.md"))
    );
    assert!(
        says(&dir, "b.answer.md", "B"),
        "{}",
        read(&dir.join("b.answer.md"))
    );
    assert!(
        says(&dir, "c.answer.md", "C"),
        "{}",
        read(&dir.join("c.answer.md"))
    );
    for file in ["a.md", "b.md", "c.md"] {
        assert!(
            done.join(file).exists() && !dir.join(file).exists(),
            "{file}"
        );
    }
    // `b.md` came while `a.md`'s turn ran, and went only once it ended.
    let order = &app.world().resource::<Order>().0;
    let main_items: Vec<&str> = order
        .iter()
        .map(String::as_str)
        .filter(|it| !it.ends_with("c.md"))
        .collect();
    assert_eq!(
        main_items,
        ["start a.md", "end a.md", "start b.md", "end b.md"]
    );

    app.update();
    let rows = app.world().resource::<TuiTestScreen>().rows();
    let tag = rows.iter().position(|row| row.contains("inbox: a.md"));
    let text = rows.iter().position(|row| row.contains("say A"));
    assert!(tag.is_some() && tag < text, "{}", rows.join("\n"));
}

/// Wants the user to approve each `shell` call.
fn ask_for_shell(
    calls: Query<(Entity, &ToolRequest), (ReadyToStart, ToolCalls, Without<Approved>)>,
    mut commands: Commands,
) {
    for (call, request) in &calls {
        if request.call.function.name.as_str() == "shell" {
            commands
                .entity(call)
                .queue(NeedsApproval::add("test", "It runs a command."));
        }
    }
}

fn shell(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("shell"), ToolResult::success("ran".into()));
    }
}

#[test]
fn headless_items_fail_deny_approvals_and_retry_after_a_restart() {
    let files = [
        ("d.md", "say D"),
        ("r.md", "say R"),
        ("s.md", "list the files"),
    ];
    let (root, dir) = folder("sp-inbox-folder-headless", &files);
    // `r.md` was mid-turn when the last run stopped.
    let state = json!({ "r.md": { "agent": null, "retried": false } });
    fs::write(dir.join(STATE), state.to_string()).ok();
    let main = MockCompletionModel::from_stream_turns([
        vec![MockStreamEvent::error("the model is gone")],
        reply("R"),
        calls("s1", "shell", json!({ "command": "ls" })),
        reply("I could not list them."),
    ]);
    let models = scripted_models([("test/main", &main)]);
    let (mut app, wakes) = start(&root, models);
    app.add_plugins(AskPlugin)
        .add_systems(
            AgentPass,
            ask_for_shell.in_set(Stage::Gate).before(ApprovalSystems),
        )
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/main".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
    });

    let finished = |_: &World| {
        dir.join(FAILED).join("d.md").exists()
            && ["r.md", "s.md"]
                .iter()
                .all(|f| dir.join(DONE).join(f).exists())
    };
    run(&mut app, &wakes, finished);

    let error = read(&dir.join("d.error.md"));
    assert!(error.contains("failed"), "{error}");
    assert!(!dir.join("d.md").exists());
    let retried = read(&dir.join("r.answer.md"));
    assert!(
        retried.contains("(retried)") && says(&dir, "r.answer.md", "R"),
        "{retried}"
    );
    let denied = read(&dir.join("s.answer.md"));
    assert!(
        denied.contains("## Denied") && denied.contains("`shell`"),
        "{denied}"
    );
    assert_eq!(read(&dir.join(STATE)).trim(), "{}");
}
