use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    AnswerAsks, Answers, Collected, Ended, app_in, calls, collect, first_agent, log_ended, reply,
    run_until, scratch_home, scripted_models,
};
use serde_json::json;

use super::*;

/// The test's `RIG_HOME`, with the git repository inside it.
fn repo(home: &Home) -> PathBuf {
    home.root().join("repo")
}

fn git(home: &Home, args: &[&str]) -> String {
    let git = Git {
        root: repo(home),
        launcher: Launcher::default(),
    };
    git.run(None, args)
        .unwrap_or_else(|failure| format!("failed: {failure}"))
}

fn read(home: &Home, path: &str) -> String {
    fs::read_to_string(repo(home).join(path)).unwrap_or_default()
}

fn refs(home: &Home) -> String {
    git(home, &["for-each-ref", "--format=%(refname)", REFS])
}

/// A repository with one commit and one stash entry.
fn init(home: &Home) {
    let root = repo(home);
    fs::create_dir_all(&root).ok();
    git(home, &["init", "-q", "-b", "main"]);
    fs::write(root.join(".gitignore"), "target/\n").ok();
    fs::write(root.join("README"), "hello\n").ok();
    git(home, &["add", "."]);
    git(home, &["commit", "-q", "-m", "first"]);
    fs::write(root.join("README"), "stashed\n").ok();
    git(home, &["stash", "-q"]);
}

fn start(home: &Home, session: &SessionId, model: &MockCompletionModel) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(home, session);
    app.insert_resource(ProjectRoot(repo(home)))
        .insert_resource(scripted_models([("test/model", model)]))
        .insert_resource(Answers::new([Answer::choice("yes")]))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            PickPlugin,
            WriteTool,
            AnswerAsks,
            CheckpointRewindPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Recalled>(&mut app);
    app.finish();
    app.update();
    (app, wakes)
}

fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// The open picker and the position of its item for the message `text`.
fn item(app: &mut App, text: &str) -> Option<(Entity, usize)> {
    let world = app.world_mut();
    let mut pickers = world.query::<(Entity, &Picker)>();
    let (entity, picker) = pickers.iter(world).next()?;
    let index = picker
        .items
        .iter()
        .position(|item| item.label.ends_with(text))?;
    Some((entity, index))
}

#[test]
fn rewind_restores_files_and_conversation_without_touching_the_branch() {
    let home = scratch_home("checkpoint-rewind");
    init(&home);
    let file = repo(&home).join("a.txt").to_string_lossy().into_owned();
    let model = MockCompletionModel::from_stream_turns([
        reply("hi"),
        calls("1", "write", json!({ "path": file, "content": "1" })),
        reply("made a"),
        calls("2", "write", json!({ "path": file, "content": "2" })),
        reply("changed a"),
    ]);
    let session = SessionId::generate();
    let (mut app, wakes) = start(&home, &session, &model);
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model_name = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model_name,
        kind: ChoiceKind::ForThisRun,
    });
    let head = git(&home, &["rev-parse", "HEAD"]);
    let branch = git(&home, &["symbolic-ref", "HEAD"]);
    let stash = git(&home, &["stash", "list"]);
    let status = git(&home, &["status", "--porcelain"]);

    ask(&mut app, &wakes, agent, "say hi");
    ask(&mut app, &wakes, agent, "make a");
    ask(&mut app, &wakes, agent, "change a");
    assert_eq!(read(&home, "a.txt"), "2");
    // Turn 1 began with the files turn 0 began with: no ref of its own.
    let ours = format!("{REFS}/{session}");
    assert_eq!(refs(&home), format!("{ours}/0\n{ours}/2"));

    // An edit of the user's since the last turn is undone only once they
    // said yes.
    fs::write(repo(&home).join("notes.md"), "mine\n").ok();
    app.world_mut()
        .trigger(RunCommand::new(agent, "rewind".to_owned()));
    app.update();
    let picked = item(&mut app, "change a");
    assert!(picked.is_some(), "a picker with turn 2's message");
    let (picker, index) = picked.unwrap_or((Entity::PLACEHOLDER, 0));
    let chosen = Some(Chosen::item(index));
    app.world_mut().trigger(Pick::person(picker, chosen));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Collected<Recalled>>().0.is_empty()
    });

    let asked = &app.world().resource::<Answers>().asked;
    assert_eq!(asked, &["Rewind to before turn 2?"]);
    assert_eq!(read(&home, "a.txt"), "1");
    assert!(!repo(&home).join("notes.md").exists());
    let conversation = app.world().get::<Conversation>(agent);
    let last = conversation.and_then(|conversation| conversation.messages().last());
    assert_eq!(
        last.and_then(|message| message.final_answer()).as_deref(),
        Some("made a")
    );
    let recalled = &app.world().resource::<Collected<Recalled>>().0;
    let texts: Vec<&str> = recalled
        .iter()
        .map(|recalled| recalled.text.as_str())
        .collect();
    assert_eq!(texts, ["change a"]);
    assert_eq!(refs(&home), format!("{ours}/0"));
    assert_eq!(git(&home, &["rev-parse", "HEAD"]), head);
    assert_eq!(git(&home, &["symbolic-ref", "HEAD"]), branch);
    assert_eq!(git(&home, &["stash", "list"]), stash);
    // The user's index is untouched: a.txt is still untracked.
    let untracked = format!("{status}\n?? a.txt");
    assert_eq!(
        git(&home, &["status", "--porcelain"]),
        untracked.trim_start()
    );

    // A restart keeps both the cut and the checkpoints left.
    drop(app);
    let (mut app, _) = start(&home, &session, &model);
    let world = app.world_mut();
    let mut agents = world.query::<(&Conversation, &Checkpoints)>();
    let restored = agents.iter(world).next();
    let last = restored.and_then(|(conversation, _)| conversation.messages().last());
    assert_eq!(
        last.and_then(|message| message.final_answer()).as_deref(),
        Some("made a")
    );
    let turns = restored.map(|(_, kept)| kept.list.iter().map(|kept| kept.turn).collect());
    assert_eq!(turns, Some(vec![0, 1]));

    // Deleting the session from another one deletes its checkpoints.
    drop(app);
    let other: SessionId = format!("{session}-2")
        .parse()
        .unwrap_or_else(|_| SessionId::generate());
    let (mut app, wakes) = start(&home, &other, &model);
    app.world_mut().trigger(DeleteSession(session));
    run_until(&mut app, &wakes, |_| refs(&home).is_empty());
    assert_eq!(refs(&home), "");
    fs::remove_dir_all(home.root()).ok();
}

/// Holds a test's tool call that never runs.
#[derive(Component)]
struct Stuck;

/// Holds every call writing `c.txt`, so the turn is still running.
fn hold_c(
    calls: Query<(Entity, &ToolRequest), (ReadyToStart, ToolCalls)>,
    stuck: Query<&HoldOf, With<Stuck>>,
    mut commands: Commands,
) {
    for (call, request) in &calls {
        let path = request.call.function.arguments.get("path");
        let writes_c = path
            .and_then(|path| path.as_str())
            .is_some_and(|path| path.ends_with("c.txt"));
        if writes_c && !stuck.iter().any(|&HoldOf(held)| held == call) {
            commands.spawn((Stuck, HoldOf(call)));
        }
    }
}

#[test]
fn a_stopped_turns_edits_are_not_the_users() {
    let home = scratch_home("checkpoint-rewind-stopped");
    init(&home);
    let path = |name: &str| repo(&home).join(name).to_string_lossy().into_owned();
    let model = MockCompletionModel::from_stream_turns([
        reply("hi"),
        calls(
            "1",
            "write",
            json!({ "path": path("b.txt"), "content": "b" }),
        ),
        calls(
            "2",
            "write",
            json!({ "path": path("c.txt"), "content": "c" }),
        ),
        reply("ok"),
    ]);
    let (mut app, wakes) = start(&home, &SessionId::generate(), &model);
    app.add_systems(AgentPass, hold_c.in_set(Stage::Gate));
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model_name = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model_name,
        kind: ChoiceKind::ForThisRun,
    });
    ask(&mut app, &wakes, agent, "say hi");

    // The agent writes b.txt, and is stopped while it would write c.txt.
    app.world_mut()
        .trigger(Deliver::new(agent, "write b and c", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Stuck>())
    });
    app.world_mut().trigger(StopTurn::new(agent));
    run_until(&mut app, &wakes, |world| {
        let noting = world
            .iter_entities()
            .any(|entity| entity.contains::<Settling>());
        world.resource::<Ended>().0.len() == 2 && !noting
    });

    // Only the file the user wrote since counts as theirs.
    fs::write(repo(&home).join("notes.md"), "mine\n").ok();
    ask(&mut app, &wakes, agent, "go on");
    let checkpoints = app.world().get::<Checkpoints>(agent);
    let last = checkpoints.and_then(|checkpoints| checkpoints.list.last());
    let edits = last.map(|last| (last.text.as_str(), last.user_edits.clone()));
    assert_eq!(edits, Some(("go on", vec!["notes.md".to_owned()])));

    // Rewinding the conversation to before it keeps the stopped turn's
    // tool results, stopped.
    let rewind = "rewind --chat-only 2".to_owned();
    app.world_mut().trigger(RunCommand::new(agent, rewind));
    app.update();
    let recalled = &app.world().resource::<Collected<Recalled>>().0;
    let texts: Vec<&str> = recalled
        .iter()
        .map(|recalled| recalled.text.as_str())
        .collect();
    assert_eq!(texts, ["go on"]);
    let conversation = app.world().get::<Conversation>(agent);
    let last = conversation.and_then(|conversation| conversation.messages().last());
    let content = match last.map(|message| &**message) {
        Some(message::Message::User { content }) => content.as_slice(),
        _ => &[],
    };
    let results = content
        .iter()
        .filter(|item| matches!(item, UserContent::ToolResult(_)));
    assert_eq!((results.count(), content.len()), (1, 1), "{last:?}");
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(CheckpointRewindPlugin);
}
