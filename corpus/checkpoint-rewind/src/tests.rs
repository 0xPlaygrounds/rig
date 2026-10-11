use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// How many turns ended, and what came back to the input line.
#[derive(Resource, Default)]
struct Seen {
    ended: usize,
    recalled: Vec<String>,
    picks: usize,
}

/// A git repository inside the crate, removed when dropped.
struct Repo(PathBuf);

impl Drop for Repo {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

impl Repo {
    fn git(&self, args: &[&str]) -> String {
        git(&self.0, None, args).unwrap_or_else(|failure| format!("failed: {failure}"))
    }

    fn read(&self, path: &str) -> String {
        fs::read_to_string(self.0.join(path)).unwrap_or_default()
    }
}

/// A repository with one commit and one stash entry.
fn repo() -> Repo {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("test-repo-{}", std::process::id()));
    fs::remove_dir_all(&path).ok();
    fs::create_dir_all(&path).ok();
    let repo = Repo(path);
    repo.git(&["init", "-q", "-b", "main"]);
    fs::write(repo.0.join(".gitignore"), "target/\n").ok();
    fs::write(repo.0.join("README"), "hello\n").ok();
    repo.git(&["add", "."]);
    repo.git(&["commit", "-q", "-m", "first"]);
    fs::write(repo.0.join("README"), "stashed\n").ok();
    repo.git(&["stash", "-q"]);
    repo
}

fn start(store: &MemoryStore, repo: &Repo) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(CheckpointRepo {
        root: repo.0.clone(),
        session: "test".to_owned(),
    })
    .add_plugins((
        AgentPlugin,
        JournalPlugin,
        WriteTool,
        CheckpointRewindPlugin,
    ))
    .init_resource::<Seen>()
    .add_observer(|_: On<TurnEnded>, mut seen: ResMut<Seen>| seen.ended += 1)
    .add_systems(
        Last,
        |mut recalled: MessageReader<Recalled>,
         mut picks: MessageReader<PickRequest>,
         mut seen: ResMut<Seen>| {
            seen.recalled
                .extend(recalled.read().map(|recalled| recalled.text.clone()));
            seen.picks += picks.read().count();
        },
    );
    app.finish();
    app.update();
    (app, wakes)
}

fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Seen>().ended;
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| world.resource::<Seen>().ended > ended);
    app.update();
}

fn command(app: &mut App, agent: Entity, line: &str) {
    let line = line.to_owned();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line,
    });
    app.update();
}

#[test]
fn rewind_restores_files_and_conversation_without_touching_the_branch() {
    let repo = repo();
    let file = repo.0.join("a.txt").to_string_lossy().into_owned();
    let model = MockCompletionModel::from_stream_turns([
        reply("hi"),
        calls("1", "write", json!({ "path": file, "content": "1" })),
        reply("made a"),
        calls("2", "write", json!({ "path": file, "content": "2" })),
        reply("changed a"),
    ]);
    let (mut app, wakes) = start(&MemoryStore::default(), &repo);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let found = agent.zip(connect(&model));
    assert!(found.is_some(), "an agent and the catalog's model");
    let Some((agent, connection)) = found else {
        return;
    };
    let choice = ModelChoice(connection.spec.reference());
    world.entity_mut(agent).insert((connection, choice));
    let head = repo.git(&["rev-parse", "HEAD"]);
    let branch = repo.git(&["symbolic-ref", "HEAD"]);
    let stash = repo.git(&["stash", "list"]);
    let status = repo.git(&["status", "--porcelain"]);

    ask(&mut app, &wakes, agent, "say hi");
    ask(&mut app, &wakes, agent, "make a");
    ask(&mut app, &wakes, agent, "change a");
    assert_eq!(repo.read("a.txt"), "2");
    // Turn 2 began with the files turn 1 left: no ref of its own.
    let refs = repo.git(&["for-each-ref", "--format=%(refname)", REFS]);
    assert_eq!(
        refs,
        "refs/rig/checkpoints/test/1\nrefs/rig/checkpoints/test/3"
    );

    // An edit of the user's since the last turn asks before it is undone.
    fs::write(repo.0.join("notes.md"), "mine\n").ok();
    command(&mut app, agent, "rewind 3");
    assert_eq!(app.world().resource::<Seen>().picks, 1);
    assert_eq!(repo.read("a.txt"), "2");
    command(&mut app, agent, "rewind --confirm 3");

    assert_eq!(repo.read("a.txt"), "1");
    assert!(!repo.0.join("notes.md").exists());
    let conversation = app.world().get::<Conversation>(agent);
    let last = conversation.and_then(|conversation| conversation.messages().last());
    assert_eq!(last.and_then(final_answer).as_deref(), Some("made a"));
    assert_eq!(app.world().resource::<Seen>().recalled, ["change a"]);
    // Turn 3's ref went with its message; turn 2's, turn 1's, stays.
    let refs = repo.git(&["for-each-ref", "--format=%(refname)", REFS]);
    assert_eq!(refs, "refs/rig/checkpoints/test/1");
    assert_eq!(repo.git(&["rev-parse", "HEAD"]), head);
    assert_eq!(repo.git(&["symbolic-ref", "HEAD"]), branch);
    assert_eq!(repo.git(&["stash", "list"]), stash);
    // The user's index is untouched: a.txt is still untracked.
    assert_eq!(
        repo.git(&["status", "--porcelain"]),
        format!("{status}?? a.txt").trim_start()
    );
}
