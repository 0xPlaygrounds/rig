use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness_test_support::{
    AnswerAsks, Answers, Ended, app_on, calls, connect, log_ended, reply, run_until,
};
use serde_json::json;

use super::*;

/// What came back to the input line.
#[derive(Resource, Default)]
struct Seen(Vec<String>);

/// A git repository and a `RIG_HOME` inside the crate, removed when
/// dropped.
struct Dirs(PathBuf);

impl Drop for Dirs {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

impl Dirs {
    fn repo(&self) -> PathBuf {
        self.0.join("repo")
    }

    fn git(&self, args: &[&str]) -> String {
        let git = Git {
            root: self.repo(),
            launcher: Launcher::default(),
        };
        git.run(None, args)
            .unwrap_or_else(|failure| format!("failed: {failure}"))
    }

    fn read(&self, path: &str) -> String {
        fs::read_to_string(self.repo().join(path)).unwrap_or_default()
    }

    fn refs(&self) -> String {
        self.git(&["for-each-ref", "--format=%(refname)", REFS])
    }
}

/// A repository with one commit and one stash entry.
fn dirs() -> Dirs {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("test-{}", std::process::id()));
    fs::remove_dir_all(&path).ok();
    let dirs = Dirs(path);
    fs::create_dir_all(dirs.repo()).ok();
    dirs.git(&["init", "-q", "-b", "main"]);
    fs::write(dirs.repo().join(".gitignore"), "target/\n").ok();
    fs::write(dirs.repo().join("README"), "hello\n").ok();
    dirs.git(&["add", "."]);
    dirs.git(&["commit", "-q", "-m", "first"]);
    fs::write(dirs.repo().join("README"), "stashed\n").ok();
    dirs.git(&["stash", "-q"]);
    dirs
}

fn start(store: &MemoryStore, dirs: &Dirs, session: &SessionId) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    let home = Home::at(dirs.0.join("home"));
    app.insert_resource(ProjectRoot(dirs.repo()))
        .insert_resource(SessionPaths(home.session(session)))
        .insert_resource(RigHome(home))
        .insert_resource(Answers::new([Answer::Choice("yes".to_owned())]))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            WriteTool,
            AnswerAsks,
            CheckpointRewindPlugin,
        ))
        .init_resource::<Ended>()
        .init_resource::<Seen>()
        .add_observer(log_ended)
        .add_systems(
            Last,
            |mut recalled: MessageReader<Recalled>, mut seen: ResMut<Seen>| {
                seen.0
                    .extend(recalled.read().map(|recalled| recalled.text.clone()));
            },
        );
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

/// The command of the picker's item for the message `text`.
fn picked(app: &mut App, text: &str) -> Option<String> {
    let world = app.world_mut();
    let mut pickers = world.query::<&Picker>();
    let picker = pickers.iter(world).next()?;
    let item = picker
        .items
        .iter()
        .find(|item| item.label.ends_with(text))?;
    Some(item.command.clone())
}

#[test]
fn rewind_restores_files_and_conversation_without_touching_the_branch() {
    let dirs = dirs();
    let file = dirs.repo().join("a.txt").to_string_lossy().into_owned();
    let model = MockCompletionModel::from_stream_turns([
        reply("hi"),
        calls("1", "write", json!({ "path": file, "content": "1" })),
        reply("made a"),
        calls("2", "write", json!({ "path": file, "content": "2" })),
        reply("changed a"),
    ]);
    let session: SessionId = "1-1".parse().unwrap_or_else(|_| SessionId::generate());
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store, &dirs, &session);
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
    let head = dirs.git(&["rev-parse", "HEAD"]);
    let branch = dirs.git(&["symbolic-ref", "HEAD"]);
    let stash = dirs.git(&["stash", "list"]);
    let status = dirs.git(&["status", "--porcelain"]);

    ask(&mut app, &wakes, agent, "say hi");
    ask(&mut app, &wakes, agent, "make a");
    ask(&mut app, &wakes, agent, "change a");
    assert_eq!(dirs.read("a.txt"), "2");
    // Turn 1 began with the files turn 0 began with: no ref of its own.
    assert_eq!(dirs.refs(), format!("{REFS}/1-1/0\n{REFS}/1-1/2"));

    // An edit of the user's since the last turn is undone only once they
    // said yes.
    fs::write(dirs.repo().join("notes.md"), "mine\n").ok();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: "rewind".to_owned(),
    });
    app.update();
    let command = picked(&mut app, "change a");
    assert_eq!(command.as_deref(), Some("rewind 2"));
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: command.unwrap_or_default(),
    });
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Seen>().0.is_empty()
    });

    let asked = &app.world().resource::<Answers>().asked;
    assert_eq!(asked, &["Rewind to before turn 2?"]);
    assert_eq!(dirs.read("a.txt"), "1");
    assert!(!dirs.repo().join("notes.md").exists());
    let conversation = app.world().get::<Conversation>(agent);
    let last = conversation.and_then(|conversation| conversation.messages().last());
    assert_eq!(last.and_then(final_answer).as_deref(), Some("made a"));
    assert_eq!(app.world().resource::<Seen>().0, ["change a"]);
    assert_eq!(dirs.refs(), format!("{REFS}/1-1/0"));
    assert_eq!(dirs.git(&["rev-parse", "HEAD"]), head);
    assert_eq!(dirs.git(&["symbolic-ref", "HEAD"]), branch);
    assert_eq!(dirs.git(&["stash", "list"]), stash);
    // The user's index is untouched: a.txt is still untracked.
    let untracked = format!("{status}\n?? a.txt");
    assert_eq!(dirs.git(&["status", "--porcelain"]), untracked.trim_start());

    // A restart keeps both the cut and the checkpoints left.
    drop(app);
    let (mut app, wakes) = start(&store, &dirs, &session);
    let world = app.world_mut();
    let mut agents = world.query::<(&Conversation, &Checkpoints)>();
    let restored = agents.iter(world).next();
    let last = restored.and_then(|(conversation, _)| conversation.messages().last());
    assert_eq!(last.and_then(final_answer).as_deref(), Some("made a"));
    let turns = restored.map(|(_, kept)| kept.list.iter().map(|kept| kept.turn).collect());
    assert_eq!(turns, Some(vec![0, 1]));

    // Deleting the session deletes its checkpoints.
    app.world_mut().write_message(SessionDeleted(session));
    run_until(&mut app, &wakes, |_| dirs.refs().is_empty());
    assert_eq!(dirs.refs(), "");
}
