use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig::harness_protocol::{Home, SessionDir, SessionId};
use rig_core::completion::Message;
use rig_core::conversation::{self, Author, Entry};
use rig_ecs::AgentPlugin;
use rig_ecs::agent::{AgentId, Conversation, Notice};
use rig_ecs::journal::{JournalPlugin, SessionStore};
use rig_harness_test_support::{commit_message, first_agent, scratch_home};
use rig_memory::{DirStore, load, save};

use super::{
    DeleteSession, FirstStart, ForkSession, ForkedFrom, Scratch, SessionDeleted, SessionForked,
    SessionPaths, SessionPlugin, SessionStores, first_typed, log_layer, typed,
};
use crate::host::cli::{AppCliExt, Front};
use crate::host::launcher::LauncherRun;
use crate::services::RigHome;

/// What the session plugin said, frame after frame.
#[derive(Resource, Default)]
struct Seen {
    forked: Vec<SessionForked>,
    deleted: Vec<SessionId>,
    notices: Vec<String>,
}

fn watch(
    mut forked: MessageReader<SessionForked>,
    mut deleted: MessageReader<SessionDeleted>,
    mut notices: MessageReader<Notice>,
    mut seen: ResMut<Seen>,
) {
    seen.forked.extend(forked.read().cloned());
    seen.deleted
        .extend(deleted.read().map(|deleted| deleted.0.clone()));
    seen.notices
        .extend(notices.read().map(|notice| notice.text.clone()));
}

/// The store of the session in `session`, kept in its directory.
fn dir_store(session: &SessionDir) -> Option<DirStore> {
    DirStore::open(session.path()).ok()
}

/// Every directory and file under `dir`, with each file's bytes.
fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    let mut seen = BTreeMap::new();
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                seen.insert(path.clone(), None);
                dirs.push(path);
            } else {
                seen.insert(path.clone(), fs::read(&path).ok());
            }
        }
    }
    seen
}

#[test]
fn a_fork_copies_the_session_and_another_session_can_be_deleted() {
    let root = std::env::temp_dir().join(format!("rig-fork-{}", std::process::id()));
    let home = Home::at(&root);
    let id: SessionId = "1700000000-1"
        .parse()
        .unwrap_or_else(|_| SessionId::generate());
    let session = home.session(&id);
    fs::create_dir_all(session.path()).ok();
    fs::write(session.directory(), root.to_string_lossy().as_bytes()).ok();
    let store = dir_store(&session);
    assert!(store.is_some(), "the session's store opens");
    let Some(store) = store else { return };
    let mut said = conversation::Conversation::from_messages([Message::user("hi")]);
    assert!(save(&store, "agent-1", &mut said).is_ok());
    let mut app = App::new();
    app.add_message::<Notice>()
        .init_resource::<Seen>()
        .insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session.clone()))
        .insert_resource(SessionStore::new(store))
        .add_plugins(SessionPlugin)
        .add_systems(Last, watch.after(super::fork_sessions));
    app.world_mut().trigger(ForkSession::default());
    app.update();
    let forked = app.world().resource::<Seen>().forked.first().cloned();
    assert_eq!(forked.as_ref().map(|forked| &forked.from), Some(&id));
    let Some(to) = forked.map(|forked| forked.to) else {
        return;
    };
    let copy = home.session(&to);
    let copied = dir_store(&copy).and_then(|copy| load(&copy, "agent-1").ok());
    let messages = copied.map(|copied| copied.messages().to_vec());
    assert_eq!(messages, Some(vec![Message::user("hi")]));
    assert_eq!(copy.working_directory(), Some(root.clone()));
    // The copy names the session it came from, and is saved, so it can be
    // resumed.
    let parent = ForkedFrom {
        session: id.clone(),
        before: None,
        text: None,
    };
    let parent_of = |session: &SessionDir| dir_store(session).as_ref().and_then(ForkedFrom::of);
    assert_eq!(parent_of(&copy), Some(parent));
    assert_eq!(parent_of(&session), None);
    assert!(copy.is_saved());

    // A session another rig runs, and this one, stay.
    let lock = File::create(copy.launcher_lock()).ok();
    assert!(lock.as_ref().is_some_and(|lock| lock.try_lock().is_ok()));
    app.world_mut().trigger(DeleteSession(to.clone()));
    app.world_mut().trigger(DeleteSession(id.clone()));
    app.update();
    assert!(copy.path().is_dir());
    drop(lock);
    app.world_mut().trigger(DeleteSession(to.clone()));
    app.update();
    let seen = app.world().resource::<Seen>();
    assert_eq!(seen.deleted, [to]);
    assert!(!copy.path().exists());
    assert_eq!(seen.notices.len(), 2, "{:?}", seen.notices);
    fs::remove_dir_all(&root).ok();
}

/// A delete reaches a store a plugin keeps outside the session's
/// directory, keyed by the session's id, and a delete a crash cut short
/// after marking the session is hidden, then finished by the next start.
#[test]
fn a_delete_reaches_a_store_kept_elsewhere_and_the_next_start_finishes_a_cut_short_one() {
    let home = scratch_home("rig-session-delete");
    let elsewhere = home.root().join("elsewhere");
    let kept = elsewhere.clone();
    let stores = SessionStores::new(move |dir| {
        let id = dir.id().map(|id| id.to_string()).unwrap_or_default();
        Ok(Arc::new(DirStore::open(kept.join(id))?))
    });
    let saved_in = |id: &str| {
        let id: SessionId = id.parse().ok()?;
        let session = home.session(&id);
        fs::create_dir_all(session.path()).ok();
        let store = stores.open(&session).ok()?;
        let mut said = conversation::Conversation::from_messages([Message::user("hi")]);
        save(&*store, "agent-1", &mut said).ok()?;
        session.is_saved().then_some((id, session))
    };
    let sessions = (
        saved_in("1700000000-20"),
        saved_in("1700000000-21"),
        saved_in("1700000000-22"),
    );
    assert!(matches!(sessions, (Some(_), Some(_), Some(_))), "saved");
    let (Some((gone, gone_dir)), Some((cut, cut_dir)), Some((here, here_dir))) = sessions else {
        return;
    };
    let start = |stores: &SessionStores| {
        let mut app = App::new();
        app.add_message::<Notice>()
            .init_resource::<Seen>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(here_dir.clone()))
            .insert_resource(stores.clone())
            .add_plugins(SessionPlugin)
            .add_systems(Last, watch.after(super::fork_sessions));
        drop(std::panic::take_hook());
        app
    };
    let logs = |id: &SessionId| elsewhere.join(id.to_string()).join("logs");
    let mut app = start(&stores);
    app.update();
    app.world_mut().trigger(DeleteSession(gone.clone()));
    app.update();
    assert!(!gone_dir.path().exists() && !logs(&gone).exists());
    assert_eq!(app.world().resource::<Seen>().deleted, [gone]);

    // A crash right after marking: nothing is removed yet, but the session
    // is no longer listed.
    assert!(cut_dir.mark_deleting().is_ok());
    assert!(!cut_dir.is_saved() && logs(&cut).exists());
    let mut next = start(&stores);
    next.update();
    assert!(!cut_dir.path().exists() && !logs(&cut).exists());
    assert_eq!(next.world().resource::<Seen>().deleted, [cut]);
    assert!(
        here_dir.is_saved() && logs(&here).exists(),
        "this one stays"
    );
    fs::remove_dir_all(home.root()).ok();
}

/// A session an older rig saved is refused before the harness writes
/// anything: resuming a copy of one leaves every byte of its directory as
/// it was, its text log included, and what the refused run logged goes to
/// the launcher's run directory.
#[test]
fn resuming_a_session_an_older_rig_saved_changes_none_of_its_bytes() {
    let home = scratch_home("rig-session-older");
    let Ok(id) = "1700000000-23".parse::<SessionId>() else {
        return;
    };
    let session = home.session(&id);
    fs::create_dir_all(session.path()).ok();
    let mut log = None;
    for line in include_str!("../../../../rig-memory/src/dir/baseline-session.txt").lines() {
        if let Some(name) = line.strip_prefix("=== ") {
            log = Some(name);
        } else if let Some(blob) = log.and_then(|name| name.strip_prefix("blob ")) {
            fs::create_dir_all(session.path().join("blobs")).ok();
            fs::write(session.path().join("blobs").join(blob), line).ok();
        } else if let Some(agent) = log {
            let path = session.path().join(format!("{agent}.jsonl"));
            let mut bytes = fs::read(&path).unwrap_or_default();
            bytes.extend_from_slice(format!("{line}\n").as_bytes());
            fs::write(path, bytes).ok();
        }
    }
    fs::write(session.log(), "what the agent did\n").ok();
    for marker in [session.started(), session.path().join("saved")] {
        fs::write(marker, b"").ok();
    }
    fs::write(
        session.directory(),
        home.root().to_string_lossy().as_bytes(),
    )
    .ok();
    let before = snapshot(session.path());
    assert!(session.is_saved() && before.len() > 4, "{before:?}");
    let run = home.run(1);
    fs::create_dir_all(run.path()).ok();

    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session.clone()))
        .insert_resource(LauncherRun(run.clone()));
    let layer = log_layer(&mut app);
    let file = app.world().resource::<super::TextLog>().file.clone();
    std::io::Write::write_all(&mut super::LogWriter(file), b"this run\n").ok();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, JournalPlugin))
        .add_plugins(SessionPlugin);
    drop(std::panic::take_hook());
    app.finish();
    app.update();
    app.update();
    let stored = app.world().contains_resource::<SessionStore>();
    drop((app, layer));
    assert!(!stored, "the refused session has no store");
    assert_eq!(snapshot(session.path()), before);
    let refusal = fs::read_to_string(run.refusal()).unwrap_or_default();
    assert!(refusal.contains("older version of rig"), "{refusal}");
    let logged = fs::read_to_string(run.log()).unwrap_or_default();
    assert!(logged.starts_with("this run\n"), "{logged}");
    fs::remove_dir_all(home.root()).ok();
}

/// A branch before a message keeps what the user wrote in it with its
/// parent, so a view of the family reads no other log.
#[test]
fn a_fork_before_a_message_keeps_its_text_with_its_parent_and_names_who_asked() {
    let root = std::env::temp_dir().join(format!("rig-fork-text-{}", std::process::id()));
    let home = Home::at(&root);
    let id: SessionId = "1700000000-2"
        .parse()
        .unwrap_or_else(|_| SessionId::generate());
    let session = home.session(&id);
    std::fs::create_dir_all(session.path()).ok();
    let mut app = App::new();
    app.init_resource::<Seen>()
        .insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session.clone()))
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin, JournalPlugin))
        .add_plugins(SessionPlugin)
        .add_systems(Last, watch.after(super::fork_sessions));
    if let Some(store) = dir_store(&session) {
        app.insert_resource(SessionStore::new(store));
    }
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "the app has an agent");
    let Some(agent) = agent else { return };
    for message in [
        Message::user("one"),
        Message::assistant("ok"),
        Message::user("  two  "),
    ] {
        let added = commit_message(app.world_mut(), agent, message);
        assert!(added.is_ok(), "{added:?}");
    }
    app.update();
    let world = app.world();
    let agent_id = world.get::<AgentId>(agent).cloned();
    let message = world
        .get::<Conversation>(agent)
        .and_then(|conversation| conversation.id(2));
    let before = agent_id.zip(message);
    assert!(before.is_some(), "the message was logged");
    let asker = app.world_mut().spawn_empty().id();
    app.world_mut().trigger(ForkSession {
        before: before.clone(),
        asker: Some(asker),
    });
    app.update();
    let forked = app.world().resource::<Seen>().forked.first().cloned();
    assert_eq!(forked.as_ref().and_then(|forked| forked.asker), Some(asker));
    let parent = forked.and_then(|forked| {
        let copy = dir_store(&home.session(&forked.to))?;
        ForkedFrom::of(&copy)
    });
    let text = parent.as_ref().and_then(|parent| parent.text.as_deref());
    assert_eq!(text, Some("two"), "{parent:?}");
    assert_eq!(parent.and_then(|parent| parent.before), before);
    drop(std::panic::take_hook());
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn a_sessionless_front_leaves_no_directory_and_a_failed_run_keeps_its_log() {
    let root = std::env::temp_dir().join(format!("rig-sessionless-{}", std::process::id()));
    let home = Home::at(&root);
    let run = home.run(1);
    std::fs::create_dir_all(run.path()).ok();
    let ended = |sessionless: bool, exit: AppExit| {
        let id = format!("{}-{}", u8::from(sessionless), exit.is_success());
        let id: SessionId = id.replace("true", "1").replace("false", "0").parse().ok()?;
        let session = home.session(&id);
        std::fs::create_dir_all(session.path()).ok();
        std::fs::write(session.log(), id.as_str()).ok();
        let front = Front::new("replay").headless();
        let front = if sessionless {
            front.sessionless()
        } else {
            front
        };
        let mut app = App::new();
        app.add_message::<Notice>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()))
            .insert_resource(LauncherRun(run.clone()))
            .add_plugins(SessionPlugin)
            .add_front(front);
        assert!(app.runs_front("replay"));
        app.world_mut().write_message(exit);
        app.update();
        Some(session.path().exists())
    };
    assert_eq!(ended(true, AppExit::Success), Some(false));
    assert!(!run.log().exists(), "a run that ended well keeps no log");
    assert_eq!(ended(true, AppExit::from_code(1)), Some(false));
    let log = std::fs::read_to_string(run.log()).ok();
    assert_eq!(log.as_deref(), Some("1-0"), "the launcher points at it");
    assert_eq!(ended(false, AppExit::from_code(1)), Some(true));
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn a_new_session_and_a_fork_start_first_and_a_restart_does_not() {
    let root = std::env::temp_dir().join(format!("rig-first-{}", std::process::id()));
    let home = Home::at(&root);
    let id: SessionId = "1700000000-3"
        .parse()
        .unwrap_or_else(|_| SessionId::generate());
    let session = home.session(&id);
    std::fs::create_dir_all(session.path()).ok();
    std::fs::write(session.directory(), root.to_string_lossy().as_bytes()).ok();
    let start = |session: &rig::harness_protocol::SessionDir| {
        let mut app = App::new();
        app.add_message::<Notice>()
            .init_resource::<Seen>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()))
            .add_plugins(SessionPlugin)
            .add_systems(Last, watch.after(super::fork_sessions));
        app.finish();
        app.update();
        app
    };
    let mut first = start(&session);
    assert!(first.world().contains_resource::<FirstStart>());
    assert!(!start(&session).world().contains_resource::<FirstStart>());
    first.world_mut().trigger(ForkSession::default());
    first.update();
    let fork = first.world().resource::<Seen>().forked.first().cloned();
    assert!(fork.is_some(), "the session was forked");
    if let Some(fork) = fork {
        let fork = start(&home.session(&fork.to));
        assert!(fork.world().contains_resource::<FirstStart>());
    }
    drop(std::panic::take_hook());
    std::fs::remove_dir_all(&root).ok();
}

/// A plugin keeps its files of a session in a directory of its own there,
/// beside the agent logs, and a restore reads only the logs.
#[test]
fn a_restore_reads_the_agent_logs_and_leaves_a_plugins_files_alone() {
    let home = scratch_home("rig-restore-plugin-files");
    let Ok(id) = "1700000000-4".parse::<SessionId>() else {
        return;
    };
    let session = home.session(&id);
    std::fs::create_dir_all(session.path()).ok();
    let start = || {
        let mut app = App::new();
        app.init_resource::<Seen>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()))
            .add_plugins((TaskPoolPlugin::default(), AgentPlugin, JournalPlugin))
            .add_plugins(SessionPlugin)
            .add_systems(Last, watch.after(super::fork_sessions));
        if let Some(store) = dir_store(&session) {
            app.insert_resource(SessionStore::new(store));
        }
        // The session's panic hook would hide a failed assertion.
        drop(std::panic::take_hook());
        app.update();
        app
    };
    let mut first = start();
    let agent = first_agent(first.world_mut());
    let said = agent.map(|agent| commit_message(first.world_mut(), agent, Message::user("hi")));
    assert_eq!(said, Some(Ok(())));
    first.update();
    drop(first);
    let plugin = session.path().join("tui");
    std::fs::create_dir_all(&plugin).ok();
    std::fs::write(plugin.join("notices.jsonl"), "{\"text\":\"kept\"}\n").ok();

    let mut restored = start();
    let agent = first_agent(restored.world_mut());
    let messages = agent
        .and_then(|agent| restored.world().get::<Conversation>(agent))
        .map(|conversation| conversation.messages().len());
    assert_eq!(messages, Some(1));
    let notices = &restored.world().resource::<Seen>().notices;
    let failed = notices.iter().any(|notice| notice.contains("restore"));
    assert!(!failed, "{notices:?}");
    std::fs::remove_dir_all(session.path()).ok();
}

/// The running session's store comes from the app's stores, which mark
/// it saved on its first log, and it stays saved after a clean exit.
#[test]
fn a_session_opened_by_the_stores_is_saved_by_its_first_log_and_stays_saved() {
    let home = scratch_home("rig-session-saved");
    let Ok(id) = "1700000000-5".parse::<SessionId>() else {
        return;
    };
    let session = home.session(&id);
    std::fs::create_dir_all(session.path()).ok();
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session.clone()))
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin, JournalPlugin))
        .add_plugins(SessionPlugin);
    drop(std::panic::take_hook());
    app.finish();
    app.update();
    assert!(app.world().contains_resource::<SessionStore>());
    assert!(!session.is_saved(), "nothing was logged yet");
    let agent = first_agent(app.world_mut());
    let said = agent.map(|agent| commit_message(app.world_mut(), agent, Message::user("hi")));
    assert_eq!(said, Some(Ok(())));
    app.update();
    app.world_mut().write_message(AppExit::Success);
    app.update();
    drop(app);
    assert!(session.is_saved());
    let store = SessionStores::default().open(&session).ok();
    let read = store.as_ref().and_then(|store| store.logs().ok());
    assert!(read.is_some_and(|logs| !logs.is_empty()));
    // The plain markers beside the store are none of its files.
    let files = store.and_then(|store| store.files().ok());
    assert!(session.started().exists());
    assert_eq!(files, Some(Vec::new()));
    std::fs::remove_dir_all(home.root()).ok();
}

/// The text log moves where a plugin puts the scratch place, and is gone
/// with none.
#[test]
fn the_text_log_follows_the_scratch_place() {
    let home = scratch_home("rig-session-scratch");
    let elsewhere = home.root().join("elsewhere");
    let log_with = |id: &str, scratch: Scratch| {
        let id: SessionId = id.parse().ok()?;
        let session = home.session(&id);
        std::fs::create_dir_all(session.path()).ok();
        let mut app = App::new();
        app.insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()));
        let layer = log_layer(&mut app);
        let logged = session.log().exists();
        app.insert_resource(scratch).add_plugins(SessionPlugin);
        drop(std::panic::take_hook());
        app.finish();
        drop((app, layer));
        Some((logged, session.log().exists()))
    };
    let moved = log_with("1700000000-6", Scratch(Some(elsewhere.clone())));
    assert_eq!(moved, Some((true, false)));
    assert!(elsewhere.join("agent.log").exists());
    let dropped = log_with("1700000000-7", Scratch(None));
    assert_eq!(dropped, Some((true, false)));
    std::fs::remove_dir_all(home.root()).ok();
}

/// A session's text log from earlier runs stays as it was when a later
/// run puts its own log elsewhere or nowhere: only what that run wrote
/// moves.
#[test]
fn a_text_log_from_earlier_runs_stays_where_it_was() {
    let home = scratch_home("rig-session-earlier-log");
    let elsewhere = home.root().join("elsewhere");
    let log_with = |id: &str, scratch: Scratch| {
        let id: SessionId = id.parse().ok()?;
        let session = home.session(&id);
        std::fs::create_dir_all(session.path()).ok();
        std::fs::write(session.log(), "earlier run\n").ok();
        let mut app = App::new();
        app.insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()));
        let layer = log_layer(&mut app);
        let file = app.world().resource::<super::TextLog>().file.clone();
        let mut write = super::LogWriter(file);
        std::io::Write::write_all(&mut write, b"this run\n").ok();
        app.insert_resource(scratch).add_plugins(SessionPlugin);
        drop(std::panic::take_hook());
        app.finish();
        drop((app, layer));
        std::fs::read_to_string(session.log()).ok()
    };
    let earlier = Some("earlier run\n".to_owned());
    assert_eq!(log_with("1700000000-8", Scratch(None)), earlier);
    assert_eq!(
        log_with("1700000000-9", Scratch(Some(elsewhere.clone()))),
        earlier
    );
    let moved = std::fs::read_to_string(elsewhere.join("agent.log")).ok();
    assert_eq!(moved.as_deref(), Some("this run\n"));
    std::fs::remove_dir_all(home.root()).ok();
}

#[test]
fn what_the_user_typed_leaves_out_attached_files_and_delivered_notes() {
    let attached = "<file path=\"a.txt\">\nfile text\n</file>";
    let mut conversation = conversation::Conversation::new();
    conversation.push(Entry::new(Message::user("build failed")).by(Author::plugin("build")));
    conversation.push(Message::assistant("ok"));
    conversation.push(Message::user(attached));
    conversation.push(Message::user("  fix it  "));
    conversation.push(Message::user("and test"));
    assert_eq!(typed(&conversation, 0), None);
    assert_eq!(typed(&conversation, 1), None);
    assert_eq!(typed(&conversation, 2), None);
    assert_eq!(typed(&conversation, 4).as_deref(), Some("and test"));
    assert_eq!(first_typed(&conversation).as_deref(), Some("fix it"));
}
