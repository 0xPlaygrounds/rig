use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::completion::Message;
use rig_core::conversation::MessageId;
use rig_core::conversation::{self, Author};
use rig_ecs::agent::{AgentId, Conversation, Notice};
use rig_ecs::session_log::{SessionLogPlugin, SessionStore};
use rig_ecs::{AgentPlugin, WriteLogs};
use rig_harness_protocol::{Home, Invocation, SessionDir, SessionId, StartReason};
use rig_harness_test_support::{commit_message, first_agent, scratch_home};
use rig_memory::{
    DirStore, Files, MemoryStore, SPLIT, Session, SessionMeta, SplitPointer, Store, Strings, load,
    save,
};

use super::{
    AppStart, DeleteSession, ForkCut, ForkSession, ForkSessions, ForkedFrom, Scratch,
    SessionDeleteFailed, SessionDeleted, SessionForkFailed, SessionForked, SessionPaths,
    SessionPlugin, StoreOpener, log_layer, rewrite_stopped,
};
use crate::host::cli::{AppCliExt, Front};
use crate::host::exit::ExitReason;
use crate::host::launcher::{Invoked, LauncherRun};
use crate::load::AppKeepsExt;
use crate::plugin_data::PluginData;
use crate::services::RigHome;

/// What the session plugin said, frame after frame.
#[derive(Resource, Default)]
struct Seen {
    forked: Vec<SessionForked>,
    fork_failed: Vec<SessionForkFailed>,
    deleted: Vec<SessionId>,
    failed: Vec<(SessionId, String)>,
    notices: Vec<String>,
}

fn watch(
    (mut forked, mut fork_failed): (
        MessageReader<SessionForked>,
        MessageReader<SessionForkFailed>,
    ),
    (mut deleted, mut failed): (
        MessageReader<SessionDeleted>,
        MessageReader<SessionDeleteFailed>,
    ),
    mut notices: MessageReader<Notice>,
    mut seen: ResMut<Seen>,
) {
    seen.forked.extend(forked.read().cloned());
    seen.fork_failed.extend(fork_failed.read().cloned());
    let failed = failed
        .read()
        .map(|failed| (failed.id.clone(), failed.why.clone()));
    seen.failed.extend(failed);
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
    // The session plugin needs no agent plugin for its notices.
    let mut app = App::new();
    app.init_resource::<Seen>()
        .insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session.clone()))
        .insert_resource(SessionStore::new(store))
        .add_plugins(SessionPlugin)
        .add_systems(Last, watch.after(ForkSessions));
    app.world_mut().write_message(ForkSession::default());
    app.update();
    let forked = app.world().resource::<Seen>().forked.first().cloned();
    assert_eq!(forked.as_ref().map(|forked| &forked.from), Some(&id));
    let Some(to) = forked.map(|forked| forked.to) else {
        return;
    };
    let copy = home.session(&to);
    let copied = dir_store(&copy).and_then(|copy| load(&copy, "agent-1").ok());
    let messages = copied.map(|copied| copied.messages().to_vec());
    assert_eq!(messages, Some(vec![Message::user("hi").into()]));
    assert_eq!(copy.working_directory(), Some(root.clone()));
    // The copy names the session it came from, and is saved, so it can be
    // resumed.
    let parent = ForkedFrom {
        session: id.to_string(),
        ..ForkedFrom::default()
    };
    let parent_of = |session: &SessionDir| dir_store(session).as_ref().and_then(ForkedFrom::read);
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
    assert_eq!(seen.deleted, std::slice::from_ref(&to));
    assert!(!copy.path().exists());
    assert_eq!(seen.notices.len(), 2, "{:?}", seen.notices);
    let failed = [
        (to, "another rig runs it".to_owned()),
        (id, "it is this session".to_owned()),
    ];
    assert_eq!(seen.failed, failed);
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
    let stores = StoreOpener::new(move |dir| {
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
        session.mark_saved().ok()?;
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
    let start = |stores: &StoreOpener| {
        let mut app = App::new();
        app.add_message::<Notice>()
            .init_resource::<Seen>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(here_dir.clone()))
            .insert_resource(stores.clone())
            .add_plugins(SessionPlugin)
            .add_systems(Last, watch.after(ForkSessions));
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

/// The layout and logs rig wrote before `rig-session/1`: `<agent>.jsonl`
/// at the top of the session directory, and its blobs.
fn write_old_layout(session: &SessionDir) {
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
}

/// A session saved in format `rig-session/3`, by the rig before this
/// format, in this rig's layout.
fn write_format_3(session: &SessionDir) {
    let mut file = None;
    for line in include_str!("../../../../rig-memory/src/dir/session-3.txt").lines() {
        if let Some(path) = line.strip_prefix("=== ") {
            let path = session.path().join(path);
            path.parent().map(fs::create_dir_all);
            fs::write(&path, b"").ok();
            file = Some(path);
        } else if let Some(path) = &file {
            let mut bytes = fs::read(path).unwrap_or_default();
            bytes.extend_from_slice(format!("{line}\n").as_bytes());
            fs::write(path, bytes).ok();
        }
    }
}

/// A session an older rig saved is refused before the harness writes
/// anything: resuming a copy of one, in the layout before `rig-session/2`
/// or in format `rig-session/3`, leaves every byte of its directory as it was, its
/// text log included, and what the refused run logged goes to the
/// launcher's run directory.
#[test]
fn resuming_a_session_an_older_rig_saved_changes_none_of_its_bytes() {
    let older: [(&str, fn(&SessionDir)); 2] = [
        ("1700000000-23", write_old_layout),
        ("1700000000-24", write_format_3),
    ];
    let home = scratch_home("rig-session-older");
    for (id, write) in older {
        let Ok(id) = id.parse::<SessionId>() else {
            return;
        };
        let session = home.session(&id);
        fs::create_dir_all(session.path()).ok();
        write(&session);
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
        app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, SessionLogPlugin))
            .add_plugins(SessionPlugin);
        drop(std::panic::take_hook());
        app.finish();
        app.update();
        app.update();
        let stored = app.world().contains_resource::<SessionStore>();
        let data = app.world().contains_resource::<PluginData>();
        drop((app, layer));
        assert!(!stored, "the refused session has no store");
        assert!(
            data,
            "a refused run has plugin data, its session's in memory"
        );
        assert_eq!(snapshot(session.path()), before);
        let refusal = fs::read_to_string(run.refusal()).unwrap_or_default();
        assert!(
            refusal.contains("made by another version of rig; start a new session"),
            "{refusal}"
        );
        let logged = fs::read_to_string(run.log()).unwrap_or_default();
        assert!(logged.starts_with("this run\n"), "{logged}");
        fs::remove_dir_all(run.path()).ok();
    }
    fs::remove_dir_all(home.root()).ok();
}

/// A fork that cannot be made says why to whoever asked, as data, beside
/// the error notice.
#[test]
fn a_failed_fork_tells_its_asker_why() {
    let home = scratch_home("rig-fork-failed");
    let Ok(id) = "1700000000-6".parse::<SessionId>() else {
        return;
    };
    let session = home.session(&id);
    fs::create_dir_all(session.path()).ok();
    let mut app = App::new();
    app.add_message::<Notice>()
        .init_resource::<Seen>()
        .insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session))
        .add_plugins(SessionPlugin)
        .add_systems(Last, watch.after(ForkSessions));
    drop(std::panic::take_hook());
    let asker = app.world_mut().spawn_empty().id();
    // No store: the session was never opened.
    app.world_mut().write_message(ForkSession {
        asker: Some(asker),
        ..ForkSession::default()
    });
    app.update();
    let seen = app.world().resource::<Seen>();
    let failed: Vec<_> = seen
        .fork_failed
        .iter()
        .map(|failed| (failed.from.clone(), failed.cut.clone(), failed.asker))
        .collect();
    assert_eq!(failed, [(None, ForkCut::Whole, Some(asker))]);
    let why = seen.fork_failed.first().map(|failed| failed.why.as_str());
    assert_eq!(why, Some("this session is not stored"));
    assert_eq!(seen.notices.len(), 1, "{:?}", seen.notices);
    fs::remove_dir_all(home.root()).ok();
}

/// The app of the saved session `id` of `home`, which says what forks
/// it made, with the agent's conversation `said` (an empty one when
/// `None`): the agent's entity.
fn running_with(home: &Home, id: &SessionId, said: &[Message]) -> Option<(App, Entity)> {
    let session = home.session(id);
    fs::create_dir_all(session.path()).ok()?;
    let mut app = App::new();
    app.init_resource::<Seen>()
        .insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session.clone()))
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin, SessionLogPlugin))
        .add_plugins(SessionPlugin)
        .add_systems(Last, watch.after(ForkSessions));
    drop(std::panic::take_hook());
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    for message in said {
        commit_message(app.world_mut(), agent, message.clone()).ok()?;
    }
    app.update();
    Some((app, agent))
}

/// Four messages, a question asked twice.
fn four() -> [Message; 4] {
    [
        Message::user("one"),
        Message::assistant("two"),
        Message::user("three"),
        Message::assistant("four"),
    ]
}

/// A split of this session's line through a message is a new session
/// holding that line, which names its parent and the message it copied
/// through, and leaves a pointer after that message; a move also takes
/// the range out of the line, the pointer in its place.
#[test]
fn a_split_of_this_session_copies_its_line_and_leaves_a_pointer() {
    let home = scratch_home("rig-split-here");
    let Ok(id) = "1700000000-30".parse::<SessionId>() else {
        return;
    };
    let Some((mut app, agent)) = running_with(&home, &id, &four()) else {
        return;
    };
    let world = app.world();
    let agent_id = world.get::<AgentId>(agent).cloned().unwrap_or_default();
    let ids: Vec<_> = world
        .get::<Conversation>(agent)
        .map(|conversation| conversation.entries().map(|entry| entry.id).collect())
        .unwrap_or_default();
    let (Some(&second), Some(&third), Some(&fourth)) = (ids.get(1), ids.get(2), ids.get(3)) else {
        return;
    };
    let line = |through, take| ForkCut::Line {
        agent: agent_id.clone(),
        line: None,
        through,
        move_from: take,
        title: "side".to_owned(),
    };
    app.world_mut().write_message(ForkSession {
        cut: line(third, None),
        ..ForkSession::default()
    });
    app.update();
    let forked = app.world().resource::<Seen>().forked.first().cloned();
    assert!(
        forked.is_some(),
        "no split: {:?}",
        app.world().resource::<Seen>().notices
    );
    let Some(forked) = forked else { return };
    let copy = dir_store(&home.session(&forked.to));
    let read = copy.as_ref().and_then(|copy| Session::read(copy).ok());
    let copied: Vec<_> = read
        .iter()
        .flat_map(|read| read.agents.iter())
        .flat_map(|agent| agent.conversation.entries().map(|entry| entry.id))
        .collect();
    assert_eq!(copied, ids.get(..3).unwrap_or_default());
    let parent = copy.as_ref().and_then(ForkedFrom::read);
    assert_eq!(
        parent,
        Some(ForkedFrom {
            session: id.to_string(),
            through: Some((agent_id.0.clone(), third)),
            line: Some("side".to_owned()),
            text: Some("three".to_owned()),
            ..ForkedFrom::default()
        })
    );
    let pointers = |app: &App| -> Vec<(Option<MessageId>, SplitPointer)> {
        let conversation = app.world().get::<Conversation>(agent);
        let asides = conversation
            .into_iter()
            .flat_map(|conversation| conversation.asides());
        asides
            .filter(|(_, aside)| aside.kind == SPLIT)
            .filter_map(|(_, aside)| {
                Some((
                    aside.after,
                    serde_json::from_value(aside.value.clone()).ok()?,
                ))
            })
            .collect()
    };
    let pointer = |to: &SessionId, moved| SplitPointer {
        session: to.to_string(),
        agent: agent_id.0.clone(),
        title: "side".to_owned(),
        moved,
        merged: false,
    };
    assert_eq!(pointers(&app), [(Some(third), pointer(&forked.to, 0))]);

    app.world_mut().write_message(ForkSession {
        cut: line(fourth, Some(second)),
        ..ForkSession::default()
    });
    app.update();
    let moved = app
        .world()
        .resource::<Seen>()
        .forked
        .get(1)
        .map(|forked| forked.to.clone());
    assert!(
        moved.is_some(),
        "no move: {:?}",
        app.world().resource::<Seen>().notices
    );
    let Some(moved) = moved else { return };
    let left: Vec<_> = app
        .world()
        .get::<Conversation>(agent)
        .map(|conversation| conversation.entries().map(|entry| entry.id).collect())
        .unwrap_or_default();
    assert_eq!(left, ids.get(..1).unwrap_or_default());
    let now = pointers(&app);
    assert_eq!(
        now.last(),
        Some(&(ids.first().copied(), pointer(&moved, 3)))
    );
    fs::remove_dir_all(home.root()).ok();
}

/// A split of a stopped session leaves its pointer there through
/// [`edit_stopped`](super::edit_stopped), which holds the session's lock;
/// a session another rig runs is copied without one, and a move out of it
/// is refused.
#[test]
fn a_split_of_a_stopped_session_leaves_its_pointer_unless_another_rig_runs_it() {
    let home = scratch_home("rig-split-stopped");
    let (Ok(here), Ok(there)) = (
        "1700000000-31".parse::<SessionId>(),
        "1700000000-32".parse::<SessionId>(),
    ) else {
        return;
    };
    let Some((app, agent)) = running_with(&home, &there, &four()) else {
        return;
    };
    let agent_id = app
        .world()
        .get::<AgentId>(agent)
        .cloned()
        .unwrap_or_default();
    let ids: Vec<_> = app
        .world()
        .get::<Conversation>(agent)
        .map(|conversation| conversation.entries().map(|entry| entry.id).collect())
        .unwrap_or_default();
    drop(app);
    let Some((mut app, _)) = running_with(&home, &here, &[]) else {
        return;
    };
    let (Some(&second), Some(&third)) = (ids.get(1), ids.get(2)) else {
        return;
    };
    let line = |take| ForkCut::Line {
        agent: agent_id.clone(),
        line: None,
        through: third,
        move_from: take,
        title: "side".to_owned(),
    };
    let stopped = home.session(&there);
    let lock = stopped.hold().ok().flatten();
    assert!(lock.is_some(), "held as another rig would");
    for take in [None, Some(second)] {
        app.world_mut().write_message(ForkSession {
            from: Some(there.clone()),
            cut: line(take),
            asker: None,
        });
        app.update();
    }
    drop(lock);
    let seen = app.world().resource::<Seen>();
    let notes: Vec<_> = seen
        .forked
        .iter()
        .map(|forked| forked.unfinished.is_some())
        .collect();
    assert_eq!(notes, [true], "copied without a pointer");
    let failed: Vec<_> = seen
        .fork_failed
        .iter()
        .map(|failed| failed.why.as_str())
        .collect();
    assert_eq!(failed, ["another rig runs it"]);

    app.world_mut().write_message(ForkSession {
        from: Some(there.clone()),
        cut: line(Some(second)),
        asker: None,
    });
    app.update();
    let seen = app.world().resource::<Seen>();
    let to = seen
        .forked
        .get(1)
        .map(|forked| (forked.to.clone(), forked.unfinished.clone()));
    assert!(matches!(to, Some((_, None))), "{to:?} {:?}", seen.notices);
    let read = dir_store(&stopped).and_then(|store| Session::read(&store).ok());
    let conversation = read
        .as_ref()
        .and_then(|read| read.agent(&agent_id.0))
        .map(|agent| &agent.conversation);
    let left: Vec<_> = conversation
        .map(|conversation| conversation.entries().map(|entry| entry.id).collect())
        .unwrap_or_default();
    let kept: Vec<_> = [ids.first(), ids.get(3)]
        .into_iter()
        .flatten()
        .copied()
        .collect();
    assert_eq!(left, kept, "the range left the stopped session");
    let pointers = conversation
        .into_iter()
        .flat_map(|conversation| conversation.asides())
        .filter(|(_, aside)| aside.kind == SPLIT)
        .count();
    assert_eq!(pointers, 1);
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
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin, SessionLogPlugin))
        .add_plugins(SessionPlugin)
        .add_systems(Last, watch.after(ForkSessions));
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
    let cut = before.clone().map_or(ForkCut::Whole, |(agent, message)| {
        ForkCut::Before(agent, message)
    });
    app.world_mut().write_message(ForkSession {
        cut,
        asker: Some(asker),
        ..ForkSession::default()
    });
    app.update();
    let forked = app.world().resource::<Seen>().forked.first().cloned();
    assert_eq!(forked.as_ref().and_then(|forked| forked.asker), Some(asker));
    let parent = forked.and_then(|forked| {
        let copy = dir_store(&home.session(&forked.to))?;
        ForkedFrom::read(&copy)
    });
    let text = parent.as_ref().and_then(|parent| parent.text.as_deref());
    assert_eq!(text, Some("two"), "{parent:?}");
    let before = before.map(|(agent, message)| (agent.0, message));
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
    // Any run that saved nothing leaves no directory either.
    assert_eq!(ended(false, AppExit::from_code(1)), Some(false));
    let log = std::fs::read_to_string(run.log()).ok();
    assert_eq!(log.as_deref(), Some("0-0"));
    std::fs::remove_dir_all(&root).ok();
}

/// A run that leaves its session before anything was saved there, such
/// as one switching to another session at startup, removes its directory,
/// whatever its front; a reload, which runs the session again, and a
/// saved session keep it. A switch the launcher is told of (its switch
/// file) keeps it too: the launcher reads where to go from there, then
/// removes the unsaved session itself.
#[test]
fn an_unsaved_session_left_for_good_leaves_no_directory() {
    let home = scratch_home("rig-unsaved-left");
    let left = |id: &str, reason: ExitReason, saved: bool| {
        let id: SessionId = id.parse().ok()?;
        let session = home.session(&id);
        fs::create_dir_all(session.path()).ok();
        if saved {
            session.mark_saved().ok()?;
        }
        if id.as_str() == "1700000000-20" {
            fs::write(session.switch(), "1700000000-19").ok()?;
        }
        let mut app = App::new();
        app.add_message::<Notice>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()))
            .add_plugins(SessionPlugin);
        app.update();
        app.world_mut().write_message(reason.exit());
        app.insert_resource(reason);
        app.update();
        Some(session.path().exists())
    };
    let switched = left("1700000000-16", ExitReason::Switch(None), false);
    let quit = left("1700000000-17", ExitReason::Quit, false);
    let reloaded = left("1700000000-18", ExitReason::Reload, false);
    let saved = left("1700000000-19", ExitReason::Switch(None), true);
    let told = left("1700000000-20", ExitReason::Switch(None), false);
    let switch_file = "1700000000-20"
        .parse()
        .ok()
        .map(|id| home.session(&id).switch());
    let switch_file = switch_file.and_then(|path| fs::read_to_string(path).ok());
    drop(std::panic::take_hook());
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(switched, Some(false), "switched away at startup");
    assert_eq!(quit, Some(false), "quit");
    assert_eq!(reloaded, Some(true), "reloaded");
    assert_eq!(saved, Some(true), "saved");
    assert_eq!(told, Some(true), "a switch the launcher is told of");
    assert_eq!(switch_file.as_deref(), Some("1700000000-19"));
}

/// A front whose runs keep no session never takes a resumed one, such as
/// `rig -r X --export Y`: its run is refused and the session stays. A
/// `/reload` of its own run is no resume.
#[test]
fn a_sessionless_front_refuses_a_resumed_session_and_keeps_it() {
    let home = scratch_home("rig-sessionless-resumed");
    let run = |reason: StartReason| {
        let id: SessionId = "1700000000-9".parse().ok()?;
        let session = home.session(&id);
        fs::create_dir_all(session.path()).ok();
        fs::write(session.started(), b"").ok();
        let mut app = App::new();
        app.add_message::<Notice>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()))
            .insert_resource(AppStart(reason))
            .add_plugins(SessionPlugin)
            .add_front(Front::new("export").headless().sessionless());
        let refusal = crate::host::cli::refusal(&mut app);
        app.world_mut().write_message(AppExit::Success);
        app.update();
        Some((refusal, session.path().exists()))
    };
    let resumed = run(StartReason::Resumed);
    let reloaded = run(StartReason::Reloaded);
    drop(std::panic::take_hook());
    fs::remove_dir_all(home.root()).ok();
    let why = "the front `export` keeps no session, so it cannot resume one";
    assert_eq!(resumed, Some((Some(why.to_owned()), true)));
    assert_eq!(reloaded, Some((None, false)));
}

/// A front with a store of its own keeps the session of a run it takes
/// there, such as `rig -r X reply` with a chat thread's store; any other
/// run keeps its session as [`StoreOpener`] does.
#[test]
fn a_fronts_own_store_keeps_only_the_sessions_of_its_runs() {
    let home = scratch_home("rig-front-store");
    let run = |command: Option<&str>| {
        let id: SessionId = "1700000000-11".parse().ok()?;
        let session = home.session(&id);
        fs::create_dir_all(session.path()).ok();
        let thread = MemoryStore::default();
        let opened = thread.clone();
        let reply = Front::new("reply")
            .on_command("reply", "", "Answer a thread")
            .session_store(move |_| Ok(Arc::new(opened.clone())));
        let invoked = Invoked {
            args: Invocation {
                command: command.map(str::to_owned),
                ..Invocation::default()
            },
            terminal: false,
        };
        let mut app = App::new();
        app.add_message::<Notice>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()))
            .insert_resource(invoked)
            .add_plugins(SessionPlugin)
            .add_front(reply)
            .add_front(Front::new("print").headless());
        app.finish();
        let mut said = conversation::Conversation::from_messages([Message::user("hi")]);
        let store = app.world().get_resource::<SessionStore>()?.0.clone();
        save(&*store, "agent-1", &mut said).ok()?;
        let kept_in_thread = !thread.logs().ok()?.is_empty();
        Some((app.runs_front("reply"), kept_in_thread))
    };
    let reply = run(Some("reply"));
    let print = run(None);
    drop(std::panic::take_hook());
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(reply, Some((true, true)));
    assert_eq!(print, Some((false, false)));
}

/// A sessionless run never marks its session saved, in a front's own
/// store (such as a question asked in memory) or in its directory, so a
/// crash leaves nothing `/resume` lists; a run that keeps its session
/// does, a front's own store included.
#[test]
fn a_sessionless_run_never_marks_its_session_saved() {
    let home = scratch_home("rig-sessionless-unmarked");
    let crashed = |id: &str, front: Front| {
        let id: SessionId = id.parse().ok()?;
        let session = home.session(&id);
        fs::create_dir_all(session.path()).ok();
        let mut app = App::new();
        app.add_message::<Notice>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()))
            .add_plugins((TaskPoolPlugin::default(), AgentPlugin, SessionLogPlugin))
            .add_plugins(SessionPlugin)
            .add_front(front);
        app.finish();
        app.update();
        let agent = first_agent(app.world_mut())?;
        commit_message(app.world_mut(), agent, Message::user("hi")).ok()?;
        app.update();
        let logged = !app
            .world()
            .get_resource::<SessionStore>()?
            .0
            .logs()
            .ok()?
            .is_empty();
        // No exit: the run is cut short.
        drop(app);
        logged.then_some(session.is_saved())
    };
    let in_memory = |front: Front| {
        let store = MemoryStore::default();
        front.session_store(move |_| Ok(Arc::new(store.clone())))
    };
    let asked = crashed(
        "1700000000-12",
        in_memory(Front::new("ask").headless().sessionless()),
    );
    let exported = crashed(
        "1700000000-13",
        Front::new("export").headless().sessionless(),
    );
    let replied = crashed("1700000000-14", in_memory(Front::new("reply").headless()));
    let printed = crashed("1700000000-15", Front::new("print").headless());
    drop(std::panic::take_hook());
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(asked, Some(false), "a sessionless front's own store");
    assert_eq!(exported, Some(false), "a sessionless run's directory");
    assert_eq!(replied, Some(true), "a front's own store");
    assert_eq!(printed, Some(true), "its directory");
}

#[test]
fn a_new_session_and_a_fork_start_new_and_a_second_start_resumes() {
    let root = std::env::temp_dir().join(format!("rig-first-{}", std::process::id()));
    let home = Home::at(&root);
    let id: SessionId = "1700000000-3"
        .parse()
        .unwrap_or_else(|_| SessionId::generate());
    let session = home.session(&id);
    std::fs::create_dir_all(session.path()).ok();
    std::fs::write(session.directory(), root.to_string_lossy().as_bytes()).ok();
    let start = |session: &rig_harness_protocol::SessionDir| {
        let mut app = App::new();
        app.add_message::<Notice>()
            .init_resource::<Seen>()
            .insert_resource(RigHome(home.clone()))
            .insert_resource(SessionPaths(session.clone()))
            .add_plugins(SessionPlugin)
            .add_systems(Last, watch.after(ForkSessions));
        app.finish();
        app.update();
        app
    };
    let reason = |app: &App| {
        app.world()
            .get_resource::<AppStart>()
            .map(|started| started.0)
    };
    let mut first = start(&session);
    assert_eq!(reason(&first), Some(StartReason::New));
    // A started session is resumed, whatever else, without a launcher.
    assert_eq!(reason(&start(&session)), Some(StartReason::Resumed));
    first.world_mut().write_message(ForkSession::default());
    first.update();
    let fork = first.world().resource::<Seen>().forked.first().cloned();
    assert!(fork.is_some(), "the session was forked");
    if let Some(fork) = fork {
        let fork = start(&home.session(&fork.to));
        assert_eq!(reason(&fork), Some(StartReason::New));
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
            .add_plugins((TaskPoolPlugin::default(), AgentPlugin, SessionLogPlugin))
            .add_plugins(SessionPlugin)
            .add_systems(Last, watch.after(ForkSessions));
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

/// The running session's store comes from the app's stores; the session
/// is marked saved once its log first writes, and stays saved after a
/// clean exit.
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
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin, SessionLogPlugin))
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
    let store = StoreOpener::default().open(&session).ok();
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

/// The frame a plugin's system last wrote the session's store in.
#[derive(Resource, Default)]
struct Frame(u32);

/// Writes the frame's number into a plugin file of the session's store.
fn note_frame(mut frame: ResMut<Frame>, store: Res<SessionStore>) {
    frame.0 += 1;
    let text = frame.0.to_string();
    store
        .0
        .write_file("plugins/notes/frame", text.as_bytes())
        .ok();
}

#[test]
fn a_fork_has_what_a_plugin_wrote_before_it_in_its_frame() {
    let home = scratch_home("rig-session-fork-order");
    let id: SessionId = "1700000000-3"
        .parse()
        .unwrap_or_else(|_| SessionId::generate());
    let session = home.session(&id);
    fs::create_dir_all(session.path()).ok();
    let Some(store) = dir_store(&session) else {
        return;
    };
    let mut said = conversation::Conversation::from_messages([Message::user("hi")]);
    assert!(save(&store, "agent-1", &mut said).is_ok());
    let mut app = App::new();
    app.init_resource::<Seen>()
        .init_resource::<Frame>()
        .insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session))
        .insert_resource(SessionStore::new(store))
        .add_plugins(SessionPlugin)
        .add_systems(
            Last,
            (
                note_frame.after(WriteLogs).before(ForkSessions),
                watch.after(ForkSessions),
            ),
        );
    app.update();
    app.world_mut().write_message(ForkSession::default());
    app.update();
    let forked = app.world().resource::<Seen>().forked.first().cloned();
    let copy = forked.and_then(|forked| dir_store(&home.session(&forked.to)));
    let noted = copy.and_then(|copy| copy.read_file("plugins/notes/frame").ok());
    assert_eq!(noted.as_deref(), Some(&b"2"[..]), "the fork's frame");
    drop(std::panic::take_hook());
    fs::remove_dir_all(home.root()).ok();
}

/// Replaces `sk-123` with `[REDACTED]` wherever a session keeps it.
fn redact() -> Strings<impl Fn(&mut String) -> bool + Send + Sync> {
    Strings(|text: &mut String| {
        let hit = text.contains("sk-123");
        if hit {
            *text = text.replace("sk-123", "[REDACTED]");
        }
        hit
    })
}

/// Whether a file under `dir` holds `sk-123`, and which.
fn secret_in(dir: &Path) -> Vec<PathBuf> {
    let files = snapshot(dir).into_iter();
    let holds = |bytes: &Vec<u8>| bytes.windows(6).any(|w| w == b"sk-123");
    files
        .filter(|(_, bytes)| bytes.as_ref().is_some_and(holds))
        .map(|(path, _)| path)
        .collect()
}

/// A rewrite of a stopped session reaches its scratch files too; the
/// running session, and one another rig runs, are refused untouched.
#[test]
fn a_rewrite_of_a_stopped_session_reaches_its_scratch_files() {
    let home = scratch_home("rig-session-rewrite-scratch");
    let ids = ["1700000000-4", "1700000000-5"].map(|id| id.parse::<SessionId>());
    let [Ok(id), Ok(other)] = ids else {
        return;
    };
    let (session, stopped) = (home.session(&id), home.session(&other));
    fs::create_dir_all(stopped.path().join("spill")).ok();
    fs::write(stopped.path().join("spill/shell-1.txt"), "key sk-123\n").ok();
    assert!(stopped.set_name(Some("key sk-123")).is_ok());
    let mut said = conversation::Conversation::from_messages([Message::user("sk-123")]);
    let saved = dir_store(&stopped).map(|store| save(&store, "agent-1", &mut said));
    assert!(saved.is_some_and(|saved| saved.is_ok()));
    fs::write(stopped.log(), "it said sk-123\n").ok();

    let stores = StoreOpener::default();
    let by = || Author::plugin("redact");
    let this = rewrite_stopped(&stores, &stopped, &stopped, by(), &redact());
    assert!(this.is_err(), "the running session is refused");
    let lock = File::create(stopped.launcher_lock()).ok();
    assert!(lock.as_ref().is_some_and(|lock| lock.try_lock().is_ok()));
    let elsewhere = rewrite_stopped(&stores, &session, &stopped, by(), &redact());
    assert_eq!(elsewhere.err().as_deref(), Some("another rig runs it"));
    assert_eq!(secret_in(stopped.path()).len(), 4, "nothing was changed");
    drop(lock);
    let done = rewrite_stopped(&stores, &session, &stopped, by(), &redact());
    assert_eq!(done.map(|done| done.logs.len()), Ok(1));
    assert_eq!(secret_in(stopped.path()), Vec::<PathBuf>::new());
    assert_eq!(stopped.name().as_deref(), Some("key [REDACTED]"));
    fs::remove_dir_all(home.root()).ok();
}

/// Saved values made secret by a rewrite; files are left.
struct Secret;

impl rig_memory::Rewrite for Secret {
    fn value(&self, _: &rig_memory::Place<'_>, _: &str, value: &mut serde_json::Value) -> bool {
        let hit = value.as_str() == Some("ship");
        if hit {
            *value = serde_json::Value::from("[secret]");
        }
        hit
    }

    fn file(&self, _: &str, _: &mut Vec<u8>) -> bool {
        false
    }
}

/// An edit and a rewrite of a stopped session keep its listing, as a
/// running rig does: a listed value set or rewritten is listed (an unlisted
/// one is not), the message count follows, and `updated` moves on.
#[test]
fn an_edit_or_a_rewrite_of_a_stopped_session_keeps_its_listing() {
    let home = scratch_home("rig-stopped-listing");
    let (Ok(here), Ok(there)) = (
        "1700000000-41".parse::<SessionId>(),
        "1700000000-42".parse::<SessionId>(),
    ) else {
        return;
    };
    let (session, stopped) = (home.session(&here), home.session(&there));
    let store = fs::create_dir_all(stopped.path())
        .ok()
        .and_then(|()| StoreOpener::default().open(&stopped).ok());
    let Some(store) = store else {
        return;
    };
    let mut said = conversation::Conversation::from_messages([Message::user("hi")]);
    assert!(save(&*store, "agent-1", &mut said).is_ok());
    let old = SessionMeta {
        title: "hi".to_owned(),
        updated: 1,
        messages: 1,
        ..SessionMeta::default()
    };
    assert!(old.write(&*store).is_ok());
    fs::create_dir_all(session.path()).ok();
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(session.clone()))
        .add_plugins(SessionPlugin);
    app.saved_as::<String>("goals.goal").listed();
    let _ = app.saved_as::<u32>("goals.count");
    drop(std::panic::take_hook());
    app.finish();
    app.update();
    let stores = app.world().resource::<StoreOpener>().clone();
    let edited = super::edit_stopped(&stores, &session, &stopped, |read| {
        read.saved.insert("goals.goal".to_owned(), "ship".into());
        read.saved.insert("goals.count".to_owned(), 3.into());
        let agent = read.agent_mut("agent-1").ok_or("no agent")?;
        agent.conversation.push(Message::assistant("hello"));
        Ok(())
    });
    assert_eq!(edited, Ok(()));
    let listed =
        |goal: &str| BTreeMap::from([("goals.goal".to_owned(), serde_json::Value::from(goal))]);
    let meta = SessionMeta::read(&*store).unwrap_or_default();
    assert_eq!(
        (meta.title.as_str(), meta.messages, &meta.listed),
        ("hi", 2, &listed("ship"))
    );
    assert!(meta.updated > 1, "updated");

    assert!(old.write(&*store).is_ok());
    let by = Author::plugin("redact");
    let rewritten = rewrite_stopped(&stores, &session, &stopped, by, &Secret);
    assert!(rewritten.is_ok());
    let meta = SessionMeta::read(&*store).unwrap_or_default();
    assert_eq!((meta.messages, &meta.listed), (2, &listed("[secret]")));
    assert!(meta.updated > 1, "updated");
    drop(app);
    fs::remove_dir_all(home.root()).ok();
}
