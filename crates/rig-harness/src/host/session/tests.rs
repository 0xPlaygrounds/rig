use std::fs::File;

use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig::harness_protocol::{Home, SessionId};
use rig_cassette::journal::{JournalStore, JsonlDirStore};
use rig_core::completion::Message;
use rig_ecs::AgentPlugin;
use rig_ecs::agent::{AgentId, Conversation, Notice};
use rig_ecs::journal::{JournalPlugin, SessionStore};
use rig_harness_test_support::{commit_message, first_agent, scratch_home};

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

#[test]
fn a_fork_copies_the_session_and_another_session_can_be_deleted() {
    let root = std::env::temp_dir().join(format!("rig-fork-{}", std::process::id()));
    let home = Home::at(&root);
    let id: SessionId = "1700000000-1"
        .parse()
        .unwrap_or_else(|_| SessionId::generate());
    let session = home.session(&id);
    std::fs::create_dir_all(session.path()).ok();
    std::fs::write(session.directory(), root.to_string_lossy().as_bytes()).ok();
    let store = JsonlDirStore::new(session.path());
    store.append("agent-1", b"{\"line\":1}\n").ok();
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
    assert_eq!(
        JsonlDirStore::new(copy.path()).read("agent-1").ok(),
        Some(b"{\"line\":1}\n".to_vec())
    );
    assert_eq!(copy.working_directory(), Some(root.clone()));
    // The copy names the session it came from, and is saved, so it can be
    // resumed.
    let parent = ForkedFrom {
        session: id.clone(),
        before: None,
        text: None,
    };
    assert_eq!(
        ForkedFrom::of(&JsonlDirStore::new(copy.path())),
        Some(parent)
    );
    assert_eq!(ForkedFrom::of(&JsonlDirStore::new(session.path())), None);
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
    std::fs::remove_dir_all(&root).ok();
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
        .insert_resource(SessionStore::new(JsonlDirStore::new(session.path())))
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin, JournalPlugin))
        .add_plugins(SessionPlugin)
        .add_systems(Last, watch.after(super::fork_sessions));
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
    let parent = forked
        .and_then(|forked| ForkedFrom::of(&JsonlDirStore::new(home.session(&forked.to).path())));
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
            .insert_resource(SessionStore::new(JsonlDirStore::new(session.path())))
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
            .insert_resource(SessionStore::new(JsonlDirStore::new(session.path())))
            .add_plugins((TaskPoolPlugin::default(), AgentPlugin, JournalPlugin))
            .add_plugins(SessionPlugin)
            .add_systems(Last, watch.after(super::fork_sessions));
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
    let read = SessionStores::default()
        .open(&session)
        .ok()
        .and_then(|store| store.agents().ok());
    assert!(read.is_some_and(|agents| !agents.is_empty()));
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

#[test]
fn what_the_user_typed_leaves_out_attached_files_and_delivered_notes() {
    use rig_cassette::journal::Origin;
    use rig_ecs::journal::Commit;

    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin));
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "the app has an agent");
    let Some(agent) = agent else { return };
    let attached = "<file path=\"a.txt\">\nfile text\n</file>";
    let messages = [
        (Message::user("build failed"), Some(Origin::plugin("build"))),
        (Message::assistant("ok"), None),
        (Message::user(attached), None),
        (Message::user("  fix it  "), None),
        (Message::user("and test"), None),
    ];
    let commit = |In((agent, messages)): In<(Entity, Vec<(Message, Option<Origin>)>)>,
                  mut agents: Query<&mut Conversation>,
                  mut commit: Commit| {
        if let Ok(mut conversation) = agents.get_mut(agent) {
            for (message, origin) in messages {
                commit.message_from(agent, &mut conversation, message, origin);
            }
        }
    };
    let committed = app
        .world_mut()
        .run_system_cached_with(commit, (agent, messages.to_vec()));
    assert!(committed.is_ok());
    let conversation = app.world().get::<Conversation>(agent);
    assert!(conversation.is_some(), "the agent has a conversation");
    let Some(conversation) = conversation else {
        return;
    };
    let both = Some("fix it\n\nand test".to_owned());
    assert_eq!(typed(conversation, 0), None);
    assert_eq!(typed(conversation, 2), both);
    assert_eq!(first_typed(conversation), both);
}
