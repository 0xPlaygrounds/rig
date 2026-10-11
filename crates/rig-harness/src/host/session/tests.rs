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
use rig_harness_test_support::{commit_message, first_agent};

use super::{
    DeleteSession, ForkSession, ForkedFrom, SessionDeleted, SessionForked, SessionPaths,
    SessionPlugin,
};
use crate::host::cli::{AppCliExt, Front};
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
    // The copy names the session it came from.
    let parent = ForkedFrom {
        session: id.clone(),
        before: None,
        text: None,
    };
    assert_eq!(ForkedFrom::of(&copy), Some(parent));
    assert_eq!(ForkedFrom::of(&session), None);

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
fn a_fork_before_a_message_keeps_its_text_with_its_parent() {
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
    let seq = world
        .get::<Conversation>(agent)
        .and_then(|conversation| conversation.seq(2));
    let before = agent_id.zip(seq);
    assert!(before.is_some(), "the message was logged");
    app.world_mut().trigger(ForkSession {
        before: before.clone(),
    });
    app.update();
    let forked = app.world().resource::<Seen>().forked.first().cloned();
    let parent = forked.and_then(|forked| ForkedFrom::of(&home.session(&forked.to)));
    let text = parent.as_ref().and_then(|parent| parent.text.as_deref());
    assert_eq!(text, Some("two"), "{parent:?}");
    assert_eq!(parent.and_then(|parent| parent.before), before);
    drop(std::panic::take_hook());
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn a_sessionless_front_leaves_no_directory_after_a_run_that_ended_well() {
    let root = std::env::temp_dir().join(format!("rig-sessionless-{}", std::process::id()));
    let home = Home::at(&root);
    let ended = |sessionless: bool, exit: AppExit| {
        let id = format!("{}-{}", u8::from(sessionless), exit.is_success());
        let id: SessionId = id.replace("true", "1").replace("false", "0").parse().ok()?;
        let session = home.session(&id);
        std::fs::create_dir_all(session.path()).ok();
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
            .add_plugins(SessionPlugin)
            .add_front(front);
        assert!(app.runs_front("replay"));
        app.world_mut().write_message(exit);
        app.update();
        Some(session.path().exists())
    };
    assert_eq!(ended(true, AppExit::Success), Some(false));
    let failed = ended(true, AppExit::from_code(1));
    assert_eq!(failed, Some(true), "a failed run keeps its log");
    assert_eq!(ended(false, AppExit::Success), Some(true));
    std::fs::remove_dir_all(&root).ok();
}
