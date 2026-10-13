use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex, PoisonError};

use rig_core::completion::Message;
use rig_ecs::AgentPlugin;
use rig_ecs::journal::JournalPlugin;
use rig_ecs::prelude::{Author, RewriteSession, SavedAgent, Session, Store, Strings};
use rig_harness::harness_protocol::Home;
use rig_harness::prelude::{ForkSession, ForkedFrom, SessionForked};
use rig_harness_test_support::{MemoryStore, app_in, commit_message, first_agent, scratch_home};

use super::*;

/// An empty home of this test's own.
fn home(test: &str) -> Home {
    scratch_home(&format!("rig-sessions-{test}"))
}

/// An app running session `id` under `home`, which an agent logged to, so
/// it is saved.
fn running(home: &Home, id: &str) -> Option<App> {
    let id: SessionId = id.parse().ok()?;
    let dir = home.session(&id);
    fs::create_dir_all(dir.path()).ok()?;
    let store = SessionStores::default().open(&dir).ok()?;
    store.append_log("a1", b"").ok()?;
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(dir))
        .insert_resource(SessionStore(store))
        .add_plugins((AgentPlugin, SessionsPlugin));
    Some(app)
}

/// The `/resume` lines of the sessions under `home`.
fn lines(home: &Home) -> Vec<String> {
    let listed = list(home, &SessionStores::default(), Path::new("/nowhere"));
    listed.iter().map(SessionEntry::label).collect()
}

#[test]
fn a_label_is_kept_with_the_session_and_shown_in_its_line() {
    let home = home("label");
    let Some(mut app) = running(&home, "1-1") else {
        return;
    };
    let label = app
        .world_mut()
        .spawn(ResumeLabel::new("goal", "Ship the\nparser"))
        .id();
    app.update();
    let shown = lines(&home);
    assert_eq!(shown.len(), 1);
    assert!(
        shown
            .iter()
            .all(|line| line.contains("  · Ship the parser")),
        "{shown:?}"
    );

    app.world_mut().despawn(label);
    app.update();
    let shown = lines(&home);
    assert!(
        shown.iter().all(|line| !line.contains("parser")),
        "{shown:?}"
    );
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn naming_another_session_changes_only_its_title() {
    let home = home("name");
    let other: SessionId = "1-1".parse().unwrap_or_else(|_| SessionId::generate());
    let dir = home.session(&other);
    fs::create_dir_all(dir.path()).ok();
    let store = SessionStores::default().open(&dir);
    assert!(store.is_ok(), "the default store opens");
    let Ok(store) = store else { return };
    store.append_log("a1", b"").ok();
    let before = SessionMeta {
        title: "Fix the parser".to_owned(),
        labels: BTreeMap::from([("spent".to_owned(), "$0.50".to_owned())]),
        updated: 1000,
    };
    assert!(before.write(&*store).is_ok());
    let Some(mut app) = running(&home, "2-2") else {
        return;
    };
    app.world_mut().trigger(NameSession {
        session: other,
        name: "The parser".to_owned(),
    });
    app.update();
    let name = SessionTitle::read(&*store).and_then(|title| title.name);
    assert_eq!(name.as_deref(), Some("The parser"));
    assert_eq!(SessionMeta::read(&*store), Some(before));
    assert_eq!(app.world().resource::<SessionTitle>().name, None);
    let running: SessionId = "2-2".parse().unwrap_or_else(|_| SessionId::generate());
    let stores = SessionStores::default();
    let listed = list(&home, &stores, home.session(&running).path());
    let labels: Vec<String> = listed.iter().map(SessionEntry::label).collect();
    assert_eq!(labels, ["The parser  · $0.50"]);
    fs::remove_dir_all(home.root()).ok();
}

/// Stores a plugin keeps in memory, one per session directory.
#[derive(Clone, Default)]
struct InMemory(Arc<Mutex<HashMap<PathBuf, MemoryStore>>>);

impl InMemory {
    /// The store of the session in `dir`.
    fn of(&self, dir: &Path) -> MemoryStore {
        let mut stores = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        stores.entry(dir.to_owned()).or_default().clone()
    }

    /// The app's stores, as a plugin puts them in.
    fn stores(&self) -> SessionStores {
        let kept = self.clone();
        SessionStores::new(move |dir| Ok(Arc::new(kept.of(dir.path())) as Arc<dyn Store>))
    }
}

/// A plugin's store holds every file of the session, a fork's included,
/// and the session directory only the plain files the launcher reads; the
/// fork is saved, so `/resume` lists it, and the session stays saved
/// after a clean exit.
#[test]
fn a_store_a_plugin_puts_in_holds_every_session_file_and_a_fork_is_listed() {
    let home = home("stores");
    let id = SessionId::generate();
    let dir = home.session(&id);
    let memory = InMemory::default();
    let (mut app, _wakes) = app_in(&home, &id);
    app.insert_resource(memory.stores())
        .add_plugins((AgentPlugin, JournalPlugin, SessionsPlugin));
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let said = commit_message(app.world_mut(), agent, Message::user("Fix the parser"));
    assert_eq!(said, Ok(()));
    app.world_mut().resource_mut::<SessionTitle>().name = Some("Parser".to_owned());
    app.update();
    app.world_mut().trigger(ForkSession::default());
    app.update();
    let mut forks = app.world_mut().resource_mut::<Messages<SessionForked>>();
    let fork = forks.drain().next().map(|forked| forked.to);
    assert!(fork.is_some(), "the session was forked");
    let Some(fork) = fork else { return };
    app.world_mut().write_message(AppExit::Success);
    app.update();
    drop(app);

    let session = memory.of(dir.path());
    assert!(session.logs().is_ok_and(|logs| !logs.is_empty()));
    let name = SessionTitle::read(&session).and_then(|title| title.name);
    assert_eq!(name.as_deref(), Some("Parser"));
    let copy = home.session(&fork);
    let forked = memory.of(copy.path());
    assert!(forked.logs().is_ok_and(|logs| !logs.is_empty()));
    let parent = ForkedFrom::read(&forked).map(|parent| parent.session);
    assert_eq!(parent, Some(id.to_string()));

    // On disk, only the plain markers the launcher reads.
    let plain = |dir: &SessionDir| -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir.path())
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
            .collect();
        names.sort();
        names
    };
    assert_eq!(plain(&dir), ["saved", "started"]);
    assert_eq!(plain(&copy), ["saved"]);
    assert!(dir.is_saved() && copy.is_saved());

    let listed = list(&home, &memory.stores(), Path::new("/nowhere"));
    let ids: Vec<_> = listed.iter().map(|entry| entry.id.clone()).collect();
    assert!(ids.contains(&fork) && ids.contains(&id), "{ids:?}");
    fs::remove_dir_all(home.root()).ok();
}

/// A session written from data, such as an import, through the app's
/// stores is saved, so `/resume` lists it.
#[test]
fn a_session_written_through_the_stores_is_listed() {
    let home = home("import");
    let made = home.new_session();
    assert!(made.is_ok(), "{made:?}");
    let Ok((id, dir)) = made else { return };
    let stores = SessionStores::default();
    let mut agent = SavedAgent {
        id: "imported".to_owned(),
        ..SavedAgent::default()
    };
    agent.conversation.push(Message::user("Port the CLI"));
    let mut session = Session {
        agents: vec![agent],
        ..Session::default()
    };
    let written = stores.open(&dir).and_then(|store| session.save(&*store));
    assert!(written.is_ok(), "{written:?}");
    let listed = list(&home, &stores, Path::new("/nowhere"));
    let ids: Vec<_> = listed.iter().map(|entry| entry.id.clone()).collect();
    assert_eq!(ids, [id]);
    fs::remove_dir_all(home.root()).ok();
}

#[cfg(feature = "tui")]
#[test]
fn resume_picks_from_the_sessions_of_the_apps_home() {
    let home = home("resume");
    let Some(mut earlier) = running(&home, "1-1") else {
        return;
    };
    earlier
        .world_mut()
        .spawn(ResumeLabel::new("goal", "Ship it"));
    earlier.update();
    let Some(mut app) = running(&home, "2-2") else {
        return;
    };
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    world.trigger(rig_harness::commands::RunCommand::new(
        agent,
        "resume".to_owned(),
    ));
    world.flush();
    let mut pickers = world.query::<&rig_harness::pick::Picker>();
    let items: Vec<_> = pickers
        .iter(world)
        .flat_map(|picker| picker.items.clone())
        .collect();
    let commands: Vec<&str> = items.iter().map(|item| item.command.as_str()).collect();
    assert_eq!(commands, ["resume 1-1"]);
    assert!(
        items.iter().all(|item| item.label.contains("Ship it")),
        "{items:?}"
    );
    fs::remove_dir_all(home.root()).ok();
}

/// A session resumed as it was listed is not written again, so it keeps
/// its place in `/resume`.
#[test]
fn a_session_resumed_as_it_was_listed_keeps_its_listing() {
    let home = home("resumed");
    let Some(mut app) = running(&home, "1-1") else {
        return;
    };
    let store = app.world().resource::<SessionStore>().0.clone();
    let listed = SessionMeta {
        labels: BTreeMap::from([("goal".to_owned(), "Ship it".to_owned())]),
        updated: 1000,
        ..SessionMeta::default()
    };
    assert!(listed.write(&*store).is_ok());
    app.world_mut().spawn(ResumeLabel::new("goal", "Ship it"));
    app.update();
    app.update();
    assert_eq!(
        SessionMeta::read(&*store).map(|meta| meta.updated),
        Some(1000)
    );
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn a_rewritten_session_keeps_its_rewritten_title() {
    const SECRET: &str = "sk-123";
    let home = home("rewrite");
    let Some(mut app) = running(&home, "1-1") else {
        return;
    };
    app.add_plugins(JournalPlugin);
    app.finish();
    app.update();
    app.world_mut().resource_mut::<SessionTitle>().name = Some(format!("key {SECRET}"));
    app.update();
    let store = app.world().resource::<SessionStore>().0.clone();
    let name = || SessionTitle::read(&*store).and_then(|title| title.name);
    assert_eq!(name(), Some(format!("key {SECRET}")));

    app.world_mut().trigger(RewriteSession {
        by: Author::plugin("redact"),
        with: Arc::new(Strings(|text: &mut String| {
            let hit = text.contains(SECRET);
            if hit {
                *text = text.replace(SECRET, "[REDACTED]");
            }
            hit
        })),
        asker: None,
    });
    app.update();
    // The next write of the listing keeps the title as rewritten.
    app.world_mut().spawn(ResumeLabel::new("goal", "ship"));
    app.update();
    assert_eq!(name().as_deref(), Some("key [REDACTED]"));
    let title = app.world().resource::<SessionTitle>().name.clone();
    assert_eq!(title.as_deref(), Some("key [REDACTED]"));
    fs::remove_dir_all(home.root()).ok();
}
