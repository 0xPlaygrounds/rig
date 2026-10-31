use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use bevy_ecs::message::Messages;
use rig_core::completion::Message;
use rig_ecs::AgentPlugin;
use rig_ecs::inbox::Recalled;
use rig_ecs::prelude::{SavedAgent, Session, Store};
use rig_ecs::session_log::SessionLogPlugin;
use rig_harness::commands::RunCommand;
use rig_harness::harness_protocol::Home;
use rig_harness::prelude::{AppKeepsExt, ReflectSaved, StoreOpener};
use rig_harness_test_support::{MemoryStore, app_in, commit_message, first_agent, scratch_home};

use super::*;
use crate::session_picker::{SessionEntry, SessionSource};

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
    let store = StoreOpener::default().open(&dir).ok()?;
    store.append_log("a1", 0, b"").ok()?;
    dir.mark_saved().ok()?;
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(dir))
        .insert_resource(SessionStore(store))
        .add_plugins((AgentPlugin, SessionsPlugin));
    Some(app)
}

/// The saved sessions under `home` as a session listing reads them, each
/// store opened by `stores`.
fn list(home: &Home, stores: &StoreOpener) -> Vec<SessionEntry> {
    let saved = home.saved_sessions();
    let read = saved.iter().map(|(_, dir)| SessionEntry::read(dir, stores));
    read.flatten().collect()
}

/// The `/resume` lines of the sessions under `home`.
fn lines(home: &Home) -> Vec<String> {
    let listed = list(home, &StoreOpener::default());
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
    let store = StoreOpener::default().open(&dir);
    assert!(store.is_ok(), "the default store opens");
    let Ok(store) = store else { return };
    store.append_log("a1", 0, b"").ok();
    dir.mark_saved().ok();
    let before = SessionMeta {
        title: "Fix the parser".to_owned(),
        labels: BTreeMap::from([("spent".to_owned(), "$0.50".to_owned())]),
        updated: 1000,
        ..SessionMeta::default()
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
    assert_eq!(dir.name().as_deref(), Some("The parser"));
    assert_eq!(SessionMeta::read(&*store), Some(before));
    assert_eq!(app.world().resource::<SessionTitle>().name, None);
    let running: SessionId = "2-2".parse().unwrap_or_else(|_| SessionId::generate());
    let stores = StoreOpener::default();
    let listed = list(&home, &stores);
    let others = listed.iter().filter(|entry| entry.id != running);
    let labels: Vec<String> = others.map(SessionEntry::label).collect();
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
    fn stores(&self) -> StoreOpener {
        let kept = self.clone();
        StoreOpener::new(move |dir| Ok(Arc::new(kept.of(dir.path())) as Arc<dyn Store>))
    }
}

/// A plugin's store holds every file of the session, and the session
/// directory only the plain files the launcher reads; the session stays
/// saved after a clean exit, so `/resume` lists it.
#[test]
fn a_store_a_plugin_puts_in_holds_every_session_file_and_the_session_is_listed() {
    let home = home("stores");
    let id = SessionId::generate();
    let dir = home.session(&id);
    let memory = InMemory::default();
    let (mut app, _wakes) = app_in(&home, &id);
    app.insert_resource(memory.stores()).add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        SessionsPlugin,
    ));
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let said = commit_message(app.world_mut(), agent, Message::user("Fix the parser"));
    assert_eq!(said, Ok(()));
    app.world_mut().resource_mut::<SessionTitle>().name = Some("Parser".to_owned());
    app.update();
    app.world_mut().write_message(AppExit::Success);
    app.update();
    drop(app);

    let session = memory.of(dir.path());
    assert!(session.logs().is_ok_and(|logs| !logs.is_empty()));
    let name = SessionTitle::read(&dir, &session).and_then(|title| title.name);
    assert_eq!(name.as_deref(), Some("Parser"));

    // On disk, only the plain files the launcher reads.
    let plain = |dir: &SessionDir| -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir.path())
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
            .collect();
        names.sort();
        names
    };
    assert_eq!(plain(&dir), ["name", "saved", "started"]);
    assert!(dir.is_saved());

    let listed = list(&home, &memory.stores());
    let ids: Vec<_> = listed.iter().map(|entry| entry.id.clone()).collect();
    assert_eq!(ids, [id]);
    fs::remove_dir_all(home.root()).ok();
}

/// A session written from data, such as an import, through the app's
/// stores and marked saved is listed by `/resume`.
#[test]
fn a_session_written_through_the_stores_is_listed() {
    let home = home("import");
    let made = home.new_session();
    assert!(made.is_ok(), "{made:?}");
    let Ok((id, dir)) = made else { return };
    let stores = StoreOpener::default();
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
    assert!(dir.mark_saved().is_ok());
    let listed = list(&home, &stores);
    let ids: Vec<_> = listed.iter().map(|entry| entry.id.clone()).collect();
    assert_eq!(ids, [id]);
    fs::remove_dir_all(home.root()).ok();
}

/// `/resume` opens the native session picker over the saved sessions,
/// picking one resumes it; with no other session it says so.
#[test]
fn resume_opens_the_session_picker() {
    let home = home("resume");
    let Some(mut app) = running(&home, "2-2") else {
        return;
    };
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let resume = || rig_harness::commands::RunCommand::new(agent, "resume".to_owned());
    world.trigger(resume());
    world.flush();
    assert_eq!(world.query::<&SessionPicker>().iter(world).count(), 0);
    let Some(mut earlier) = running(&home, "1-1") else {
        return;
    };
    earlier.update();
    let world = app.world_mut();
    world.trigger(resume());
    world.flush();
    let pickers: Vec<SessionPicker> = world
        .query::<&SessionPicker>()
        .iter(world)
        .cloned()
        .collect();
    let commands: Vec<&str> = pickers
        .iter()
        .map(|picker| picker.command.as_str())
        .collect();
    assert_eq!(commands, ["resume {session}"]);
    assert!(
        pickers
            .iter()
            .all(|picker| picker.source == SessionSource::Saved)
    );
    fs::remove_dir_all(home.root()).ok();
}

/// A plugin's goal, listed.
#[derive(Resource, Reflect, Clone, Default)]
#[reflect(Resource, Saved, Clone, Default)]
struct Goal(String);

/// A listed value and the main agent's model are kept in the session's
/// listing as they change, so a listing shows them without the logs.
#[test]
fn a_listed_value_and_the_model_are_kept_in_the_listing() {
    let home = home("listed");
    let Some(mut app) = running(&home, "1-1") else {
        return;
    };
    app.register_type::<Goal>()
        .insert_resource(Goal("ship the parser".to_owned()))
        .saved_as::<Goal>("goals.goal")
        .listed();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut()
        .entity_mut(agent)
        .insert(ModelChoice("openai/gpt-5".to_owned()));
    app.update();
    let store = app.world().resource::<SessionStore>().0.clone();
    let meta = SessionMeta::read(&*store).unwrap_or_default();
    let goal = meta.listed.get("goals.goal").cloned();
    assert_eq!(goal, Some(serde_json::json!("ship the parser")));
    assert_eq!(meta.model, "openai/gpt-5");

    app.world_mut().resource_mut::<Goal>().0 = "ship the lexer".to_owned();
    app.update();
    let meta = SessionMeta::read(&*store).unwrap_or_default();
    let goal = meta.listed.get("goals.goal").cloned();
    assert_eq!(goal, Some(serde_json::json!("ship the lexer")));
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

/// The exits a session asked for.
#[derive(Resource, Default)]
struct Exits(Vec<ExitReason>);

#[test]
fn new_refuses_text_without_switching_sessions() {
    let home = home("new-args");
    let running = running(&home, "1-1");
    assert!(running.is_some());
    let Some(mut app) = running else { return };
    app.init_resource::<Exits>()
        .add_observer(|exit: On<Exit>, mut exits: ResMut<Exits>| exits.0.push(exit.0.clone()));
    let mut conversation = Conversation::default();
    conversation.push(Message::user("Keep this session"));
    let world = app.world_mut();
    let agent = world.spawn((Agent, conversation)).id();
    world.trigger(RunCommand::new(agent, "new PELICAN"));
    world.flush();
    let recalled: Vec<_> = world
        .resource::<Messages<Recalled>>()
        .iter_current_update_messages()
        .collect();
    assert_eq!(recalled.len(), 1);
    assert!(recalled.iter().all(|recalled| {
        recalled.agent == agent
            && recalled.text == "/new PELICAN"
            && recalled
                .why
                .as_ref()
                .is_some_and(|why| why.contains("/new") && why.contains("PELICAN"))
    }));
    assert!(world.resource::<Exits>().0.is_empty());
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn name_refuses_another_saved_sessions_normalized_name_but_accepts_its_own() {
    let home = home("unique-command-name");
    let saved = home.new_session();
    assert!(saved.is_ok());
    let Ok((other, dir)) = saved else { return };
    // Both spellings shorten to the same stored title.
    let name = "P".repeat(65);
    assert!(dir.set_name(Some(&title_of(&name))).is_ok());
    assert!(dir.mark_saved().is_ok());
    let running = running(&home, "1-1");
    assert!(running.is_some());
    let Some(mut app) = running else { return };
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let line = format!("name   {name}  extra");
    world.trigger(RunCommand::new(agent, line.clone()));
    world.flush();
    let recalled: Vec<_> = world.resource_mut::<Messages<Recalled>>().drain().collect();
    assert_eq!(recalled.len(), 1);
    assert!(recalled.iter().all(|recalled| {
        recalled.text == format!("/{line}")
            && recalled
                .why
                .as_ref()
                .is_some_and(|why| why.contains(other.as_str()) && why.contains(&title_of(&name)))
    }));
    assert_eq!(world.resource::<SessionTitle>().name, None);
    for _ in 0..2 {
        world.trigger(RunCommand::new(agent, "name  Own\n name"));
        world.flush();
    }
    assert_eq!(
        world.resource::<SessionTitle>().name.as_deref(),
        Some("Own name")
    );
    assert!(world.resource::<Messages<Recalled>>().is_empty());
    app.update();
    let world = app.world_mut();
    world.trigger(RunCommand::new(agent, "name Own name"));
    world.flush();
    assert!(world.resource::<Messages<Recalled>>().is_empty());
    assert_eq!(
        world.resource::<SessionPaths>().name().as_deref(),
        Some("Own name")
    );
    fs::remove_dir_all(home.root()).ok();
}

#[test]
fn naming_another_session_refuses_the_current_sessions_pending_name() {
    let home = home("unique-event-name");
    let saved = home.new_session();
    assert!(saved.is_ok());
    let Ok((other, dir)) = saved else { return };
    assert!(dir.mark_saved().is_ok());
    let running = running(&home, "1-1");
    assert!(running.is_some());
    let Some(mut app) = running else { return };
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    world.trigger(RunCommand::new(agent, "name Old"));
    world.flush();
    app.update();
    let world = app.world_mut();
    world.trigger(RunCommand::new(agent, "name  Own\n name"));
    world.flush();
    assert_eq!(
        world.resource::<SessionPaths>().name().as_deref(),
        Some("Old")
    );
    world.trigger(NameSession {
        session: other.clone(),
        name: "Own  name".to_owned(),
    });
    world.flush();
    let errors: Vec<_> = world
        .resource_mut::<Messages<Notice>>()
        .drain()
        .filter(|notice| notice.level == rig_ecs::agent::NoticeLevel::Error)
        .collect();
    assert_eq!(errors.len(), 1);
    assert!(
        errors
            .iter()
            .all(|notice| { notice.text.contains("1-1") && notice.text.contains("Own name") })
    );
    assert_eq!(dir.name(), None);
    let current = world.resource::<SessionPaths>().id();
    assert!(current.is_some());
    if let Some(current) = current {
        world.trigger(NameSession {
            session: current,
            name: "Own name".to_owned(),
        });
        world.flush();
    }
    assert!(world.resource::<Messages<Notice>>().is_empty());
    assert_eq!(
        world.resource::<SessionTitle>().name.as_deref(),
        Some("Own name")
    );
    world.trigger(NameSession {
        session: other,
        name: "Old".to_owned(),
    });
    world.flush();
    assert!(world.resource::<Messages<Notice>>().is_empty());
    assert_eq!(dir.name().as_deref(), Some("Old"));
    fs::remove_dir_all(home.root()).ok();
}

/// A name given in one session is kept in its directory, and `/resume`
/// in another runs the session it names.
#[test]
fn resume_runs_the_session_a_name_names() {
    let home = home("resume-name");
    let Some(mut named) = running(&home, "1-1") else {
        return;
    };
    named.world_mut().resource_mut::<SessionTitle>().name = Some("Parser".to_owned());
    named.update();
    let Some(mut app) = running(&home, "2-2") else {
        return;
    };
    app.init_resource::<Exits>()
        .add_observer(|exit: On<Exit>, mut exits: ResMut<Exits>| exits.0.push(exit.0.clone()));
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    for args in ["resume Parser", "resume Nobody"] {
        world.trigger(rig_harness::commands::RunCommand::new(agent, args));
    }
    world.flush();
    let exits = world.resource::<Exits>().0.clone();
    let named: Option<SessionId> = "1-1".parse().ok();
    let kept = named.as_ref().and_then(|id| home.session(id).name());
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(kept.as_deref(), Some("Parser"));
    assert_eq!(exits, [ExitReason::Switch(named)]);
}

#[test]
fn an_unopened_or_refused_store_keeps_the_session_name_untouched()
-> Result<(), Box<dyn std::error::Error>> {
    let home = home("refused-name");
    let id = SessionId::generate();
    let dir = home.session(&id);
    fs::create_dir_all(dir.path())?;
    dir.set_name(Some("preserve this name"))?;
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(dir.clone()))
        .add_plugins((AgentPlugin, SessionsPlugin));
    app.update();
    if dir.name().as_deref() != Some("preserve this name") {
        return Err(std::io::Error::other("unopened store changed session name").into());
    }
    fs::remove_dir_all(home.root())?;
    Ok(())
}
