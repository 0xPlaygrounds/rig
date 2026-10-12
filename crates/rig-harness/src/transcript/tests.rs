use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_cassette::journal::MemoryStore;
use rig_ecs::agent::Agent;
use rig_ecs::journal::{JournalPlugin, SessionStore};

use super::{EntryOf, TranscriptEntries, TranscriptEntry, TranscriptPlugin};

/// An app on `store` with the transcript plugin, after its first frame
/// restored the session.
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((rig_ecs::AgentPlugin, JournalPlugin, TranscriptPlugin))
        .insert_resource(SessionStore::new(store.clone()));
    app.finish();
    app.update();
    app
}

fn first_agent(app: &mut App) -> Option<Entity> {
    let mut agents = app.world_mut().query_filtered::<Entity, With<Agent>>();
    agents.iter(app.world()).next()
}

/// The titles of `agent`'s entries, in order.
fn titles(app: &App, agent: Entity) -> Vec<String> {
    let world = app.world();
    let entries = world.get::<TranscriptEntries>(agent);
    entries
        .into_iter()
        .flat_map(TranscriptEntries::iter)
        .filter_map(|entry| world.get::<TranscriptEntry>(entry))
        .map(|entry| entry.title.clone())
        .collect()
}

#[test]
fn an_entry_comes_back_with_the_session_and_an_unsaved_one_does_not() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let agent = first_agent(&mut first);
    assert!(agent.is_some());
    let Some(agent) = agent else {
        return;
    };
    let kept = TranscriptEntry::new("btw", None, "Why tabs?", "The style says so.");
    let mut passing = TranscriptEntry::new("btw", None, "Passing", "");
    passing.saved = false;
    first.world_mut().spawn((kept, EntryOf(agent)));
    first.world_mut().spawn((passing, EntryOf(agent)));
    first.update();
    assert_eq!(titles(&first, agent), ["Why tabs?", "Passing"]);

    let mut second = app(&store);
    let agent = first_agent(&mut second);
    assert!(agent.is_some(), "the session was not kept");
    let Some(agent) = agent else {
        return;
    };
    assert_eq!(titles(&second, agent), ["Why tabs?"]);
    // Restoring it is no change to keep: the entry is there once.
    second.update();
    second.update();
    assert_eq!(titles(&second, agent), ["Why tabs?"]);

    // A removed entry is gone from the session too.
    let entries: Vec<Entity> = second
        .world()
        .get::<TranscriptEntries>(agent)
        .into_iter()
        .flat_map(TranscriptEntries::iter)
        .collect();
    for entry in entries {
        second.world_mut().despawn(entry);
    }
    second.update();
    let mut third = app(&store);
    let agent = first_agent(&mut third);
    let titles = agent.map(|agent| titles(&third, agent)).unwrap_or_default();
    assert!(titles.is_empty(), "{titles:?}");
}
