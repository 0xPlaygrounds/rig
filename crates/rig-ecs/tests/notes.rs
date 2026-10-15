//! A note is saved and restored with its conversation, an unsaved one not.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::conversation::{Aside, Author, Note};
use rig_ecs::agent::{Agent, Conversation};
use rig_ecs::session_log::{SessionLogPlugin, SessionStore};
use rig_memory::MemoryStore;

/// An app on `store`, after its first frame restored the session.
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((rig_ecs::AgentPlugin, SessionLogPlugin))
        .insert_resource(SessionStore::new(store.clone()));
    app.finish();
    app.update();
    app
}

/// The first agent's conversation, mutably.
fn conversation(app: &mut App) -> Option<Mut<'_, Conversation>> {
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.get_mut::<Conversation>(agent)
}

/// The titles of the first agent's notes, in order.
fn titles(app: &mut App) -> Vec<String> {
    conversation(app)
        .map(|conversation| {
            conversation
                .asides()
                .filter_map(|(_, aside)| Note::of(aside))
                .map(|note| note.title)
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn a_note_comes_back_with_the_session_and_an_unsaved_one_does_not() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let btw = Author::plugin("btw");
    let added = conversation(&mut first).map(|mut conversation| {
        let kept = Note::new("Why tabs?", "The style says so.");
        conversation.add_aside(kept.aside(btw.clone(), None));
        let passing = Note::new("Passing", "").aside(btw.clone(), None);
        conversation.add_aside(passing.unsaved());
        // Another kind of aside is no note.
        let other = Aside::new("other", btw.clone(), serde_json::json!({"title": "x"}));
        conversation.add_aside(other);
    });
    assert!(added.is_some(), "the app has an agent");
    first.update();
    assert_eq!(titles(&mut first), ["Why tabs?", "Passing"]);

    let mut second = app(&store);
    assert_eq!(titles(&mut second), ["Why tabs?"]);
    let removed = conversation(&mut second).map(|mut conversation| {
        let ids: Vec<_> = conversation.asides().map(|(id, _)| id).collect();
        for id in ids {
            conversation.remove_aside(id);
        }
    });
    assert!(removed.is_some());
    second.update();
    let mut third = app(&store);
    assert!(titles(&mut third).is_empty());
}
