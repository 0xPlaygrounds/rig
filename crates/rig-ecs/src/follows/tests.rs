use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::ReflectDefault;
use bevy_reflect::{Reflect, TypePath};
use rig_core::completion::Message;
use rig_core::conversation::{Aside, Author};
use rig_memory::{MemoryStore, SHELVED};
use serde_json::json;

use super::{FOLLOWS, ReflectFollows};
use crate::AgentPlugin;
use crate::agent::{Agent, Conversation};
use crate::restore::Shelved;
use crate::session_log::tests::{edit, push_message};
use crate::session_log::{SessionLogPlugin, SessionStore};

/// A plugin's mode, kept with the line it was set on.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Component, Follows, Default)]
struct Mode(String);

fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .insert_resource(SessionStore::new(store.clone()));
    app.finish();
    app.update();
    app
}

fn agent(app: &mut App) -> Entity {
    let mut agents = app.world_mut().query_filtered::<Entity, With<Agent>>();
    let agent = agents.iter(app.world()).next();
    assert!(agent.is_some(), "an agent");
    agent.unwrap_or(Entity::PLACEHOLDER)
}

fn mode(app: &App, agent: Entity) -> Option<String> {
    app.world().get::<Mode>(agent).map(|mode| mode.0.clone())
}

fn set_mode(app: &mut App, agent: Entity, mode: &str) {
    app.world_mut()
        .entity_mut(agent)
        .insert(Mode(mode.to_owned()));
    app.update();
}

fn say(app: &mut App, agent: Entity, text: &str) {
    push_message(app.world_mut(), agent, Message::user(text));
    app.update();
}

/// An edit that takes out the entries values followed keeps one value per
/// key, the newest, after what stands for them now.
#[test]
fn values_of_entries_a_splice_took_out_collapse_into_one() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let agent = agent(&mut app);
    say(&mut app, agent, "one");
    set_mode(&mut app, agent, "plan");
    say(&mut app, agent, "two");
    set_mode(&mut app, agent, "build");
    edit(app.world_mut(), agent, |conversation| {
        let summary = rig_core::conversation::Entry::new(Message::user("summary"));
        conversation.splice(..2, [summary], Author::plugin("compaction"));
    });
    app.update();
    let conversation = app.world().get::<Conversation>(agent);
    let follows: Vec<_> = conversation
        .map(|c| {
            let summary = c.id(0);
            c.asides()
                .filter(|(_, aside)| aside.kind == FOLLOWS)
                .map(|(_, aside)| (aside.after == summary, aside.value.get("value").cloned()))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(follows, [(true, Some("build".into()))]);
    assert_eq!(mode(&app, agent).as_deref(), Some("build"));
}

/// The value of `key` the agent's active line has.
fn line_value(app: &App, agent: Entity, key: &str) -> Option<serde_json::Value> {
    let saved = super::followed(app.world().get::<Conversation>(agent)?, key)?;
    Some(json!({"v": saved.v, "value": saved.value}))
}

/// Puts `value` on the agent's line as its followed mode, after its last
/// entry, as another build of the plugin would have kept it.
fn keep_raw(app: &mut App, agent: Entity, value: serde_json::Value) {
    let key = <Mode as TypePath>::type_path();
    edit(app.world_mut(), agent, move |conversation| {
        let last = conversation.entries().next_back().map(|entry| entry.id);
        let mut value = value;
        if let Some(value) = value.as_object_mut() {
            value.insert("key".to_owned(), key.into());
        }
        conversation.add_aside(Aside::new(FOLLOWS, Author::rig(), value).after(last));
    });
    app.update();
}

/// A followed value that no step reads is set aside, not frozen: the next
/// change moves it to its side key and is kept, so nothing is lost and the
/// plugin's state follows its line again. A newer rig's value is held.
#[test]
fn an_unreadable_value_is_set_aside_on_the_next_change() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let agent = agent(&mut app);
    let key = <Mode as TypePath>::type_path();
    say(&mut app, agent, "one");
    set_mode(&mut app, agent, "plan");
    say(&mut app, agent, "two");
    keep_raw(&mut app, agent, json!({"v": 1, "value": 42}));
    assert_eq!(mode(&app, agent).as_deref(), Some("plan"), "not read");
    let set_aside = app.world().resource::<Shelved>();
    assert!(set_aside.0.iter().any(|value| value.key == key));
    set_mode(&mut app, agent, "build");
    assert_eq!(
        line_value(&app, agent, key),
        Some(json!({"v": 1, "value": "build"})),
        "the change is kept"
    );
    let side = format!("{SHELVED}{key}/v1");
    assert_eq!(
        line_value(&app, agent, &side),
        Some(json!({"v": 1, "value": 42})),
        "the old value is kept aside"
    );
    say(&mut app, agent, "three");
    keep_raw(&mut app, agent, json!({"v": 9, "value": "later"}));
    set_mode(&mut app, agent, "ask");
    assert_eq!(
        line_value(&app, agent, key),
        Some(json!({"v": 9, "value": "later"})),
        "a newer value is held"
    );
    drop(app);
    let mut again = self::app(&store);
    let agent = self::agent(&mut again);
    assert_eq!(mode(&again, agent), None, "held, not read");
    assert_eq!(
        line_value(&again, agent, key),
        Some(json!({"v": 9, "value": "later"}))
    );
}
