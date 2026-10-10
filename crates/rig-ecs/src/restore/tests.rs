use std::collections::HashMap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{Reflect, TypePath};
use rig_cassette::journal::{JournalStore, MemoryStore};
use rig_core::completion::{Message, Reasoning, Usage};
use rig_core::message::UserContent;
use serde_json::Value;

use crate::AgentPlugin;
use crate::agent::{
    Agent, AgentId, Condensed, Conversation, Halt, Interrupt, LastUsage, Notice, STOPPED,
    SystemPrompt, ToolAccess,
};
use crate::inbox::{Deliver, DeliveryMode};
use crate::journal::{
    Commit, Committed, JournalPlugin, ReflectSaved, SessionLog, SessionStore, commit_message,
};
use crate::model::{Effort, ModelChoice};
use crate::restore::Restored;

/// An app on `store`, after its first frame restored the session. Like
/// the harness, it only warns about a command on a despawned entity.
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(SessionStore::new(store.clone()))
        .add_plugins((AgentPlugin, JournalPlugin));
    app.finish();
    app.update();
    app
}

fn first_agent(app: &mut App) -> Option<Entity> {
    let mut agents = app.world_mut().query_filtered::<Entity, With<Agent>>();
    agents.iter(app.world()).next()
}

fn say(app: &mut App, agent: Entity, message: Message) {
    commit_message(app.world_mut(), agent, message);
}

/// A plugin's per-agent counter, as the plugin guide's example keeps one.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct ToolCounts(HashMap<String, u32>);

/// Every saved component of the first agent, by type path, as reflection
/// writes it.
fn saved(app: &mut App) -> HashMap<String, Value> {
    let agent = first_agent(app);
    let world = app.world();
    let registry = world.resource::<AppTypeRegistry>().read();
    let entity = agent.and_then(|agent| world.get_entity(agent).ok());
    registry
        .iter_with_data::<ReflectSaved>()
        .filter_map(|(registration, _)| {
            let component = registration.data::<ReflectComponent>()?.reflect(entity?)?;
            let value = TypedReflectSerializer::new(component.as_partial_reflect(), &registry);
            let path = registration.type_info().type_path().to_owned();
            Some((path, serde_json::to_value(value).ok()?))
        })
        .collect()
}

#[test]
fn a_restored_agent_has_its_saved_components_and_the_messages_its_summary_kept() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let agent = first_agent(&mut first);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    let usage = Usage::new().input_tokens(10).output_tokens(5);
    let counts = |reads| ToolCounts(HashMap::from([("read".to_owned(), reads)]));
    first.world_mut().entity_mut(agent).insert((
        ModelChoice("ollama/deepseek-v4-flash".to_owned()),
        Effort(Some(Reasoning::Off)),
        SystemPrompt("You review code.".to_owned()),
        ToolAccess::Only(vec!["read".to_owned()]),
        LastUsage(Some(usage)),
        counts(1),
    ));
    for n in 0..4 {
        say(&mut first, agent, Message::user(format!("question {n}")));
        say(&mut first, agent, Message::assistant(format!("answer {n}")));
        // The newest value wins.
        first.world_mut().entity_mut(agent).insert(counts(n + 1));
        first.update();
    }
    let summary = "The user asked three questions.".to_owned();
    let condensed = Condensed { upto: 6, summary };
    first
        .world_mut()
        .entity_mut(agent)
        .insert(condensed.clone());
    first.update();
    let before = saved(&mut first);
    assert_eq!(before.len(), 6, "{before:?}");
    let newest = serde_json::json!({ "read": 4 });
    assert_eq!(before.get(ToolCounts::type_path()), Some(&newest));
    let id = first.world().get::<AgentId>(agent).cloned();
    drop(first);
    let gone = r#"{"seq":99,"t":0,"type":"component","component":"a_plugin::Gone","value":1}"#;
    let id = id.unwrap_or_default().0;
    assert!(store.append(&id, format!("{gone}\n").as_bytes()).is_ok());

    let mut second = app(&store);
    // What was saved before the summary still comes back.
    assert_eq!(saved(&mut second), before);
    let agent = first_agent(&mut second);
    let world = second.world();
    let notices = world.resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    let mut texts = cursor.read(notices).map(|notice| notice.text.as_str());
    assert!(texts.any(|text| text.contains("a_plugin::Gone")));
    let kept = agent.and_then(|agent| world.get::<Conversation>(agent));
    let kept = kept.map(|conversation| conversation.messages().to_vec());
    let last = [Message::user("question 3"), Message::assistant("answer 3")];
    assert_eq!(kept.as_deref(), Some(last.as_slice()));
    let restored = agent.and_then(|agent| world.get::<Condensed>(agent));
    let summary = Condensed {
        upto: 0,
        ..condensed
    };
    assert_eq!(restored, Some(&summary));
    // Requests send the summary at the head of the first kept message.
    let sent = restored.map(|condensed| condensed.request(&last));
    let first_sent = sent.as_ref().and_then(|sent| sent.first());
    assert!(
        matches!(first_sent, Some(Message::User { content }) if content.len() == 2),
        "{sent:?}"
    );
}

#[test]
fn every_change_the_log_records_is_committed() {
    let mut app = app(&MemoryStore::default());
    let agent = first_agent(&mut app);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    // No model answers it, so it is taken out; a note is left halted.
    let hello = Deliver::user(agent, "hello", DeliveryMode::Steer, Vec::new());
    app.world_mut().trigger(hello);
    app.update();
    let note = Deliver::user(agent, "a note", DeliveryMode::Note, Vec::new());
    app.world_mut().trigger(note);
    let committed = app.world().resource::<Messages<Committed>>();
    let mut cursor = committed.get_cursor();
    let changes = cursor.read(committed).map(|change| match change {
        Committed::Message { .. } => "message",
        Committed::Retract { .. } => "retract",
        Committed::Halt { .. } => "halt",
    });
    assert_eq!(
        changes.collect::<Vec<_>>(),
        ["message", "retract", "message", "halt"]
    );
}

/// The messages of the one agent of the session in `store`, restored and
/// kept idle, after `act` ran on it.
fn session(store: &MemoryStore, act: impl FnOnce(&mut World, Entity)) -> Vec<Message> {
    let mut app = App::new();
    app.insert_resource(SessionStore::new(store.clone()))
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_observer(|mut restored: On<Restored>| restored.event_mut().resume = false);
    app.finish();
    app.update();
    let world = app.world_mut();
    let Ok(agent) = world.query_filtered::<Entity, With<Agent>>().single(world) else {
        return Vec::new();
    };
    act(world, agent);
    world.resource::<SessionLog>().flush();
    let conversation = world.get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |conversation| conversation.messages().to_vec())
}

#[test]
fn a_message_after_a_stopped_request_says_it_was_stopped_unless_it_was_retried() {
    let store = MemoryStore::default();
    let texts = [
        "run sleep 120 && echo finished",
        "what is 2 + 2?",
        "and 3 + 3?",
    ];
    // The second is asked for again, and that turn fails and keeps it.
    let keep = |In(agent): In<Entity>, mut agents: Query<&mut Conversation>, mut commit: Commit| {
        if let Ok(mut conversation) = agents.get_mut(agent) {
            assert!(conversation.resume());
            commit.halt(agent, &mut conversation, Halt::Kept);
        }
    };
    let live = session(&store, |world, agent| {
        for (at, text) in texts.into_iter().enumerate() {
            world.trigger(Deliver::user(agent, text, DeliveryMode::Steer, Vec::new()));
            // The observers' commands start the turn, and end it.
            world.flush();
            world.trigger(Interrupt { entity: agent });
            world.flush();
            if at == 1 {
                assert!(world.run_system_cached_with(keep, agent).is_ok());
            }
        }
    });
    let [first, second, third] = texts;
    let content = [first, STOPPED, second, third].map(UserContent::text);
    assert_eq!(
        live,
        [Message::User {
            content: content.into()
        }]
    );
    assert_eq!(session(&store, |_, _| {}), live);
}
