use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::completion::Message;
use rig_core::message::UserContent;

use crate::AgentPlugin;
use crate::agent::{Agent, AgentId, Condensed, Conversation, Halt, Interrupt, STOPPED, ToolAccess};
use crate::inbox::{Deliver, DeliveryMode};
use crate::journal::{JournalPlugin, SessionLog};
use crate::restore::Restored;
use crate::store::{MemoryStore, SessionStore};

/// An app on `store`, after its first frame restored the session. Like
/// the harness, it only warns about a command on a despawned entity.
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(SessionStore::new(store.clone()))
        .add_plugins((AgentPlugin, JournalPlugin));
    app.update();
    app
}

fn first_agent(app: &mut App) -> Option<Entity> {
    let mut agents = app.world_mut().query_filtered::<Entity, With<Agent>>();
    agents.iter(app.world()).next()
}

fn say(app: &mut App, agent: Entity, message: Message) {
    let log = app.world().resource::<SessionLog>().clone();
    let mut entity = app.world_mut().entity_mut(agent);
    let id = entity.get::<AgentId>().cloned();
    if let (Some(id), Some(mut conversation)) = (id, entity.get_mut::<Conversation>()) {
        log.commit(&id, &mut conversation, message, None);
    }
}

#[test]
fn a_restored_agent_starts_from_the_messages_its_summary_kept() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let agent = first_agent(&mut first);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    for n in 0..4 {
        say(&mut first, agent, Message::user(format!("question {n}")));
        say(&mut first, agent, Message::assistant(format!("answer {n}")));
    }
    let summary = "The user asked three questions.".to_owned();
    first.world_mut().entity_mut(agent).insert(Condensed {
        upto: 6,
        summary: summary.clone(),
    });
    first.update();
    let mut second = app(&store);
    let agent = first_agent(&mut second);
    assert!(agent.is_some(), "the agent was restored");
    let Some(agent) = agent else { return };
    let world = second.world();
    let kept = world
        .get::<Conversation>(agent)
        .map(|conversation| conversation.messages().to_vec());
    let last = [Message::user("question 3"), Message::assistant("answer 3")];
    assert_eq!(kept.as_deref(), Some(last.as_slice()));
    let condensed = world.get::<Condensed>(agent);
    assert_eq!(condensed, Some(&Condensed { upto: 0, summary }));
    // Requests send the summary at the head of the first kept message.
    let sent = condensed.map(|condensed| condensed.request(&last));
    let first_sent = sent.as_ref().and_then(|sent| sent.first());
    assert!(
        matches!(first_sent, Some(Message::User { content }) if content.len() == 2),
        "{sent:?}"
    );
}

#[test]
fn a_restored_agent_keeps_the_tools_a_plugin_narrowed_it_to() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let agent = first_agent(&mut first);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    say(&mut first, agent, Message::user("hello"));
    first
        .world_mut()
        .entity_mut(agent)
        .insert(ToolAccess::Only(vec!["read".to_owned()]));
    first.update();
    let mut second = app(&store);
    let tools = first_agent(&mut second)
        .and_then(|agent| second.world().get::<ToolAccess>(agent))
        .and_then(|access| match access {
            ToolAccess::All => None,
            ToolAccess::Only(names) => Some(names.clone()),
        });
    assert_eq!(tools, Some(vec!["read".to_owned()]));
}

/// The messages of the one agent of the session in `store`, restored and
/// kept idle, after `act` ran on it.
fn session(store: &MemoryStore, act: impl FnOnce(&mut World, Entity)) -> Vec<Message> {
    let mut app = App::new();
    app.insert_resource(SessionStore::new(store.clone()))
        .add_plugins((AgentPlugin, JournalPlugin))
        .add_observer(|mut restored: On<Restored>| restored.event_mut().resume = false);
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
    let live = session(&store, |world, agent| {
        let log = world.resource::<SessionLog>().clone();
        let id = world.get::<AgentId>(agent).cloned().unwrap_or_default();
        for (at, text) in texts.into_iter().enumerate() {
            world.trigger(Deliver::user(agent, text, DeliveryMode::Steer, Vec::new()));
            // The observers' commands start the turn, and end it.
            world.flush();
            world.trigger(Interrupt { entity: agent });
            world.flush();
            // The second is asked for again, and that turn fails and keeps it.
            if let Some(mut conversation) = world.get_mut::<Conversation>(agent)
                && at == 1
            {
                assert!(conversation.resume());
                log.halt(&id, &mut conversation, Halt::Kept);
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
