use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::completion::{AssistantContent, CompletionResponse, Message, Usage};
use rig_core::message::{Origin, UserContent};

use crate::AgentPlugin;
use crate::agent::{
    Agent, AgentId, CallOf, Conversation, Halt, Interrupt, STOPPED, ToolAccess, TurnOf,
};
use crate::calls::Done;
use crate::compaction::{CompactReason, Summarizing, Summary};
use crate::inbox::{Deliver, DeliveryMode};
use crate::journal::{JournalPlugin, SessionLog};
use crate::restore::Restored;
use crate::store::{MemoryStore, SessionStore};
use crate::usage::Spending;

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
fn a_restored_agent_keeps_the_context_its_compaction_left() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let agent = first_agent(&mut first);
    assert!(agent.is_some());
    let Some(agent) = agent else { return };
    for n in 0..4 {
        say(&mut first, agent, Message::user(format!("question {n}")));
        say(&mut first, agent, Message::assistant(format!("answer {n}")));
    }
    // What the last reply reported, near the model's window.
    if let Some(mut spent) = first.world_mut().get_mut::<Spending>(agent) {
        spent.context = Some(150_000);
    }
    let turn = first.world_mut().spawn(TurnOf(agent)).id();
    let summary = CompletionResponse::new(
        vec![AssistantContent::text("The user asked four questions.")],
        Usage::new().input_tokens(100).output_tokens(10),
        Origin::new("test", "test", "test/model"),
        serde_json::Value::Null,
    );
    first.world_mut().spawn((
        CallOf(turn),
        Summarizing {
            reason: CompactReason::Asked {
                focus: String::new(),
            },
            upto: 6,
            messages: 6,
            tokens: 0,
            kept: 2,
            kept_tokens: 0,
            tracked: Vec::new(),
        },
        Done(Summary(Ok(summary))),
    ));
    first.update();
    let context = |app: &mut App| {
        first_agent(app)
            .and_then(|agent| app.world().get::<Spending>(agent))
            .and_then(|spent| spent.context)
    };
    let left = context(&mut first);
    assert!(left.is_some_and(|left| left < 150_000), "{left:?}");
    let mut second = app(&store);
    assert_eq!(context(&mut second), left);
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
