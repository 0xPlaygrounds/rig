use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::completion::Message;
use rig_memory::MemoryStore;

use super::{AgentForked, ForkAgent};
use crate::AgentPlugin;
use crate::agent::{Agent, AgentId, CacheKey, Conversation, SystemPrompt, ToolAccess};
use crate::session_log::tests::push_message;
use crate::session_log::{SessionLogPlugin, SessionStore};

/// The forks made, in order.
#[derive(Resource, Default)]
struct Forks(Vec<AgentForked>);

/// An app on `store`, keeping the forks made, after its first frame
/// restored the session.
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .insert_resource(SessionStore::new(store.clone()))
        .init_resource::<Forks>()
        .add_observer(|forked: On<AgentForked>, mut forks: ResMut<Forks>| {
            forks.0.push(forked.clone());
        });
    app.finish();
    app.update();
    app
}

/// The agent whose id is `id`.
fn agent(app: &mut App, id: &AgentId) -> Option<Entity> {
    let world = app.world_mut();
    let mut agents = world.query::<(Entity, &AgentId)>();
    id.find_in(agents.iter(world))
}

/// A fork is a new agent with the source's conversation, saved components
/// and prompt cache, and comes back as it is after a restart.
#[test]
fn a_fork_keeps_the_conversation_the_saved_components_and_the_cache_and_is_restored() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let source = app.world_mut().spawn(Agent).id();
    for message in [Message::user("one?"), Message::assistant("one.")] {
        push_message(app.world_mut(), source, message);
    }
    app.world_mut().entity_mut(source).insert((
        SystemPrompt("be brief".to_owned()),
        ToolAccess::Only(vec!["read".to_owned()]),
    ));
    app.update();
    app.world_mut().trigger(ForkAgent {
        agent: source,
        through: None,
    });
    app.update();
    let forks = &app.world().resource::<Forks>().0;
    let fork = forks.first().map(|forked| (forked.from, forked.fork));
    assert_eq!(fork.map(|(from, _)| from), Some(source));
    let Some((_, fork)) = fork else { return };
    let world = app.world();
    let (Some(source_id), Some(fork_id)) = (
        world.get::<AgentId>(source).cloned(),
        world.get::<AgentId>(fork).cloned(),
    ) else {
        return;
    };
    assert_ne!(source_id, fork_id);
    let read = |entity: Entity| {
        let world = app.world();
        let messages = world
            .get::<Conversation>(entity)
            .map(|c| c.messages().to_vec());
        let prompt = world.get::<SystemPrompt>(entity).map(|p| p.0.clone());
        let tools = world.get::<ToolAccess>(entity).map(|t| t.allows("shell"));
        let key = world.get::<CacheKey>(entity).cloned();
        (messages, prompt, tools, key)
    };
    let (messages, prompt, tools, key) = read(fork);
    assert_eq!(messages, read(source).0);
    assert_eq!(prompt.as_deref(), Some("be brief"));
    assert_eq!(tools, Some(false));
    assert_eq!(key, Some(CacheKey(source_id.0.clone())));
    drop(app);

    let mut again = self::app(&store);
    let fork = agent(&mut again, &fork_id);
    let source = agent(&mut again, &source_id);
    assert!(fork.is_some() && source.is_some(), "both are restored");
    let (Some(fork), Some(source)) = (fork, source) else {
        return;
    };
    let world = again.world();
    let messages = |entity| {
        world
            .get::<Conversation>(entity)
            .map(|c| c.messages().to_vec())
    };
    assert_eq!(messages(fork), messages(source));
    assert_eq!(
        world.get::<CacheKey>(fork).cloned(),
        Some(CacheKey(source_id.0))
    );
    let prompt = world.get::<SystemPrompt>(fork).map(|p| p.0.clone());
    assert_eq!(prompt.as_deref(), Some("be brief"));
}
