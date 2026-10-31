use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_reflect::prelude::*;
use rig_core::completion::Message;
use rig_memory::{MemoryStore, Store};

use super::{AgentForked, ForkAgent, ForkedFrom};
use crate::AgentPlugin;
use crate::agent::{Agent, AgentId, CacheKey, Conversation, Notice, SystemPrompt, ToolAccess};
use crate::session_log::tests::push_message;
use crate::session_log::{Never, ReflectSaved, Reset, SessionLogPlugin, SessionStore};

/// The forks made, in order.
#[derive(Resource, Default)]
struct Forks(Vec<AgentForked>);

/// An app on `store`, keeping the forks made, after its first frame
/// restored the session. Every agent requires a [`Duty`].
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .register_required_components::<Agent, Duty>()
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

/// Without a type registry to copy the saved components with, a fork is a
/// notice, and no log is copied for it.
#[test]
fn a_fork_without_a_type_registry_is_a_notice_and_copies_no_log() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let source = app.world_mut().spawn(Agent).id();
    for message in [Message::user("one?"), Message::assistant("one.")] {
        push_message(app.world_mut(), source, message);
    }
    app.update();
    let logs = Store::logs(&store).map(|logs| logs.len()).ok();
    app.world_mut().remove_resource::<AppTypeRegistry>();
    app.world_mut().trigger(ForkAgent {
        agent: source,
        through: None,
    });
    app.update();
    assert!(app.world().resource::<Forks>().0.is_empty());
    let notices = app.world().resource::<Messages<Notice>>();
    let refused = (notices.get_cursor().read(notices))
        .any(|notice| notice.text.starts_with("Could not fork the agent"));
    assert!(refused);
    assert_eq!(Store::logs(&store).map(|logs| logs.len()).ok(), logs);
}

/// A plugin's setting, which a fork copies.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Component, Saved, Default)]
struct Setting(u32);

/// A plugin's meter, which a fork starts at zero.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Component, Saved(Reset), Default)]
struct Meter(u32);

/// A plugin's duty, which a fork never has.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Component, Saved(Never), Default)]
struct Duty(u32);

/// A fork copies a saved component, starts a `Reset` one at its default
/// and does not get a `Never` one, even one every agent requires, and so
/// it is restored, where it has the `Never` one only as a new agent would.
#[test]
fn a_fork_copies_resets_or_leaves_out_each_saved_component_as_its_type_says() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let source = app.world_mut().spawn(Agent).id();
    push_message(app.world_mut(), source, Message::user("one?"));
    push_message(app.world_mut(), source, Message::assistant("one."));
    app.world_mut()
        .entity_mut(source)
        .insert((Setting(1), Meter(2), Duty(3)));
    app.update();
    app.world_mut().trigger(ForkAgent {
        agent: source,
        through: None,
    });
    app.update();
    let fork = app.world().resource::<Forks>().0.first().map(|f| f.fork);
    assert!(fork.is_some(), "a fork is made");
    let Some(fork) = fork else { return };
    let read = |world: &World, entity: Entity| {
        (
            world.get::<Setting>(entity).cloned(),
            world.get::<Meter>(entity).cloned(),
            world.get::<Duty>(entity).cloned(),
            world.get::<ForkedFrom>(entity).cloned(),
        )
    };
    let source_id = app.world().get::<AgentId>(source).cloned();
    let fork_id = app.world().get::<AgentId>(fork).cloned();
    let (Some(source_id), Some(fork_id)) = (source_id, fork_id) else {
        return;
    };
    let from = Some(ForkedFrom(source_id.0.clone()));
    let forked = (Some(Setting(1)), Some(Meter(0)), None, from);
    assert_eq!(read(app.world(), fork), forked);
    assert_eq!(
        read(app.world(), source),
        (Some(Setting(1)), Some(Meter(2)), Some(Duty(3)), None)
    );
    drop(app);

    let mut again = self::app(&store);
    let fork = agent(&mut again, &fork_id);
    assert!(fork.is_some(), "the fork is restored");
    let Some(fork) = fork else { return };
    // Restored, it has the duty every agent requires, not the source's.
    let restored = (forked.0, forked.1, Some(Duty(0)), forked.3);
    assert_eq!(read(again.world(), fork), restored);
}
