use bevy_app::prelude::*;
use bevy_ecs::relationship::Relationship;
use serde_json::Value;

use super::*;
use crate::tools::{AppToolsExt, ToolCalled, ToolOptions};

/// An app with the tools `read` and `task`, which starts agents, where
/// agents nest at most `max_depth` deep.
fn app(max_depth: usize) -> App {
    let mut app = App::new();
    app.add_message::<Notice>()
        .add_observer(within_depth)
        .insert_resource(SpawnPolicy {
            max_depth,
            root_depth: 0,
        });
    for tool in ["read", "task"] {
        let options = ToolOptions::default();
        app.add_open_tool(tool, "A tool.", options, |_: On<ToolCalled<Value>>| {});
    }
    app.starts_agents("task");
    app
}

fn parent(world: &World, agent: Entity) -> Option<Entity> {
    world.get::<SpawnedBy>(agent).map(SpawnedBy::parent)
}

#[test]
fn a_child_takes_what_its_spec_leaves_out_from_its_origin_and_never_passes_the_limit() {
    let mut app = app(2);
    let world = app.world_mut();
    let root = spawn_root(world)
        .insert((
            SystemPrompt("root".into()),
            EffectParent(EffectId::from_raw(7)),
        ))
        .id();
    let spec = AgentSpec {
        tools: Some(vec!["read".into(), "task".into()]),
        ..AgentSpec::default()
    };
    let child = spawn(world, Origin::ChildOf(root), spec).unwrap_or(Entity::PLACEHOLDER);
    let prompt = world
        .get::<SystemPrompt>(child)
        .map(|prompt| prompt.0.clone());
    assert_eq!(prompt.as_deref(), Some("root"));
    let under = |world: &World, agent| world.get::<EffectParent>(agent).map(|under| under.0);
    assert_eq!(under(world, child), under(world, root));
    // At the limit, it has no tool that starts agents, and may not be
    // given one.
    let spec = AgentSpec {
        tools: Some(vec!["task".into()]),
        ..AgentSpec::default()
    };
    let named = spawn(world, Origin::ChildOf(child), spec);
    let refused = SpawnRefused::StartsAgents {
        tool: "task".into(),
        depth: 2,
    };
    assert_eq!(named, Err(refused));
    let deepest = spawn(world, Origin::ChildOf(child), AgentSpec::default());
    let deepest = deepest.unwrap_or(Entity::PLACEHOLDER);
    let tools = |world: &World, agent| {
        let access = world.get::<ToolAccess>(agent);
        access.map(|access| (access.keeps("read"), access.keeps("task")))
    };
    assert_eq!(tools(world, child), Some((true, true)));
    assert_eq!(tools(world, deepest), Some((true, false)));
    // Beside the deepest agent is still at the limit; below it is past.
    let like = spawn(world, Origin::Like(deepest), AgentSpec::default());
    let like = like.unwrap_or(Entity::PLACEHOLDER);
    assert_eq!(parent(world, like), Some(child));
    assert_eq!(tools(world, like), Some((true, false)));
    let below = spawn(world, Origin::ChildOf(deepest), AgentSpec::default());
    assert_eq!(below, Err(SpawnRefused::TooDeep { depth: 3, max: 2 }));
}

#[test]
fn a_link_made_round_the_spawner_never_puts_an_agent_past_the_limit() {
    let mut app = app(1);
    let world = app.world_mut();
    let root = spawn_root(world).id();
    let child = spawn(world, Origin::ChildOf(root), AgentSpec::default());
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    let other = spawn_root(world).id();
    world
        .entity_mut(other)
        .insert(<SpawnedBy as Relationship>::from(child));
    world.flush();
    assert_eq!(parent(world, other), None);
    // Its parent doesn't list it either, so it neither stops nor goes with it.
    let lists = |world: &World, of: Entity, agent: Entity| {
        world
            .get::<Spawned>(of)
            .is_some_and(|spawned| spawned.iter().any(|listed| listed == agent))
    };
    assert!(!lists(world, child, other));
    // A root with an agent below it is too deep to go below another.
    let top = spawn_root(world).id();
    world
        .entity_mut(root)
        .insert(<SpawnedBy as Relationship>::from(top));
    world.flush();
    assert!(!lists(world, top, root));
    assert_eq!(
        (parent(world, root), parent(world, child)),
        (None, Some(root))
    );
    let notices = world.resource::<Messages<Notice>>();
    assert_eq!(notices.iter_current_update_messages().count(), 2);
}

/// A root at the depth limit, as in a `rig -p` a deep agent started, keeps
/// a tool registered after it, such as an MCP server's, and still starts
/// no agent.
#[test]
fn a_root_at_the_limit_keeps_later_tools_and_starts_no_agent() {
    let mut app = app(0);
    let root = spawn_root(app.world_mut()).id();
    app.add_open_tool(
        "later",
        "A tool.",
        ToolOptions::default(),
        |_: On<ToolCalled<Value>>| {},
    );
    let world = app.world_mut();
    let keeps = world
        .get::<ToolAccess>(root)
        .is_some_and(|access| access.allows("later"));
    assert!(keeps);
    let child = spawn(world, Origin::ChildOf(root), AgentSpec::default());
    assert!(
        matches!(child, Err(SpawnRefused::TooDeep { .. })),
        "{child:?}"
    );
}
