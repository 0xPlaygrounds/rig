use bevy_app::prelude::*;
use serde_json::Value;

use super::*;
use crate::tools::{AppToolsExt, ToolCalled, ToolOptions};

/// An app with the tools `read` and `task`.
fn app() -> App {
    let mut app = App::new();
    for tool in ["read", "task"] {
        let options = ToolOptions::default();
        app.add_open_tool(tool, "A tool.", options, |_: On<ToolCalled<Value>>| {});
    }
    app
}

fn parent(world: &World, agent: Entity) -> Option<Entity> {
    world.get::<SpawnedBy>(agent).map(SpawnedBy::parent)
}

#[test]
fn a_child_takes_what_its_spec_leaves_out_from_its_origin_at_any_depth() {
    let mut app = app();
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
    // Each agent below keeps its origin's tools.
    let mut deepest = child;
    for _ in 0..3 {
        let below = spawn(world, Origin::ChildOf(deepest), AgentSpec::default());
        let below = below.unwrap_or(Entity::PLACEHOLDER);
        assert_eq!(parent(world, below), Some(deepest));
        deepest = below;
    }
    let keeps = |world: &World, agent| {
        let access = world.get::<ToolAccess>(agent);
        access.is_some_and(|access| access.keeps("read") && access.keeps("task"))
    };
    assert!(keeps(world, deepest));
    // Beside an agent is under its parent.
    let like = spawn(world, Origin::Like(deepest), AgentSpec::default());
    let like = like.unwrap_or(Entity::PLACEHOLDER);
    assert_eq!(parent(world, like), parent(world, deepest));
    assert!(keeps(world, like));
}
