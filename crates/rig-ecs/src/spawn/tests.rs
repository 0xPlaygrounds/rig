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

/// Spawns two agents with the id `id` in one system, and keeps whether
/// each was made.
fn spawn_twice(id: &'static str) -> impl FnMut(Spawner, ResMut<Made>) {
    move |mut spawner: Spawner, mut made: ResMut<Made>| {
        for _ in 0..2 {
            let spec = AgentSpec {
                id: Some(AgentId(id.into())),
                ..AgentSpec::default()
            };
            made.0.push(spawner.spawn(Origin::Root, spec).map(|_| ()));
        }
    }
}

/// What each spawn of [`spawn_twice`] gave.
#[derive(Resource, Default)]
struct Made(Vec<Result<(), SpawnRefused>>);

#[test]
fn an_id_is_taken_as_soon_as_it_is_asked_for() {
    let mut app = app();
    app.add_plugins(crate::AgentPlugin)
        .init_resource::<Made>()
        .add_systems(Update, spawn_twice("helper"));
    app.update();
    let made = &app.world().resource::<Made>().0;
    let taken = SpawnRefused::IdTaken(AgentId("helper".into()));
    assert_eq!(made.as_slice(), [Ok(()), Err(taken)]);
    let mut ids = app.world_mut().query::<&AgentId>();
    let helpers = ids.iter(app.world()).filter(|id| id.0 == "helper");
    assert_eq!(helpers.count(), 1);
}

#[test]
fn an_id_that_names_no_agent_log_is_refused() {
    let mut app = app();
    let world = app.world_mut();
    // The session's own log on a disk that ignores case, and a name too
    // long for a file name with its extension.
    let long = "x".repeat(300);
    for id in [
        "session",
        "Session",
        "SESSION",
        "a/b",
        "..",
        "",
        long.as_str(),
    ] {
        let spec = AgentSpec {
            id: Some(AgentId(id.into())),
            ..AgentSpec::default()
        };
        let made = spawn(world, Origin::Root, spec);
        let refused = SpawnRefused::NotALogName(AgentId(id.into()));
        assert_eq!(made, Err(refused), "{id:?}");
    }
    let longest = "x".repeat(rig_memory::MAX_NEW_LOG_NAME);
    let spec = AgentSpec {
        id: Some(AgentId(longest)),
        ..AgentSpec::default()
    };
    assert!(spawn(world, Origin::Root, spec).is_ok());
}

/// Two agents whose ids differ only in case would share one log on a disk
/// that ignores case.
#[test]
fn an_id_is_taken_whatever_its_case() {
    let mut app = app();
    let world = app.world_mut();
    let spec = |id: &str| AgentSpec {
        id: Some(AgentId(id.into())),
        ..AgentSpec::default()
    };
    assert!(spawn(world, Origin::Root, spec("helper")).is_ok());
    let made = spawn(world, Origin::Root, spec("Helper"));
    assert_eq!(made, Err(SpawnRefused::IdTaken(AgentId("Helper".into()))));
}

#[test]
fn an_id_is_free_again_once_its_agent_is_gone_or_its_spawn_was_refused() {
    let mut app = app();
    app.add_plugins(crate::AgentPlugin);
    let world = app.world_mut();
    let spec = |tools: Option<Vec<String>>| AgentSpec {
        id: Some(AgentId("helper".into())),
        tools,
        ..AgentSpec::default()
    };
    let refused = spawn(world, Origin::Root, spec(Some(vec!["missing".into()])));
    assert!(matches!(refused, Err(SpawnRefused::Tool(_))), "{refused:?}");
    let first = spawn(world, Origin::Root, spec(None));
    assert!(first.is_ok(), "{first:?}");
    let taken = spawn(world, Origin::Root, spec(None));
    assert_eq!(taken, Err(SpawnRefused::IdTaken(AgentId("helper".into()))));
    if let Ok(first) = first {
        world.despawn(first);
    }
    let again = spawn(world, Origin::Root, spec(None));
    assert!(again.is_ok(), "{again:?}");
}
