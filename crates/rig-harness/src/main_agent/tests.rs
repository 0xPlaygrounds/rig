use bevy_app::TaskPoolPlugin;
use rig_ecs::AgentPlugin;
use rig_ecs::agent::{AgentId, SpawnedBy};
use rig_ecs::prelude::*;

use super::*;

/// The main agent, as a system finds it.
#[derive(Resource, Default)]
struct Main(Option<Entity>);

/// An app with the kernel and its first agent, which notes the main agent
/// each frame.
fn app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, CurrentAgentPlugin))
        .init_resource::<Main>()
        .add_systems(Update, |agents: MainAgentQuery, mut main: ResMut<Main>| {
            main.0 = main_agent(&agents);
        });
    app.update();
    let first = app.world().resource::<Main>().0;
    (app, first.unwrap_or(Entity::PLACEHOLDER))
}

/// Shows `agent` and runs a frame.
fn show(app: &mut App, agent: Entity) -> Option<Entity> {
    app.world_mut().resource_mut::<ShownAgent>().0 = Some(agent);
    app.update();
    app.update();
    app.world().resource::<Main>().0
}

/// The first agent is current until another top-level agent is shown;
/// showing a subagent keeps its top-level agent current and the subagent
/// shown.
#[test]
fn showing_an_agent_makes_its_top_level_agent_current() {
    let (mut app, first) = app();
    assert!(
        app.world().get::<Agent>(first).is_some(),
        "the first agent is main"
    );
    let id = |app: &App, agent: Entity| app.world().get::<AgentId>(agent).cloned();
    assert_eq!(
        Some(app.world().resource::<CurrentAgent>().0.clone()),
        id(&app, first).map(|id| id.0),
        "the shown first agent is current"
    );
    let other = app.world_mut().spawn(Agent).id();
    app.update();
    assert_eq!(app.world().resource::<Main>().0, Some(first));

    let child = app.world_mut().spawn((Agent, SpawnedBy(other))).id();
    assert_eq!(show(&mut app, child), Some(other));
    assert_eq!(app.world().resource::<ShownAgent>().0, Some(child));
    assert_eq!(
        Some(app.world().resource::<CurrentAgent>().0.clone()),
        id(&app, other).map(|id| id.0)
    );
    assert_eq!(show(&mut app, first), Some(first));
}
