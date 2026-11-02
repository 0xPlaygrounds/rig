use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::effect::{Decider, EffectKind};

use super::RequestOf;
use crate::AgentPlugin;
use crate::agent::StopTurn;
use crate::spawn::spawn_root;
use crate::stage::{Decision, EffectRequest, Outcome, Settled, SupersededBy, follow_successor};

/// A kernel app, started.
fn app() -> App {
    let mut app = App::new();
    app.add_plugins(AgentPlugin);
    app.finish();
    app.update();
    app
}

/// An open effect, which a plugin answers.
fn open() -> EffectRequest {
    EffectRequest {
        kind: EffectKind::Custom {
            kind: Arc::from("work"),
            payload: serde_json::Value::Null,
        },
        handler: None,
    }
}

/// Whether `entity` is still there.
fn alive(app: &App, entity: Entity) -> bool {
    app.world().get_entity(entity).is_ok()
}

/// What a plugin follows its effect by.
#[derive(Component)]
struct Marker;

/// A request that is an effect sent again keeps its owner: the owner going
/// takes the retry with it.
#[test]
fn a_retried_request_still_goes_with_its_owner() {
    let mut app = app();
    app.add_observer(follow_successor::<Marker>);
    let world = app.world_mut();
    let owner = world.spawn_empty().id();
    let first = world.spawn((open(), Marker, RequestOf(owner))).id();
    let retry = world.spawn(open()).id();
    world.entity_mut(first).insert(SupersededBy(retry));
    app.update();
    assert_eq!(
        app.world().get::<RequestOf>(retry).map(|of| of.0),
        Some(owner)
    );
    assert!(app.world().get::<RequestOf>(first).is_none());
    app.world_mut().despawn(owner);
    assert!(!alive(&app, retry));
}

/// An effect's requests end as it settles, while it is still there.
#[test]
fn a_request_ends_when_its_owner_settles() {
    let mut app = app();
    let owner = app.world_mut().spawn(open()).id();
    let request = app.world_mut().spawn(RequestOf(owner)).id();
    app.update();
    assert!(alive(&app, request));
    if let Some(mut decision) = app.world_mut().get_mut::<Decision>(owner) {
        decision.decide(Decider::plugin("test"), Outcome::custom("done"));
    }
    app.update();
    assert!(app.world().get::<Settled>(owner).is_some());
    assert!(!alive(&app, request));
}

/// A stop that reaches an idle agent ends its requests; a plugin's stop
/// of its turn alone leaves them.
#[test]
fn a_stop_ends_the_requests_of_an_idle_agent_it_reaches() {
    let mut app = app();
    let agent = spawn_root(app.world_mut()).id();
    let request = app.world_mut().spawn(RequestOf(agent)).id();
    app.world_mut()
        .trigger(StopTurn::plugin_alone(agent, "test", "its turn alone"));
    app.update();
    assert!(alive(&app, request));
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    assert!(alive(&app, agent));
    assert!(!alive(&app, request));
}

/// A request inserted for an owner that is gone or settled is refused.
#[test]
fn a_request_to_a_missing_or_settled_owner_is_refused() {
    let mut app = app();
    let gone = app.world_mut().spawn_empty().id();
    app.world_mut().despawn(gone);
    let settled = app.world_mut().spawn(Settled).id();
    let refused = [gone, settled].map(|owner| app.world_mut().spawn(RequestOf(owner)).id());
    app.world_mut().flush();
    assert!(refused.iter().all(|&request| !alive(&app, request)));
}
