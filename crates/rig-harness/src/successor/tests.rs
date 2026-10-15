use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use rig_core::effect::EffectKind;
use rig_ecs::AgentPlugin;
use rig_ecs::stage::{Decision, Holds, Started};

use super::*;
use rig_ecs::stage::{HoldOf, follow_successor};

/// The owner's marker on its effect.
#[derive(Component)]
struct Mine;

fn open(kind: &str) -> EffectRequest {
    let kind = EffectKind::Custom {
        kind: kind.into(),
        payload: serde_json::Value::Null,
    };
    EffectRequest {
        kind,
        handler: None,
    }
}

#[test]
fn a_resent_effect_keeps_its_owner_and_its_hold() {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_observer(follow_successor::<Mine>);
    let world = app.world_mut();
    let held = world.spawn(open("deploy")).id();
    let owned = world
        .spawn((open("check"), Mine, HoldOf(held), Name::new("check")))
        .id();
    app.update();
    assert!(app.world().get::<Started>(owned).is_some());

    app.world_mut().commands().queue(resend(owned));
    app.update();
    let world = app.world_mut();
    let successor = world
        .query_filtered::<Entity, (With<Mine>, Without<Settled>)>()
        .single(world)
        .ok();
    assert!(successor.is_some_and(|successor| successor != owned));
    assert!(world.get::<Settled>(owned).is_some(), "superseded");
    assert!(world.get::<Holds>(held).is_some());
    assert!(world.get::<Started>(held).is_none());
    let Some(successor) = successor else {
        return;
    };
    assert_eq!(
        world.get::<Name>(successor).map(Name::as_str),
        Some("check")
    );
    assert_eq!(
        world.get::<HoldOf>(successor).map(|hold| hold.0),
        Some(held)
    );

    if let Some(mut decision) = world.get_mut::<Decision>(successor) {
        decision.decide(
            rig_core::effect::Decider::plugin("test"),
            rig_ecs::stage::Outcome::custom("checked"),
        );
    }
    // Settled, the successor is despawned with the next frame's loop, which
    // releases its hold.
    app.update();
    app.update();
    assert!(app.world().get::<Started>(held).is_some());
}
