use bevy_ecs::system::RunSystemOnce;

use super::*;

/// Whether the kernel holds `agent`'s answered turn open for a message.
fn coming(world: &mut World, agent: Entity) -> bool {
    let coming = move |inboxes: Inboxes| inboxes.coming(agent);
    world.run_system_once(coming).ok() == Some(true)
}

/// A steering message for `agent`, not decided yet.
fn steer(world: &mut World, agent: Entity, text: &str) -> Entity {
    let message = Incoming::new(text, DeliveryMode::Steer);
    world
        .spawn((message, Decision::default(), ForAgent(agent)))
        .id()
}

#[test]
fn a_message_superseded_or_settled_is_not_on_its_way() {
    let mut world = World::new();
    let agent = world.spawn_empty().id();
    let held = steer(&mut world, agent, "count: !{printf 3}");
    assert!(coming(&mut world, agent));
    // Its successor is on its way in its place; once that one starts,
    // nothing is.
    let expanded = steer(&mut world, agent, "count: 3");
    world.entity_mut(held).insert(SupersededBy(expanded));
    assert!(coming(&mut world, agent));
    let at = std::time::SystemTime::UNIX_EPOCH;
    world.entity_mut(expanded).insert(Started { at });
    assert!(!coming(&mut world, agent));
    // A message that settled without starting is not coming either.
    let dropped = steer(&mut world, agent, "never mind");
    assert!(coming(&mut world, agent));
    world.entity_mut(dropped).insert(Settled);
    assert!(!coming(&mut world, agent));
}
