//! Synthetic guard for call-order observations independent of entity layout.

use super::*;
use bevy_ecs::system::RunSystemOnce;

#[test]
fn calls_follow_slot_order_instead_of_spawn_order() {
    let mut world = World::new();
    world.init_resource::<Seen>();
    let turn = world.spawn_empty().id();
    for (index, name) in [(2, "unexpected"), (1, "beta"), (0, "alpha")] {
        let call = rig::message::ToolCall::from_wire(
            name,
            rig::message::ToolFunction::new(name.into(), serde_json::json!({})),
        );
        world.spawn((
            ChildOf(turn),
            ToolCallSlot {
                index,
                id: call.id,
                provider: None,
                name: name.into(),
            },
        ));
    }
    world.run_system_once(observe_calls).expect("observer runs");
    assert_eq!(
        world.resource::<Seen>().observation.tool_calls,
        ["alpha", "beta", "unexpected"]
    );
}
