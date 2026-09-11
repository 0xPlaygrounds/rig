//! Private cleanup facts supplement public cancellation trace assertions.
use super::*;
use rig_core::effect::EffectKind;

#[test]
fn held_despawn_bookkeeping_does_not_outlive_the_entity() {
    let mut world = World::new();
    crate::bus::Bus::default().install(&mut world);
    Witnessing::install(
        &mut world,
        Arc::new(rig_core::observe::ObservationLog::default()),
    );
    let held = world
        .spawn((
            PendingEffect::new(
                "test",
                EffectKind::Custom {
                    kind: "test".into(),
                    payload: serde_json::Value::Null,
                },
            ),
            Held,
        ))
        .id();
    world.despawn(held);
    assert!(world.resource::<Despawning>().0.is_empty());
}
