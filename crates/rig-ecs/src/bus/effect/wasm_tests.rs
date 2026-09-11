//! Executed WASM coverage for private worker ownership, selected with --lib.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "contract assertions"
)]
use crate::bus::{Bus, InFlight};
use bevy_ecs::prelude::*;
use std::{cell::Cell, rc::Rc};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
async fn local_streams_drop_on_marker_removal_scheduled_despawn_replacement_and_shutdown() {
    use super::{Executions, Streaming};
    struct Local(Rc<Cell<usize>>);
    impl Drop for Local {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let drops = Rc::new(Cell::new(0));
    let stream = || {
        let local = Local(drops.clone());
        Box::pin(futures::stream::poll_fn(move |_| {
            let _local = &local;
            std::task::Poll::Pending
        })) as rig_core::streaming::StreamEvents
    };
    let mut world = World::new();
    Bus::default().install(&mut world);
    async fn dropped(drops: &Cell<usize>, expected: usize) {
        for _ in 0..1000 {
            if drops.get() == expected {
                return;
            }
            // Yield to the browser event loop, not only this Rust test task.
            rig_core::wasm_compat::sleep(std::time::Duration::from_millis(1)).await;
        }
        assert_eq!(
            drops.get(),
            expected,
            "cancelled local worker was not dropped"
        );
    }
    let (streaming, task) = Streaming::spawn(stream(), 1);
    let entity = world
        .spawn((
            InFlight {
                key: "local".into(),
            },
            streaming,
        ))
        .id();
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(entity, task);
    let (streaming, task) = Streaming::spawn(stream(), 1);
    world.entity_mut(entity).insert(streaming);
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(entity, task);
    dropped(&drops, 1).await;
    world.entity_mut(entity).remove::<InFlight>();
    dropped(&drops, 2).await;
    let (streaming, task) = Streaming::spawn(stream(), 1);
    world.entity_mut(entity).insert((
        InFlight {
            key: "local".into(),
        },
        streaming,
    ));
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(entity, task);
    let mut schedule = Schedule::default();
    schedule.add_systems(move |mut commands: Commands| {
        commands.entity(entity).despawn();
    });
    schedule.run(&mut world);
    dropped(&drops, 3).await;
    assert!(world.non_send::<Executions>().streams.is_empty());
    let (streaming, task) = Streaming::spawn(stream(), 1);
    let entity = world
        .spawn((
            InFlight {
                key: "local".into(),
            },
            streaming,
        ))
        .id();
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(entity, task);
    drop(world);
    dropped(&drops, 4).await;
}
