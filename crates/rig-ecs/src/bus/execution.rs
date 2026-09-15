//! Where a handler's running work lives.
//!
//! The effect entity owns it: the initial task is a field of
//! [`Serving`], a stream's worker a field of
//! [`Streaming`]. Removing the component,
//! despawning the effect or dropping the world cancels the work by
//! dropping it — no side table to keep coherent, and no state in which
//! flight and execution disagree.
//!
//! Browser wasm is the exception: the single-threaded pool's `Task` is
//! neither `Send` nor `Sync` there, and a provider client is `!Send`, so
//! the work stays in the world's `NonSend`
//! `Executions` table keyed by effect entity.
//! This module is the only place that difference is spelled: the driver
//! systems take [`ExecutionStore`], exclusive code takes [`store_of`], and
//! both targets see one signature.

use super::effect::{Serving, Streaming};
use bevy_ecs::prelude::*;
use bevy_ecs::system::EntityCommands;
use bevy_ecs::world::EntityWorldMut;
use rig_core::serve::Reply;
use rig_core::streaming::StreamEvents;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod owned {
    use super::{
        Entity, EntityCommands, EntityWorldMut, Reply, Serving, StreamEvents, Streaming, World,
    };
    use bevy_tasks::{Task, futures::check_ready};
    use core::marker::PhantomData;

    /// The initial task, owned by the effect entity through [`Serving`].
    pub type OwnedTask = Task<Reply>;

    /// The store the driver systems take: nothing, because the entity owns
    /// its work.
    #[derive(bevy_ecs::system::SystemParam)]
    pub struct ExecutionStore<'w> {
        marker: PhantomData<&'w ()>,
    }

    impl ExecutionStore<'_> {
        /// The store for one operation.
        pub fn as_mut(&mut self) -> StoreMut<'_> {
            StoreMut {
                marker: PhantomData,
            }
        }
    }

    /// The store an exclusive caller holds across one operation.
    pub struct StoreOwned<'w> {
        marker: PhantomData<&'w ()>,
    }

    impl StoreOwned<'_> {
        pub fn as_mut(&mut self) -> StoreMut<'_> {
            StoreMut {
                marker: PhantomData,
            }
        }
    }

    /// Hand an exclusive caller the store.
    pub fn store_of(_world: &mut World) -> StoreOwned<'_> {
        StoreOwned {
            marker: PhantomData,
        }
    }

    /// Take the initial task off an effect leaving ordinary collection.
    pub fn take_serving(effect: &mut EntityWorldMut<'_>) -> Option<OwnedTask> {
        effect.take::<Serving>().map(|serving| serving.0)
    }

    /// The operations the driver performs on a handler's running work.
    pub struct StoreMut<'a> {
        marker: PhantomData<&'a ()>,
    }

    impl StoreMut<'_> {
        /// The effect owns its initial task.
        pub fn own_task(&mut self, effect: &mut EntityCommands<'_>, task: OwnedTask) {
            effect.insert(Serving(task));
        }

        /// Whether the initial task on the entity has finished.
        pub fn poll_task(&mut self, _entity: Entity, serving: &mut Serving) -> Option<Reply> {
            check_ready(&mut serving.0)
        }

        /// Whether an initial task taken off the entity has finished.
        pub fn poll_owned(&mut self, _entity: Entity, task: &mut OwnedTask) -> Option<Reply> {
            check_ready(task)
        }

        /// Start a stream's worker, owned by the returned component.
        pub fn spawn_stream(
            &mut self,
            _entity: Entity,
            stream: StreamEvents,
            capacity: usize,
        ) -> Streaming {
            Streaming::spawn(stream, capacity)
        }

        /// Cancel a stream's worker now, before the component itself goes.
        pub fn drop_worker(&mut self, _entity: Entity, streaming: &mut Streaming) {
            streaming.worker.take();
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod table {
    use super::{
        Entity, EntityCommands, EntityWorldMut, Reply, Serving, StreamEvents, Streaming, World,
    };
    use crate::bus::effect::Executions;
    use bevy_ecs::system::NonSendMut;
    use bevy_tasks::futures::check_ready;

    /// The initial task stays in the table; the entity carries the marker.
    pub type OwnedTask = ();

    /// The store the driver systems take: the world's execution table.
    #[derive(bevy_ecs::system::SystemParam)]
    pub struct ExecutionStore<'w> {
        table: NonSendMut<'w, Executions>,
    }

    impl ExecutionStore<'_> {
        pub fn as_mut(&mut self) -> StoreMut<'_> {
            StoreMut {
                table: &mut self.table,
            }
        }
    }

    /// The table, held by an exclusive caller across one operation.
    pub struct StoreOwned<'w> {
        table: bevy_ecs::change_detection::Mut<'w, Executions>,
    }

    impl StoreOwned<'_> {
        pub fn as_mut(&mut self) -> StoreMut<'_> {
            StoreMut {
                table: &mut self.table,
            }
        }
    }

    pub fn store_of(world: &mut World) -> StoreOwned<'_> {
        StoreOwned {
            table: world.non_send_mut::<Executions>(),
        }
    }

    pub fn take_serving(effect: &mut EntityWorldMut<'_>) -> Option<OwnedTask> {
        effect.take::<Serving>().map(|_| ())
    }

    pub struct StoreMut<'a> {
        table: &'a mut Executions,
    }

    impl StoreMut<'_> {
        pub fn own_task(&mut self, effect: &mut EntityCommands<'_>, task: bevy_tasks::Task<Reply>) {
            self.table.tasks.insert(effect.id(), task);
            effect.insert(Serving);
        }

        pub fn poll_task(&mut self, entity: Entity, _serving: &mut Serving) -> Option<Reply> {
            let reply = check_ready(self.table.tasks.get_mut(&entity)?)?;
            self.table.tasks.remove(&entity);
            Some(reply)
        }

        pub fn poll_owned(&mut self, entity: Entity, _task: &mut OwnedTask) -> Option<Reply> {
            self.poll_task(entity, &mut Serving)
        }

        pub fn spawn_stream(
            &mut self,
            entity: Entity,
            stream: StreamEvents,
            capacity: usize,
        ) -> Streaming {
            let (streaming, worker) = Streaming::spawn(stream, capacity);
            self.table.streams.insert(entity, worker);
            streaming
        }

        pub fn drop_worker(&mut self, entity: Entity, _streaming: &mut Streaming) {
            self.table.streams.remove(&entity);
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use owned::*;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use table::*;
