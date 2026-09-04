//! The effect bus inside a Bevy `World`: the future `rig-bevy` crate, as a
//! module.
//!
//! # The shape
//!
//! Nothing here is a channel between two tasks. The world is the store and
//! the driver is a system:
//!
//! - **An effect is an entity.** A dispatch is `commands.spawn(PendingEffect
//!   { key, kind })`. The entity moves through [`PendingEffect`] (intent) →
//!   [`InFlight`] (taken, a handler serving it) → [`EffectOutcome`]
//!   (answered); a stream accumulates in [`Streamed`] on the way. There is
//!   no future for a host to hold and nothing to probe: readiness is the
//!   component landing (`Added<EffectOutcome>`, `Changed<Streamed>`, an
//!   `On<Add, EffectOutcome>` observer).
//! - **A handler is an entity.** [`Handlers::register`] spawns one with a
//!   [`Bound`] component (the key and the descriptor, serde) and puts the
//!   erased handler in the world's [`HandlerTable`]. The registry is a
//!   query; deregistration is a despawn.
//! - **The driver is two systems.** [`BusSet::Dispatch`] takes pending
//!   effects in [`Seq`] order and spawns each handler's future on the task
//!   pool, held in the effect entity as Bevy's own `Task`; [`BusSet::Collect`]
//!   reads what finished and writes the outcome component. Neither awaits,
//!   neither blocks (a guard greps for `block_on`).
//! - **Causality is `ChildOf`.** A handler that is a system spawns child
//!   effects `ChildOf` the one it answers; the record's `parent` is read off
//!   the relationship, its `scope` off the nearest [`Scope`] ancestor.
//!   Despawning an effect cancels it — its task drops, its handler's sink
//!   with it, the record says `Cancelled` — and Bevy despawns its
//!   descendants, so a parent's cancel reaches its children with no table.
//! - **Serial serving is a query.** Under
//!   [`ServingPolicy::serial_per_handler`] `Dispatch` takes a key only when
//!   nothing is in flight on it, and refuses (with a `Request` report,
//!   before any dispatch) an effect whose ancestor is in flight on its own
//!   key: it could only wait forever.
//! - **Interception is two system slots.** A user system in [`BusSet::Gate`]
//!   rewrites a [`PendingEffect`] (patch), replaces it with
//!   `EffectOutcome(Err(Denied))` (deny) or holds it with [`Held`] until a
//!   later tick decides; a user system in [`BusSet::Judge`] rewrites an
//!   [`EffectOutcome`] before anything after it reads it. The record was
//!   taken in `Collect`, so it holds what the handler answered: decisions
//!   are program, never record, enforced by ordering.
//! - **A handler that is a system** answers by component write:
//!   [`Handlers::register_world`] binds a `CustomEffect` type to a key; a
//!   dispatch to it lands as an [`Asked<E>`] component on the effect
//!   entity, and a user system inserts [`Answer<E>`]. No sink, no mailbox,
//!   no task.
//! - **The log is a fold over entities.** With a [`Recording`] resource
//!   installed ([`Recording::install`]), `Dispatch` opens a record as it
//!   takes an effect and `Collect` closes it as the outcome lands; a
//!   despawn before that closes it as cancelled. Under the `replay` feature
//!   a [`Replay`] loads a log's records as effect entities with their
//!   recorded ids and registers a replayer that answers each by id.
//! - **A scene is a checkpoint.** [`Scene::save`] takes the effect entities
//!   (intent, ids, outcomes, causality, scope) and the bound descriptors as
//!   serde; [`Scene::load`] spawns them back, ids reserved, outcomes kept,
//!   so `Dispatch` re-issues exactly what had not been answered.
//!
//! # The schedule
//!
//! [`BusPlugin`] adds a [`RigSchedule`] with four sets in order —
//! [`BusSet::Gate`], [`BusSet::Dispatch`], [`BusSet::Collect`],
//! [`BusSet::Judge`] — and runs it **to quiescence** from one exclusive
//! system in `Update`: as long as a plugin system reports [`Progress`], the
//! schedule runs again (capped at [`QUIESCENCE_CAP`] passes, a `warn!`
//! when reached). Users add their systems to `RigSchedule`, ordered
//! against the sets, never to `Update`: a system in `Update` sees one pass,
//! a system in `RigSchedule` sees every pass. Only a handler's real IO
//! costs a tick.
//!
//! # What it deliberately does not have
//!
//! No agent, no loop, no memory semantics, no hook trait, no policy
//! vocabulary: those are the crate's later modules, and nothing here may
//! anticipate them. A handler served as a task cannot reach the world; a
//! handler that needs the world is a [`WorldHandler`], or a key bound open
//! ([`Handlers::register_open`]) that a system answers by inserting the
//! outcome, whatever the family. Streaming answers from a system are not
//! offered (a system answers unary effects).

pub mod collect;
pub mod dispatch;
pub mod effect;
pub mod handlers;
pub mod plugin;
pub mod record;
pub mod scene;

#[cfg(feature = "replay")]
pub mod replay;

pub use collect::{Landed, collect_streams, collect_tasks, settle};
pub use dispatch::{Candidate, dispatch, handler_unavailable, reentrant};
pub use effect::{
    Answer, Asked, EffectOutcome, Held, IdCounter, InFlight, Issued, PendingEffect, Reserved,
    Scope, Seq, SeqCounter, Serving, Streamed, Streaming, Typed, WorldEffect,
};
pub use handlers::{
    Bound, HandlerTable, Handlers, Served, WorldHandler, WorldServe, answered, unbound,
};
pub use plugin::{
    BusPlugin, BusSet, Intake, Policy, Progress, QUIESCENCE_CAP, RigSchedule, run_to_quiescence,
};
pub use record::{Recording, record_bound, record_cancelled};
pub use scene::{Scene, SceneEffect};

#[cfg(feature = "replay")]
pub use replay::{EffectLogResource, Replay};

pub use rig_core::serve::ServingPolicy;
