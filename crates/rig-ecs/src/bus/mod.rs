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
//!   Despawning an effect cancels it — its task drops, its owned reply
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
//!   entity, and a user system inserts [`Answer<E>`]. Open keys accept
//!   [`WorldOutcome`]. `Collect` publishes both in submission order as
//!   [`EffectOutcome`]; later submissions become visible next pass.
//! - **The log is a fold over entities.** With a [`Recording`] resource
//!   installed ([`Recording::install`]), `Dispatch` opens a record as it
//!   takes an effect and `Collect` closes it as the outcome lands; a
//!   despawn before that closes it as cancelled. Under the `replay` feature
//!   a `Replay` loads a log's records as effect entities with their
//!   recorded ids and registers a replayer that answers each by id. An ECS
//!   recording also keeps consumer delivery batches. Policy-visible replay
//!   requires these boundaries and, for streams, kept events and error items;
//!   it does not reconstruct arbitrary resources or elapsed time.
//! - **Decisions are witnessed beside the log.** With a [`Witnessing`]
//!   resource installed ([`Witnessing::install`]), the bus's own systems
//!   emit a typed `rig_core::observe::Observation` at each decision site —
//!   an intent issued, deferred by the intake bound or refused by the
//!   driver; a stream that ended before its terminal record (with its last
//!   frames); an in-flight cancellation; a landed record and any layer
//!   verdict that differed from it — and lifecycle observers see the
//!   transitions a user system makes by component write: `Held` added and
//!   removed, an outcome landed before dispatch (a `Gate` denial), an
//!   outcome overwritten after the record closed (a `Judge` replacement).
//!   A host policy names itself through [`Witnessing::emit`]. The trace is
//!   an analysis artifact: never replay identity, never part of the log.
//! - **A scene is a checkpoint.** [`Scene::save`] takes the effect entities
//!   (intent, ids, outcomes, stream state, causality, scope) and the bound descriptors as
//!   serde; [`Scene::load`] spawns them back, ids reserved, outcomes kept,
//!   so `Dispatch` re-issues safe unanswered intents. Load refuses unfinished
//!   streams with delivered progress: no provider cursor prevents a repeated
//!   prefix. Completed streams restore without re-executing their handlers.
//!
//! # The schedule
//!
//! [`Bus::install`] adds a [`RigSchedule`] with four sets in order —
//! [`BusSet::Gate`], [`BusSet::Dispatch`], [`BusSet::Collect`],
//! [`BusSet::Judge`]. The host runs it **to quiescence** by calling
//! [`run_to_quiescence`] once per tick from the schedule or loop it owns:
//! as long as a bus system reports [`Progress`], the schedule runs again
//! (capped at [`QUIESCENCE_CAP`] passes, a `warn!` when reached). Users add
//! their systems to `RigSchedule`, ordered against the sets, never beside
//! the runner: a system beside the runner sees one pass, a system in
//! `RigSchedule` sees every pass. Intake bounds can defer a dispatch to the
//! next tick; async readiness can require later ticks.
//!
//! # What it deliberately does not have
//!
//! No agent, no loop, no memory semantics, no hook trait, no policy
//! vocabulary: those are the crate's later modules, and nothing here may
//! anticipate them. A handler served as a task cannot reach the world; a
//! handler that needs the world is a [`WorldHandler`], or a key bound open
//! ([`Handlers::register_open`]) that a system answers by submitting a
//! [`WorldOutcome`], whatever the family. Streaming answers from a system are not
//! offered (a system answers unary effects).

pub mod collect;
#[cfg(feature = "replay")]
pub mod delivery;
pub mod dispatch;
pub mod effect;
pub mod handlers;
pub mod hold;
pub mod plugin;
pub mod record;
#[cfg(feature = "reflect")]
pub mod reflect;
pub mod scene;
pub mod witness;

#[cfg(feature = "replay")]
pub mod replay;

pub use collect::{Landed, StreamingView, collect_streams, collect_tasks, settle};
pub use dispatch::{Candidate, CandidateView, dispatch, handler_unavailable, reentrant};
pub use effect::{
    Answer, Asked, EffectOutcome, Held, IdCounter, InFlight, Issued, PendingEffect, Publishing,
    Reserved, Scope, Seq, SeqCounter, Serving, Streamed, Streaming, ToolInputs, ToolOutputs, Typed,
    WorldEffect, WorldOutcome,
};
pub use handlers::{
    Bound, HandlerTable, Handlers, Served, WorldHandler, WorldServe, answered, unbound,
};
pub use hold::{HoldOwners, HoldRefused, acquire_hold, release_hold};
pub use plugin::{
    Bus, BusSet, Intake, Policy, Progress, QUIESCENCE_CAP, RigSchedule, install_bus,
    run_to_quiescence,
};
pub use record::{
    Observed, ObservedState, Recording, WorldObserver, record_bound, record_cancelled,
};
pub use scene::{Scene, SceneEffect};
pub use witness::{AdapterOperation, BUS_EMITTER, SubjectWalk, Subjects, Witnessing, bus_emitter};

#[cfg(feature = "replay")]
pub use delivery::ReplayFailure;
#[cfg(feature = "replay")]
pub use replay::{EffectLogResource, Replay};

pub use rig_core::serve::ServingPolicy;
