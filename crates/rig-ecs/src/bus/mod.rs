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
//!   `On<Add, EffectOutcome>` observer) and, once the record closed, the
//!   [`Landed`] entity event bubbling up `ChildOf` to the turn, the run and
//!   the agent.
//! - **A handler is an entity.** [`Handlers::register`] spawns one with a
//!   [`Bound`] component (the key and the descriptor, serde; immutable, so
//!   its hooks keep [`HandlerIndex`] exact), a `Name`, and the erased
//!   handler as [`Handler`] on the same entity (the erased handler itself
//!   lives in the non-send `HandlerTable`). An effect names its handler by
//!   [`ServedBy`] (its inverse [`Serves`]). The registry is a query;
//!   deregistration is a despawn.
//! - **The driver is two systems.** [`BusSet::Dispatch`] takes pending
//!   effects in [`Seq`] order and spawns each handler's future on the task
//!   pool, held in the effect entity as Bevy's own `Task` ([`Serving`],
//!   [`Streaming`]; dropping the component cancels it); [`BusSet::Collect`]
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
//!   nothing is in flight on it (the handler's [`Serves`]), and refuses
//!   (with a `Request` report, before any dispatch) an effect whose
//!   ancestor is in flight on its own key: it could only wait forever.
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
//!   despawn before that closes it as cancelled. `rig_cassette::ecs::Replay`
//!   loads records as effect entities with their recorded ids and registers
//!   a replayer that answers each by id. An ECS recording also
//!   keeps consumer delivery batches. Policy-visible replay requires these
//!   boundaries and, for streams, kept events and error items; it does not
//!   reconstruct arbitrary resources or elapsed time.
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
//! - **A checkpoint is the world.** [`crate::checkpoint::save_world`] takes
//!   every entity with a reflected component — the effect entities (intent,
//!   ids, outcomes, stream state, causality, scope) and the handlers'
//!   [`Bound`]s among them; [`crate::checkpoint::load_world`] spawns them
//!   back, ids reserved, outcomes kept, so `Dispatch` re-issues safe
//!   unanswered intents. Load refuses unfinished streams with delivered
//!   progress: no provider cursor prevents a repeated prefix. Completed
//!   streams restore without re-executing their handlers. In-flight state
//!   ([`Serving`], [`Streaming`], [`Handler`]) is never saved.
//!
//! # The schedule
//!
//! [`BusPlugin`] adds a [`RigSchedule`] after `Update` with four sets in
//! order — [`BusSet::Gate`], [`BusSet::Dispatch`], [`BusSet::Collect`],
//! [`BusSet::Judge`] — and [`RigEnd`] after it, and runs them **once per
//! app update**. Users add their systems to `RigSchedule`, ordered against
//! the sets. A host updates the app when there is something to do: every
//! task the bus spawns raises [`Wake`] as it finishes or delivers, a host
//! system that needs another pass raises it too, and the plugin's default
//! runner ([`woken_runner`]) updates on that signal, so nothing spins.
//! Intake bounds apply per update; async readiness can require later
//! updates. Cassette's `rig_cassette::ecs::ReplayPlugin` installs replay
//! diagnosis in `RigEnd`, running only after a pass that raised nothing.
//! [`BusPlugin::install`] is the runtime's world half, for a test that drives
//! `RigSchedule` itself.
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

pub mod binding;
pub mod collect;
pub mod diagnostics;
pub mod dispatch;
pub mod effect;
pub mod handlers;
pub mod hold;
pub mod plugin;
pub mod record;
pub mod reflect;
pub mod stream_delivery;
pub mod witness;

pub use binding::{
    CredentialRef, MaterializeError, MaterializeFailed, MaterializeReport, Materializer,
    ProviderBinding, Secret, materialize, materialize_bindings,
};
pub use collect::{Landed, Landing, StreamingView, collect_streams, collect_tasks, settle};
pub use diagnostics::{
    BindingReport, CredentialGuidance, ProviderDiagnostics, RegisteredProvider,
    provider_diagnostics,
};
pub use dispatch::{Candidate, CandidateView, dispatch, handler_unavailable, reentrant};
pub use effect::{
    Answer, Asked, EffectOutcome, Held, IdCounter, InFlight, Issued, PendingEffect, Publishing,
    Reserved, Scope, Seq, SeqCounter, Serving, Streamed, Streaming, Tasks, ToolInputs, ToolOutputs,
    Typed, WorldEffect, WorldOutcome,
};
pub use handlers::{
    Bound, Handler, HandlerIndex, Handlers, Registry, Served, ServedBy, Serves, WorldHandler,
    WorldServe, answered,
};
pub use hold::{HoldOwners, HoldRefused, acquire_hold, release_hold};
pub use plugin::{BusPlugin, BusSet, Policy, RigEnd, RigSchedule, Wake, woken_runner};
pub use record::{
    Observed, ObservedState, Recording, WorldObserver, record_bound, record_cancelled,
};
pub use stream_delivery::StreamItemsDelivered;
pub use witness::{
    AdapterOperation, BUS_EMITTER, Despawning, SubjectWalk, Subjects, Witnessing, bus_emitter,
};

pub use rig_core::serve::ServingPolicy;
