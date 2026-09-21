//! An effect bus whose pending work, handlers, and outcomes are world entities.
//!
//! [`BusPlugin`] orders gate, dispatch, collection, and judgement systems in
//! [`RigSchedule`]. Despawning effects cancels owned tasks and descendants;
//! [`Recording`] retains handler answers before policy rewrites. Hosts provide
//! already-built handlers, and world-served handlers answer unary effects only.
//!
//! ```
//! use rig_ecs::bus::BusPlugin;
//! let mut app = bevy_app::App::new();
//! app.add_plugins(BusPlugin::default());
//! app.update();
//! ```

pub mod collect;
pub mod dispatch;
pub mod effect;
pub mod handlers;
pub mod hold;
pub mod plugin;
pub mod record;
pub mod reflect;
pub mod stream_delivery;
pub mod witness;

pub use collect::{Landed, Landing, StreamingView, collect_streams, collect_tasks, settle};
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
