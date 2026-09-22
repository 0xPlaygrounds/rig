#![forbid(unsafe_code)]

//! Handlers return a [`Reply`]: an outcome or an owned stream. [`Dispatch`]
//! carries identity, requested delivery mode and scopes. [`ErasedHandler`]
//! is the shared registry boundary; drivers own polling and cancellation.
//! Provider and tool authors retain their domain traits through [`adapters`].
//! [`Reply::written`] offers a writer that mints stream block identities.
//!
//! ```
//! use rig_core::serve::ServingPolicy;
//!
//! let policy = ServingPolicy::default();
//! assert!(!policy.serial_per_handler);
//! ```

pub mod adapters;
mod handler;
mod layer;
mod recorder;
mod writer;

pub use handler::{
    Dispatch, ErasedHandler, HandlerFuture, Observe, Reply, Resolver, Serve, SinkClosed, StreamTap,
    cancelled, deferred, serve_inline, serve_inline_with, stream_truncated,
};
pub use layer::{Decision, Intercept, Layer, Verdict};
pub use recorder::{Origin, Recorder};
pub use writer::StreamWriter;

/// Driver queue capacities and per-handler ordering policy, retained as data
/// for recording and replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServingPolicy {
    /// Commands the driver buffers, bus-wide, before a dispatch parks at
    /// its send stage until the driver drains. The bound holds across every
    /// dispatcher and every dispatch; the caller of a dispatch is never
    /// blocked.
    pub command_capacity: usize,
    /// Driver delivery queue capacity. rig-agent bounds its consumer queue;
    /// rig-ecs uses at least one shared slot plus one sender-reserved slot.
    /// Source-internal buffers and collection work limits are separate.
    pub stream_capacity: usize,
    /// Serves one command at a time per key in arrival order when true;
    /// otherwise permits concurrent execution.
    ///
    /// Serial handlers must not dispatch to their own key and await the result,
    /// which would deadlock. Drivers reject detectable cases as request errors.
    /// Use another key or concurrent serving for nested calls to the same handler.
    pub serial_per_handler: bool,
}

impl Default for ServingPolicy {
    fn default() -> Self {
        Self {
            command_capacity: 16,
            stream_capacity: 64,
            serial_per_handler: false,
        }
    }
}
