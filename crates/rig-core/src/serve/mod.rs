//! Handlers return a [`Reply`]: an outcome or an owned stream. [`Dispatch`]
//! carries identity, requested delivery mode and scopes. [`ErasedHandler`]
//! is the shared registry boundary; drivers own polling and cancellation.
//! Provider and tool authors retain their domain traits through [`adapters`].
//! [`Reply::written`] offers a writer that mints stream block identities.

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

/// A driver's sizing and serving policy: what a program was recorded
/// under and what a host runs it under. Serve-side data, so a log names no
/// runtime and any driver — rig-agent's, a host's own — states its policy in
/// the same terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServingPolicy {
    /// Commands the driver buffers, bus-wide, before a dispatch parks at
    /// its send stage until the driver drains. The bound holds across every
    /// dispatcher and every dispatch; the caller of a dispatch is never
    /// blocked.
    pub command_capacity: usize,
    /// Events buffered in rig-agent's bounded consumer queue. rig-ecs polls
    /// owned streams directly and has no corresponding consumer queue.
    pub stream_capacity: usize,
    /// Serve one command at a time per key. `false` serves every command
    /// concurrently; `true` is the cassette-ordered property — a handler
    /// sees its dispatches in the order they arrived.
    ///
    /// Under serial serving a handler must not dispatch to **its own key**
    /// and wait for the answer: that dispatch would queue behind the
    /// command that waits on it. A driver refuses the case it can see with
    /// a `Request` report instead of hanging; a handler that needs its own
    /// key serves it from a second key, or runs with
    /// `serial_per_handler: false`.
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
