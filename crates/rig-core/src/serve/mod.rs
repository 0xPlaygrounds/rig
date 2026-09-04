//! What a handler author implements, and the one erasure a bus takes.
//!
//! A handler is a [`Serve`]: `type Family`, a descriptor, and an
//! `async fn serve(&self, kind, sink)` that answers into its
//! [`OutcomeSink`]. Provider and tool authors keep implementing the
//! impl-side traits exactly as before; the [`adapters`] wrap them. An
//! out-of-tree kind (`EffectKind::Custom`) or a replayer implements
//! [`Serve`] directly; a handler that streams writes through
//! [`OutcomeSink::writer`] (a [`StreamWriter`]) and never names a block id. [`ErasedHandler`] is rig-core's **only** erasure: the
//! handler-table entry a bus runtime (`rig_bus`) carries, and what
//! [`serve_inline`] runs without a bus.
//!
//! The driver seam — [`OutcomeSink::unary`], [`OutcomeSink::stream`],
//! [`OutcomeSink::with_tap`], [`ErasedHandler::handle`], [`StreamTap`] — is
//! what a bus driver builds to hand a handler its sink and observe what it
//! answers. `rig_bus` is one such driver; a second runtime (an ECS
//! schedule) is another.

pub mod adapters;
mod handler;
mod recorder;
mod writer;

pub use handler::{
    DetachedSink, ErasedHandler, HandlerFuture, OnEvent, OnOutcome, OutcomeSink, Serve, SinkClosed,
    StreamTap, cancelled, events_from_response, finish_unary, serve_inline, stream_truncated,
};
pub use recorder::Recorder;
pub use writer::StreamWriter;

/// A driver's sizing and serving policy: what a program was recorded
/// under and what a host runs it under. Serve-side data, so a log names no
/// runtime and any driver — rig-bus's, a host's own — states its policy in
/// the same terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServingPolicy {
    /// Commands the driver buffers, bus-wide, before a dispatch parks at
    /// its send stage until the driver drains. The bound holds across every
    /// dispatcher and every dispatch; the caller of a dispatch is never
    /// blocked.
    pub command_capacity: usize,
    /// Stream events buffered per streaming dispatch before the handler
    /// stalls (the client-side pause point).
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
