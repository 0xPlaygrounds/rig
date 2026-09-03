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
mod writer;

pub use handler::{
    ErasedHandler, HandlerFuture, OnEvent, OnOutcome, OutcomeSink, Serve, SinkClosed, StreamTap,
    events_from_response, finish_unary, serve_inline, stream_truncated,
};
pub use writer::StreamWriter;
