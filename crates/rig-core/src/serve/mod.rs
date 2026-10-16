#![forbid(unsafe_code)]

//! Handlers return a [`Reply`]: an outcome or an owned stream. [`Dispatch`]
//! carries identity, requested delivery mode and scopes. [`ErasedHandler`]
//! is the shared registry boundary; drivers own polling and cancellation.
//! Provider and tool authors retain their domain traits through [`adapters`].
//! [`Reply::written`] offers a writer that mints stream block identities.
//!
//! ```
//! use rig_core::serve::Dispatch;
//! use rig_core::effect::EffectId;
//!
//! let dispatch = Dispatch::new(EffectId::from_raw(1), false);
//! assert!(!dispatch.is_stream());
//! ```

pub mod adapters;
pub use adapters::Serving;
mod handler;
mod layer;
mod map;
mod record_buffer;
mod recorder;
mod writer;

pub use handler::{
    Dispatch, ErasedHandler, HandlerFuture, Observe, Reply, Serve, SinkClosed, StreamTap,
    cancelled, serve_inline, serve_inline_with, stream_truncated,
};
pub use layer::{AfterServe, BeforeServe, Intercept, Layer};
pub use record_buffer::RecordBuffer;
pub use recorder::{DispatchContext, Recorder, catch_panics};
pub use writer::StreamWriter;
