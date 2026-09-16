//! The chat-completions wire's streamed reply shapes.
//!
//! The types themselves live with the wire that decodes them (the
//! crate-internal `openai::wire::dto` module) — there is one definition of
//! each, read by the one [`ChatDecoder`](crate::providers::openai::wire::ChatDecoder)
//! that serves both the unary body and the stream. This module re-exports the
//! two that are public API, so `openai::FinishReason` and
//! `openai::StreamingCompletionResponse` keep resolving.

pub use crate::providers::openai::wire::dto::{FinishReason, StreamingCompletionResponse};
