//! xAI's completion model identifiers and its own view of a reply.
//!
//! xAI is a Responses dialect (see [`DIALECT`](super::DIALECT)), so it has no
//! completion model of its own. [`CompletionResponse`] is the shared
//! Responses document type: xAI answers the same envelope, and a completion
//! carries it verbatim on
//! [`CompletionResponse::raw`](crate::completion::CompletionResponse::raw).
//! Its *request* shape is xAI's own, and lives in [`api`](super::api).

pub use crate::providers::openai::responses_api::CompletionResponse;

// The model identifiers are provider data and live beside the dialect in
// `super`; they are re-exported here so `xai::completion::GROK_3` keeps
// resolving.
pub use super::{
    GROK_2_1212, GROK_2_IMAGE_1212, GROK_2_VISION_1212, GROK_3, GROK_3_FAST, GROK_3_MINI,
    GROK_3_MINI_FAST, GROK_4,
};
