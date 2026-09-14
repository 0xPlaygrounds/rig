//! Shared Anthropic-compatible wire types, header settings, and model drivers.
//! Concrete Anthropic clients are enabled separately by the `anthropic` feature.

pub mod client;
pub mod completion;
mod observation;
pub mod streaming;
