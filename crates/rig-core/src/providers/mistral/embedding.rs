//! Mistral's embedding model identifiers and its batching cap.
//!
//! The requests themselves run on the shared OpenAI embeddings wire, whose
//! [`MISTRAL`](crate::providers::openai::wire::MISTRAL) dialect carries the
//! `/v1/embeddings` path, the 256-input cap below, and Codestral Embed's
//! `output_dimension` spelling for a requested width.

pub const MISTRAL_EMBED: &str = "mistral-embed";
/// Codestral embedding model with configurable output dimensions.
pub const CODESTRAL_EMBED: &str = "codestral-embed";

/// Most inputs Mistral accepts in one `/v1/embeddings` request. Verified
/// against the live API: 256 succeeds, 257 is rejected with
/// `"Too many inputs in request, split into more batches."`.
pub const MAX_DOCUMENTS: usize = 256;
