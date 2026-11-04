//! The integration tests of rig-cassette, linked into one binary: one module
//! per cassette-backed provider suite (`providers/<provider>/`), the runtime
//! scenarios, the corpus guards, the effect-bus verification suite `verify`,
//! and the cassette safety checks, which scan every provider's fixtures.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_test_support::cassettes;
#[cfg(all(feature = "http", feature = "agent"))]
use rig_test_support::{
    cache_conformance, goldens, history_survival, matrix, raw_capture, reasoning, stream_faults,
    support,
};

#[path = "common/cassette_safety.rs"]
mod cassette_safety;
// Declared before the suites that use its macros: `verify` and the
// providers' corpus matrices.
#[cfg(feature = "agent")]
#[macro_use]
#[allow(dead_code, reason = "each suite uses a different subset of the corpus")]
#[path = "corpus/mod.rs"]
mod corpus;
#[cfg(all(feature = "http", feature = "agent"))]
#[allow(
    dead_code,
    reason = "each provider and the runtime scenarios exercise their own subset of matrix cells"
)]
#[path = "common/corpus_matrix.rs"]
mod corpus_matrix;
#[cfg(all(feature = "http", feature = "agent"))]
#[allow(dead_code, reason = "each provider uses part of the shared driver")]
#[path = "common/image_inputs.rs"]
mod image_inputs;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "common/request_identity.rs"]
mod request_identity;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "common/responses_route.rs"]
mod responses_route;

#[cfg(all(feature = "http", feature = "agent"))]
#[allow(clippy::unwrap_used, clippy::unreachable)]
#[path = "providers/anthropic/mod.rs"]
mod anthropic;
#[cfg(all(feature = "bedrock", feature = "agent"))]
#[path = "providers/bedrock/mod.rs"]
mod bedrock;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "providers/chatgpt/mod.rs"]
mod chatgpt;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "providers/cohere/mod.rs"]
mod cohere;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "providers/copilot/mod.rs"]
mod copilot;
#[cfg(all(feature = "http", feature = "agent"))]
#[allow(clippy::unwrap_used, clippy::unreachable)]
#[path = "providers/deepseek/mod.rs"]
mod deepseek;
#[cfg(all(feature = "http", feature = "agent"))]
#[allow(clippy::unwrap_used, clippy::unreachable)]
#[path = "providers/doubleword/mod.rs"]
mod doubleword;
#[cfg(all(feature = "http", feature = "agent"))]
#[allow(clippy::unwrap_used, clippy::unreachable)]
#[path = "providers/gemini/mod.rs"]
mod gemini;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "providers/groq/mod.rs"]
mod groq;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "providers/llamacpp/mod.rs"]
mod llamacpp;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "providers/mistral/mod.rs"]
mod mistral;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "providers/mistralrs/mod.rs"]
mod mistralrs;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "providers/ollama/mod.rs"]
mod ollama;
#[cfg(all(feature = "http", feature = "agent"))]
#[allow(clippy::unwrap_used, clippy::unreachable)]
#[path = "providers/openai/mod.rs"]
mod openai;
#[cfg(all(feature = "http", feature = "agent"))]
#[allow(clippy::unwrap_used)]
#[path = "providers/openrouter/mod.rs"]
mod openrouter;
#[cfg(all(feature = "http", feature = "agent"))]
#[path = "providers/perplexity/mod.rs"]
mod perplexity;
#[cfg(all(feature = "http", feature = "agent"))]
#[allow(clippy::unwrap_used, clippy::unreachable)]
#[path = "providers/venice/mod.rs"]
mod venice;
#[cfg(all(feature = "http", feature = "agent"))]
#[allow(clippy::unwrap_used)]
#[path = "providers/xai/mod.rs"]
mod xai;

#[cfg(feature = "http")]
mod cassette_cache_prefix;
#[cfg(feature = "http")]
mod cassette_history_survival;
#[cfg(feature = "http")]
mod cassette_usage_census;
#[cfg(feature = "http")]
mod chat_parity;
#[cfg(all(feature = "http", feature = "agent"))]
mod runtime;
#[cfg(feature = "agent")]
#[path = "verify/main.rs"]
mod verify;
