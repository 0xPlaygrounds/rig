//! Test utilities for deterministic completion-model tests.

mod completion;
mod embeddings;
mod http;
#[cfg(all(
    test,
    any(
        feature = "azure",
        feature = "chatgpt",
        feature = "copilot",
        feature = "deepseek",
        feature = "doubleword",
        feature = "groq",
        feature = "huggingface",
        feature = "hyperbolic",
        feature = "llamacpp",
        feature = "minimax",
        feature = "mira",
        feature = "mistral",
        feature = "moonshot",
        feature = "openai",
        feature = "openrouter",
        feature = "perplexity",
        feature = "together",
        feature = "venice",
        feature = "xai",
        feature = "xiaomimimo",
        feature = "zai"
    )
))]
pub(crate) mod internal_streaming_profiles;
mod memory;
mod model_listing;
mod streaming;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod streaming_conformance;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod streaming_conformance_suite;
mod tracing_isolation;

pub use completion::{MockCompletionModel, MockError, MockTurn};
pub use embeddings::{MockEmbeddingModel, MockMultiTextDocument, MockTextDocument};
pub use http::{
    CapturedHttpRequest, HttpErrorStreamingClient, MockHttpResponse, MockStreamingClient,
    RecordingHttpClient, SequencedHttpClient, SequencedStreamingHttpClient,
};
pub use memory::{AppendFailingMemory, CountingMemory, FailingMemory};
pub use model_listing::MockModelLister;
pub use streaming::{MOCK_PROVIDER, MockStreamEvent, mock_final, mock_final_with_total_tokens};
pub use tracing_isolation::{
    scoped_tracing_subscriber_guard, scoped_tracing_subscriber_guard_blocking,
};
