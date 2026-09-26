//! Test utilities for deterministic completion-model tests.

mod completion;
mod embeddings;
mod http;
mod memory;
pub mod observations;
mod relay;
mod streaming;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod streaming_conformance;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod streaming_conformance_suite;
mod tracing_isolation;

pub use completion::{
    MockCompletionModel, MockDecoder, MockError, MockFrame, MockScript, MockTurn, MockWire,
};
pub use embeddings::{MockEmbeddingModel, MockEmbeddings, MockMultiTextDocument, MockTextDocument};
pub use http::{
    CapturedHttpRequest, HttpErrorStreamingClient, MockHttpResponse, MockStreamingClient,
    NonSuccessStreamingClient, RecordingHttpClient, SequencedHttpClient,
    SequencedStreamingHttpClient,
};
pub use memory::{AppendFailingMemory, CountingMemory, FailingMemory};
pub use relay::MockRelay;
pub use streaming::{MOCK_PROVIDER, MockStreamEvent, mock_final, mock_final_with_total_tokens};
pub use tracing_isolation::{
    scoped_tracing_subscriber_guard, scoped_tracing_subscriber_guard_blocking,
};
