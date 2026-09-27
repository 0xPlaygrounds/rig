//! Test utilities for deterministic completion-model tests.

mod completion;
mod embeddings;
mod memory;
pub mod observations;
mod relay;
mod streaming;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod streaming_conformance;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod streaming_conformance_suite;
mod tracing_isolation;

pub use completion::{MockCompletionModel, MockError, MockScript, MockTurn};
pub use embeddings::{MockEmbeddingModel, MockEmbeddings, MockMultiTextDocument, MockTextDocument};
pub use memory::{AppendFailingMemory, CountingMemory, FailingMemory};
pub use relay::MockRelay;
pub use rig_http::test_utils::{
    CapturedHttpRequest, HttpErrorStreamingClient, MockHttpResponse, MockStreamingClient,
    NonSuccessStreamingClient, RecordingHttpClient, SequencedHttpClient,
    SequencedStreamingHttpClient,
};
#[cfg(test)]
pub(crate) use streaming::scripted_stream;
pub use streaming::{MOCK_PROVIDER, MockStreamEvent, mock_final, mock_final_with_total_tokens};
pub use tracing_isolation::{
    scoped_tracing_subscriber_guard, scoped_tracing_subscriber_guard_blocking,
};

/// The fold a call to `wire` in `mode` opens for `request`, for tests that
/// drive a decoder by hand.
#[cfg(test)]
pub(crate) fn fold_for<W: crate::wire::Wire>(
    request: &crate::wire::Request<W>,
    wire: &W,
    mode: crate::wire::Mode,
) -> <W::Op as crate::wire::Operation>::Fold {
    <W::Op as crate::wire::Operation>::fold(
        request,
        &mut crate::wire::Call::new(&wire.describe(), mode),
    )
}
