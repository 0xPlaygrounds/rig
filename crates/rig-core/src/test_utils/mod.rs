//! Test utilities for deterministic completion-model tests.

mod completion;
mod embeddings;
mod http;
#[cfg(test)]
pub(crate) mod internal_streaming_profiles;
mod memory;
mod model_listing;
mod streaming;
mod tools;
mod tracing_isolation;

pub use completion::{MockCompletionModel, MockError, MockTurn};
pub use embeddings::{MockEmbeddingModel, MockMultiTextDocument, MockTextDocument};
pub use http::{
    CapturedHttpRequest, HttpErrorStreamingClient, MockHttpResponse, MockStreamingClient,
    RecordingHttpClient, SequencedStreamingHttpClient,
};
pub use memory::{AppendFailingMemory, CountingMemory, FailingMemory};
pub use model_listing::MockModelLister;
pub use streaming::{MockResponse, MockStreamEvent};
pub use tools::{
    BarrierMockToolIndex, MockAddTool, MockBarrierTool, MockControlledTool, MockExampleTool,
    MockImageGeneratorTool, MockImageOutputTool, MockObjectOutputTool, MockOperationArgs,
    MockStringOutputTool, MockSubtractTool, MockToolError, MockToolIndex, mock_math_toolset,
};
pub use tracing_isolation::{
    scoped_tracing_subscriber_guard, scoped_tracing_subscriber_guard_blocking,
};
