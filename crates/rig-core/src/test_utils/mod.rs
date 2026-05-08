//! Test utilities for deterministic completion-model tests.

mod completion;
mod embeddings;
mod http;
#[cfg(test)]
pub(crate) mod internal_streaming_profiles;
mod memory;
mod model_listing;
mod pipeline;
mod streaming;
mod tools;

pub use completion::{MockCompletionModel, MockError, MockTurn};
pub use embeddings::{MockEmbeddingModel, MockMultiTextDocument, MockTextDocument};
pub use http::{
    CapturedHttpRequest, MockHttpResponse, MockStreamingClient, RecordingHttpClient,
    SequencedStreamingHttpClient,
};
pub use memory::{AppendFailingMemory, CountingMemory, FailingMemory};
pub use model_listing::MockModelLister;
pub use pipeline::{Foo, MockPromptModel, MockVectorStoreIndex};
pub use streaming::{MockResponse, MockStreamEvent};
pub use tools::{
    BarrierMockToolIndex, MockAddTool, MockBarrierTool, MockControlledTool, MockExampleTool,
    MockImageGeneratorTool, MockImageOutputTool, MockObjectOutputTool, MockOperationArgs,
    MockStringOutputTool, MockSubtractTool, MockToolError, MockToolIndex, mock_math_toolset,
};
