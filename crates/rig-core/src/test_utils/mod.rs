//! Test utilities for deterministic completion-model tests.

mod completion;
mod embeddings;
mod http;
#[cfg(test)]
pub(crate) mod internal_streaming_profiles;
mod memory;
mod model_conformance;
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
pub use model_conformance::{
    ConformanceToolError, ScenarioError, ScenarioReport, buffered_streaming_text_parity,
    cancellation_and_max_turns, complex_tool_arguments, decode_structured_output,
    hook_rewrites_and_request_patch, invalid_tool_recovery, optional_argument, parallel_tools,
    sequential_tools, streaming_structured_after_tool, streaming_tool, structured_after_tool,
    structured_extraction, tool_choice_modes, tool_output_serialization,
    validate_cancelled_failure, validate_extraction_fields, validate_max_turns_failure,
    validate_protocol_hygiene, validate_result_redaction, validate_rewritten_arguments,
    validate_unknown_tool_failure, zero_argument_tool,
};
pub use model_listing::MockModelLister;
pub use streaming::{MockResponse, MockStreamEvent};
pub use tools::{
    BarrierMockToolIndex, MockAddTool, MockBarrierTool, MockContextProbeTool, MockControlledTool,
    MockDeniedTool, MockExampleTool, MockFailingTool, MockFailure, MockHandledFailureTool,
    MockImageGeneratorTool, MockImageOutputTool, MockMetadataTool, MockObjectOutputTool,
    MockOperationArgs, MockRequestId, MockStringOutputTool, MockSubtractTool, MockToolError,
    MockToolIndex, SessionId, mock_math_toolset,
};
pub use tracing_isolation::{
    scoped_tracing_subscriber_guard, scoped_tracing_subscriber_guard_blocking,
};
