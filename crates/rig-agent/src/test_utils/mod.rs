//! Test utilities for the classic runtime and its provider-facing acceptance tests.

mod model_conformance;
mod tools;

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
pub use rig_core::test_utils::*;
pub use tools::{
    BarrierMockToolIndex, MockAddTool, MockBarrierTool, MockContextProbeTool, MockControlledTool,
    MockDeniedTool, MockExampleTool, MockFailingTool, MockFailure, MockHandledFailureTool,
    MockImageGeneratorTool, MockImageOutputTool, MockMetadataTool, MockObjectOutputTool,
    MockOperationArgs, MockRequestId, MockStringOutputTool, MockSubtractTool, MockToolError,
    MockToolIndex, SessionId, mock_math_toolset,
};
