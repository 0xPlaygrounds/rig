//! Test utilities for the classic runtime and its provider-facing acceptance tests.

mod model_conformance;
mod tools;

pub use model_conformance::{
    ConformanceToolError, ScenarioError, ScenarioOverrides, ScenarioReport,
    buffered_streaming_text_parity, cancellation_and_max_turns, cancellation_and_max_turns_session,
    complex_tool_arguments, complex_tool_arguments_session, decode_structured_output,
    hook_rewrites_and_request_patch, hook_rewrites_and_request_patch_session,
    invalid_tool_recovery, invalid_tool_recovery_session, optional_argument,
    optional_argument_session, parallel_tools, parallel_tools_session, sequential_tools,
    sequential_tools_session, streaming_structured_after_tool,
    streaming_structured_after_tool_session, streaming_tool, streaming_tool_session,
    structured_after_tool, structured_after_tool_session, structured_extraction,
    structured_extraction_session, tool_choice_modes, tool_output_serialization,
    tool_output_serialization_session, validate_cancelled_failure, validate_extraction_fields,
    validate_max_turns_failure, validate_protocol_hygiene, validate_result_redaction,
    validate_rewritten_arguments, validate_unknown_tool_failure, zero_argument_tool,
    zero_argument_tool_session,
};
pub use rig_core::test_utils::*;
pub use rig_memory::test_utils::{AppendFailingMemory, CountingMemory, FailingMemory};
pub use tools::{
    MockAddTool, MockBarrierTool, MockContextProbeTool, MockControlledTool, MockDeniedTool,
    MockExampleTool, MockFailingTool, MockFailure, MockHandledFailureTool, MockImageGeneratorTool,
    MockImageOutputTool, MockMetadataTool, MockObjectOutputTool, MockOperationArgs, MockRequestId,
    MockStringOutputTool, MockSubtractTool, MockToolError, SessionId, mock_math_toolset,
};
