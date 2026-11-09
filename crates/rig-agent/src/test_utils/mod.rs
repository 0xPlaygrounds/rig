//! Test utilities for the classic runtime and its provider-facing acceptance tests.

mod model_conformance;
mod tools;

pub use model_conformance::{
    COMPLEX_ARGUMENTS_RAW_PROMPT, ConformanceToolError, ScenarioError, ScenarioReport,
    buffered_streaming_text_parity, cancellation_and_max_turns, complex_tool_arguments,
    complex_tool_arguments_with_prompt, decode_structured_output, hook_rewrites_and_request_patch,
    hook_rewrites_and_request_patch_with_choice, invalid_tool_recovery,
    invalid_tool_recovery_with_choice, optional_argument, parallel_tools, sequential_tools,
    streaming_structured_after_tool, streaming_tool, structured_after_tool, structured_extraction,
    tool_choice_modes, tool_output_serialization, validate_cancelled_failure,
    validate_extraction_fields, validate_max_turns_failure, validate_protocol_hygiene,
    validate_rewritten_arguments, validate_unknown_tool_failure, zero_argument_tool,
};
pub use rig_core::test_utils::*;

/// `builder` without a sampling temperature. The model-contract scenarios
/// sample at temperature 0; a `configure` hook passes their builder through
/// this for a model that rejects `temperature`.
pub fn without_temperature(builder: crate::agent::AgentBuilder) -> crate::agent::AgentBuilder {
    builder.clear_temperature()
}
#[cfg(test)]
pub(crate) use tools::Counted;
pub use tools::{
    MockAddTool, MockContextProbeTool, MockDeniedTool, MockExampleTool, MockFailingTool,
    MockFailure, MockHandledFailureTool, MockImageGeneratorTool, MockMetadataTool,
    MockObjectOutputTool, MockOperationArgs, MockRequestId, MockSubtractTool, MockToolError,
    MockToolIndex, SessionId, mock_math_toolset,
};

/// Drive a stream to completion, panicking on any stream error, and return
/// its final response.
#[cfg(test)]
pub(crate) async fn drive_to_final_response(
    mut stream: impl futures::Stream<
        Item = Result<crate::agent::MultiTurnStreamItem, crate::completion::PromptError>,
    > + Unpin,
) -> crate::agent::PromptResponse {
    use futures::StreamExt;
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let crate::agent::MultiTurnStreamItem::FinalResponse(response) =
            item.unwrap_or_else(|err| panic!("stream item errored: {err}"))
        {
            final_response = Some(response);
        }
    }
    final_response.expect("stream should yield a final response")
}
