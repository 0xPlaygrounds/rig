//! Cassette-backed OpenRouter long-session and tool-contract regression tests.
//!
//! These scenarios stress OpenRouter's OpenAI-compatible chat-completions path
//! with multi-turn tool loops, streamed tool-call deltas, complex JSON tool
//! arguments, explicit tool choice, long caller-owned chat history, and usage
//! surfaced from routed upstream providers.

use anyhow::Result;
use rig::message::ToolChoice;
use rig::tool::Tool;

use crate::matrix::agent_tool_sessions::{ComplexSession, InspectManifest};
use crate::support::{
    assert_raw_stream_tool_call_arguments_are_objects, collect_raw_stream_observation,
};

use super::super::{TOOL_MODEL, support::with_openrouter_cassette_result};
use rig::completion::CompletionRequest;

pub(super) const SESSION_MODEL: &str = TOOL_MODEL;
const SESSION_MAX_TOKENS: Option<u64> = None;

const COMPLEX_SESSION: ComplexSession = ComplexSession {
    preamble: "\
You are a deterministic OpenRouter tool orchestration test harness. Use the tools instead of inventing values. \
For the production-readiness scenario, call exactly one tool at a time in this order: \
1. ping_empty with an empty JSON object. \
2. inspect_manifest with project rig-openrouter, flags critical=true and retries=2, steps plan weight=1 and verify weight=2, and the exact note from the user. \
3. join_labels with labels [north, beta gamma, quote:\"delta\", slash\\path] and separator |. \
4. escape_echo with the exact escaped text from the user. \
After all tool results are available, answer in one short sentence that includes EMPTY-OK, MANIFEST-OK, LABELS-OK, and ESCAPE-OK.",
    prompt: "\
Run the production-readiness scenario. The manifest note is `line one; line two says \"hello\" and path C:\\rig\\openrouter`. \
The escaped text is `Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃`.",
    project: "rig-openrouter",
    note: "line one; line two says \"hello\" and path C:\\rig\\openrouter",
    escaped_text: "Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃",
    optional_probe: None,
};

crate::matrix::case_matrix! {
    wrapper: with_openrouter_cassette_result, family: agent_tool_sessions_case;
    # [tokio :: test]
    sequential_complex_tool_calls_nonstreaming: ("agent_tool_sessions/sequential_complex_tool_calls_nonstreaming", sequential_complex_tool_calls_nonstreaming);
    # [tokio :: test]
    sequential_complex_tool_calls_streaming: ("agent_tool_sessions/sequential_complex_tool_calls_streaming", sequential_complex_tool_calls_streaming);
}

#[tokio::test]
async fn raw_stream_complex_tool_call_deltas_have_object_arguments() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/raw_stream_complex_tool_call_deltas_have_object_arguments",
        |client| async move {
            let model = client.completion(SESSION_MODEL);
            let tool = InspectManifest::default();
            let request = CompletionRequest::new(
                    "Call inspect_manifest exactly once for project rig-openrouter with critical=true, retries=2, \
                     steps [{name: plan, weight: 1}, {name: verify, weight: 2}], and note `streamed nested JSON`. \
                     Do not write normal text before the tool call.",
                )
                .preamble("Use the requested tool call and no prose before it.")
                .tool(rig::tool::tool_definition(&tool))
                .tool_choice(ToolChoice::Required);

            let observation = collect_raw_stream_observation(model.stream(request)?).await;

            assert_raw_stream_tool_call_arguments_are_objects(
                &observation,
                &[InspectManifest::NAME],
            );
            let tool_call = observation
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == InspectManifest::NAME)
                .ok_or_else(|| anyhow::anyhow!("raw stream should emit inspect_manifest"))?;
            anyhow::ensure!(tool_call.function.arguments["project"] == "rig-openrouter");
            anyhow::ensure!(tool_call.function.arguments["flags"]["critical"] == true);
            anyhow::ensure!(
                tool_call.function.arguments["steps"].as_array().map(Vec::len) == Some(2)
            );

            Ok(())
        },
    )
    .await
}
