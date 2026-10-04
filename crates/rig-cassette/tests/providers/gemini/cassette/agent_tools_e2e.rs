//! End-to-end tool execution through the built-in agent drivers: real
//! `ToolSet` execution behind `agent.prompt()` / `agent.chat()` /
//! `agent.prompt()`, pinning the wire contract of the handrolled tool
//! pipeline ahead of the rmcp migration.

use rig::providers::gemini;
use rig_agent::test_utils::tool_output_serialization;

use super::super::support::with_gemini_cassette;

#[tokio::test]
async fn string_output_sent_verbatim_and_struct_output_serialized_as_json() {
    with_gemini_cassette(
        "agent_tools/string_output_verbatim_struct_output_json",
        |client| async move {
            let report = tool_output_serialization(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                |builder| builder,
            )
            .await
            .expect("tool-output serialization conformance scenario should succeed");
            eprintln!("[gemini] {report:?}");
        },
    )
    .await;
}
