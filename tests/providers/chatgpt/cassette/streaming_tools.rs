//! ChatGPT cassette coverage for terminal responses that omit `output`.

use futures::StreamExt;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::ToolChoice;
use rig::providers::chatgpt;
use rig::streaming::StreamedAssistantContent;
use serde_json::json;

use super::super::support::with_chatgpt_cassette;
use crate::support::zero_arg_tool_definition;

#[tokio::test]
async fn stream_tool_call_completed_response_without_output() {
    with_chatgpt_cassette(
        "streaming_tools/tool_call_completed_response_without_output",
        |client| async move {
            let model = client.completion_model(chatgpt::GPT_5_3_CODEX);
            let request = model
                .completion_request(
                    "Call the ping tool with no arguments. Do not write any normal text before the tool call.",
                )
                .tool(zero_arg_tool_definition("ping"))
                .tool_choice(ToolChoice::Required)
                .build();

            let mut stream = model.stream(request).await.expect("stream should start");
            let mut saw_ping_tool_call = false;
            let mut final_usage = None;

            while let Some(chunk) = stream.next().await {
                match chunk.expect("stream item should be ok") {
                    StreamedAssistantContent::ToolCall { tool_call, .. } => {
                        if tool_call.function.name == "ping" {
                            assert_eq!(tool_call.function.arguments, json!({}));
                            saw_ping_tool_call = true;
                        }
                    }
                    StreamedAssistantContent::Final(response) => {
                        final_usage = Some(response.usage);
                    }
                    _ => {}
                }
            }

            assert!(saw_ping_tool_call, "stream should emit the ping tool call");
            let usage = final_usage.expect("stream should emit terminal usage");
            assert!(usage.input_tokens > 0, "usae should have input tokens");
            assert!(usage.output_tokens > 0, "usae should have output tokens");
        },
    )
    .await;
}
