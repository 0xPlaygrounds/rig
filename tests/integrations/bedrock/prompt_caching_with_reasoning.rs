//! Integration test: prompt caching with reasoning blocks in history.
//!
//! Verifies that the CachePoint can be unconditionally appended to the last
//! message because Bedrock requires the last message to be a User message,
//! which never contains reasoning content.

use futures::StreamExt;
use rig::bedrock::completion::CompletionModel;
use rig::client::CompletionClient;
use rig::completion::{AssistantContent, CompletionModel as _, CompletionRequest, Message};
use rig::message::{Text, ToolResult, ToolResultContent, UserContent};
use rig::streaming::StreamedAssistantContent;
use rig::tool::Tool;
use rig::OneOrMany;
use serde_json::json;

use super::{
    anthropic_adaptive_model, client,
    support::{AlphaSignal, ALPHA_SIGNAL_OUTPUT},
};

fn adaptive_thinking_params() -> serde_json::Value {
    json!({ "thinking": { "type": "adaptive" } })
}

fn cached_adaptive_model() -> CompletionModel {
    client()
        .completion_model(anthropic_adaptive_model())
        .with_prompt_caching()
}

/// Two-turn tool roundtrip with adaptive thinking + prompt caching.
///
/// Turn 1: model reasons + calls tool.
/// Turn 2: feed tool result, model responds with text.
///
/// The CachePoint is unconditionally appended to the last message (tool result).
/// This works because the last message is always a User message.
#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock access"]
async fn cache_point_after_tool_result_with_reasoning_in_history() {
    let model = cached_adaptive_model();

    let preamble = "You must call tools when asked. After receiving a tool result, answer with the exact result.";
    let tool_def = AlphaSignal.definition("".to_string()).await;

    // Turn 1: get reasoning + tool call
    let request = model
        .completion_request("Call `lookup_harbor_label` exactly once.")
        .preamble(preamble.to_string())
        .max_tokens(2048)
        .additional_params(adaptive_thinking_params())
        .tools(vec![tool_def.clone()])
        .build();

    let mut stream = model
        .stream(request)
        .await
        .expect("turn 1 stream should start");

    let mut reasoning_content: Option<AssistantContent> = None;
    let mut tool_call_content: Option<rig::message::ToolCall> = None;

    while let Some(item) = stream.next().await {
        match item.expect("turn 1 item") {
            StreamedAssistantContent::Reasoning(r) => {
                reasoning_content = Some(AssistantContent::Reasoning(r));
            }
            StreamedAssistantContent::ToolCall { tool_call, .. } => {
                tool_call_content = Some(tool_call);
            }
            _ => {}
        }
    }

    let tool_call = tool_call_content.expect("model should emit a tool call");
    let reasoning = reasoning_content.expect("model should emit reasoning with adaptive thinking");

    // Turn 2: history with reasoning + tool call, then tool result as last msg
    let mut assistant_content_vec = vec![reasoning];
    assistant_content_vec.push(AssistantContent::ToolCall(tool_call.clone()));

    let chat_history = OneOrMany::many(vec![
        Message::User {
            content: OneOrMany::one(UserContent::Text(Text {
                text: "Call `lookup_harbor_label` exactly once.".to_string(),
                additional_params: None,
            })),
        },
        Message::Assistant {
            id: None,
            content: OneOrMany::many(assistant_content_vec).expect("non-empty"),
        },
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: tool_call.id,
                call_id: None,
                content: OneOrMany::one(ToolResultContent::Text(Text {
                    text: ALPHA_SIGNAL_OUTPUT.to_string(),
                    additional_params: None,
                })),
            })),
        },
    ])
    .expect("non-empty");

    let request2 = CompletionRequest {
        preamble: Some(preamble.to_string()),
        chat_history,
        tools: vec![tool_def],
        temperature: None,
        max_tokens: Some(1024),
        additional_params: Some(adaptive_thinking_params()),
        tool_choice: None,
        model: None,
        documents: vec![],
        output_schema: None,
    };

    let result = model.stream(request2).await;
    match result {
        Ok(mut stream) => {
            let mut text = String::new();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(StreamedAssistantContent::Text(t)) => text.push_str(&t.text),
                    Ok(_) => {}
                    Err(e) => panic!("Turn 2 stream error: {e}"),
                }
            }
            assert!(!text.is_empty(), "Expected text response on turn 2");
        }
        Err(e) => {
            panic!("Turn 2 failed: {e}");
        }
    }
}
