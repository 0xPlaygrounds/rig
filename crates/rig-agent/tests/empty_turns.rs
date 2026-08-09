//! An empty assistant turn is legal, and both drivers agree about it.
//!
//! The migration makes "this turn carried nothing" representable, which is
//! what let the fabricated empty-text sentinel go. The shapes below are the
//! ones that sentinel used to hide, and each is asserted on **both** the
//! buffered and streaming drivers — the previous coverage proved the
//! streaming half only, which is how two guards that fire on legitimate turns
//! survived as long as they did.
#![cfg(feature = "test-utils")]
#![allow(clippy::expect_used)]

use futures::StreamExt;
use rig_agent::agent::MultiTurnStreamItem;
use rig_agent::completion::{Prompt, ToolDefinition};
use rig_agent::streaming::StreamingPrompt;
use rig_agent::test_utils::{MockCompletionModel, MockStreamEvent, MockTurn};
use rig_agent::{AgentBuilder, completion::CompletionModel};
use rig_core::completion::AssistantContent;
use serde_json::json;

fn record_tool() -> ToolDefinition {
    ToolDefinition {
        name: "record_value".to_string(),
        description: "Record the supplied integer.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"]
        }),
    }
}

/// A textless tool-call turn: the model answers with a lone function call and
/// no text part. Under the removed container this could not be represented
/// without inventing an empty text block alongside the call.
#[tokio::test]
async fn a_textless_tool_call_turn_carries_no_text_on_either_surface() {
    let model = MockCompletionModel::from_turns_and_stream_turns(
        [MockTurn::tool_call(
            "call-1",
            "record_value",
            json!({"value": 7}),
        )],
        [[
            MockStreamEvent::tool_call("call-2", "record_value", json!({"value": 7})),
            MockStreamEvent::final_response_with_default_usage(),
        ]],
    );

    let request = model
        .completion_request("Record 7.")
        .tool(record_tool())
        .build();
    let buffered = model
        .completion(request.clone())
        .await
        .expect("buffered completion");
    assert!(
        buffered
            .choice
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(_))),
        "expected a tool call, got {:?}",
        buffered.choice
    );
    assert!(
        !buffered
            .choice
            .iter()
            .any(|item| matches!(item, AssistantContent::Text(_))),
        "a textless tool-call turn must not carry a fabricated text part: {:?}",
        buffered.choice
    );

    let mut stream = model.stream(request).await.expect("stream opens");
    let mut streamed_calls = 0usize;
    let mut streamed_text = String::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item") {
            rig_core::streaming::StreamedAssistantContent::ToolCall { .. } => streamed_calls += 1,
            rig_core::streaming::StreamedAssistantContent::Text(text) => {
                streamed_text.push_str(&text.text);
            }
            _ => {}
        }
    }
    assert_eq!(streamed_calls, 1, "streaming surface lost the tool call");
    assert!(
        streamed_text.is_empty(),
        "streaming surface fabricated text: {streamed_text:?}"
    );
}

/// A turn that ends with no content at all after a tool-result round trip —
/// the Anthropic `end_turn` shape. Both drivers must complete the run and
/// neither may record a content-less message in history.
#[tokio::test]
async fn an_empty_terminal_turn_completes_the_run_on_either_surface() {
    let buffered_model = MockCompletionModel::new([
        MockTurn::tool_call("call-1", "record_value", json!({"value": 7})),
        MockTurn::from_contents([]),
    ]);
    let agent = AgentBuilder::new(buffered_model)
        .preamble("Call record_value, then end the turn with no content.")
        .tool(RecordValue)
        .default_max_turns(3)
        .build();
    let response = agent
        .prompt("Record 7.")
        .max_turns(3)
        .extended_details()
        .await
        .expect("an empty terminal turn must not fail the run");
    assert!(
        response.output.trim().is_empty(),
        "expected an empty final output, got {:?}",
        response.output
    );
    let messages = response.messages.expect("extended details carry history");
    assert!(
        !messages.iter().any(is_content_less),
        "history recorded a content-less message: {messages:#?}"
    );

    let streaming_model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("call-2", "record_value", json!({"value": 7})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![MockStreamEvent::final_response_with_default_usage()],
    ]);
    let streaming_agent = AgentBuilder::new(streaming_model)
        .preamble("Call record_value, then end the turn with no content.")
        .tool(RecordValue)
        .default_max_turns(3)
        .build();
    let mut stream = streaming_agent
        .stream_prompt("Record 7.")
        .max_turns(3)
        .await;
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("stream item") {
            final_response = Some(response);
        }
    }
    let streamed = final_response.expect("streaming run produced a final response");
    assert!(
        streamed.output.trim().is_empty(),
        "expected an empty streamed output, got {:?}",
        streamed.output
    );
    if let Some(messages) = streamed.messages.as_deref() {
        assert!(
            !messages.iter().any(is_content_less),
            "streaming history recorded a content-less message: {messages:#?}"
        );
    }
}

fn is_content_less(message: &rig_core::completion::Message) -> bool {
    match message {
        rig_core::completion::Message::User { content } => content.is_empty(),
        rig_core::completion::Message::Assistant { content, .. } => content.is_empty(),
        rig_core::completion::Message::System { .. } => false,
    }
}

#[derive(Clone)]
struct RecordValue;

#[derive(Debug, thiserror::Error)]
#[error("record_value failed")]
struct RecordValueError;

#[derive(serde::Deserialize)]
struct RecordValueArgs {
    value: i64,
}

impl rig_agent::tool::Tool for RecordValue {
    const NAME: &'static str = "record_value";
    type Error = RecordValueError;
    type Args = RecordValueArgs;
    type Output = String;

    fn description(&self) -> String {
        "Record the supplied integer.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        record_tool().parameters
    }

    async fn call(
        &self,
        _context: &mut rig_agent::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!("recorded {}", args.value))
    }
}
