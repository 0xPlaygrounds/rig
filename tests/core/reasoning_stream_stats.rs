use futures::stream;
use rig::agent::MultiTurnStreamItem;
use rig::completion::Usage;
use rig::message::{AssistantContent, ToolCall, ToolFunction, ToolResult, ToolResultContent};
use rig::streaming::{Item, Transcript};

use crate::reasoning::collect_stream_stats;

/// A text fragment, cut from a stream that opened its part.
fn text(text: &str) -> Item<rig::streaming::StreamEvent> {
    Transcript::parse_prefix(serde_json::json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        {"item": "event", "value": {"event": "text", "part": 0, "text": text}},
    ]))
    .expect("a stream in order")
    .into_items()
    .pop()
    .expect("the fragment")
}

#[tokio::test]
async fn collect_stream_stats_tracks_only_final_turn_text() {
    let tool_call = ToolCall::from_wire(
        "tool_1",
        ToolFunction::new(
            rig::message::ToolName::new("get_weather").expect("tool name"),
            serde_json::json!({ "city": "Tokyo" }),
        ),
    );
    let tool_result = ToolResult {
        is_error: false,
        call: tool_call.id.clone(),
        name: tool_call.function.name.clone(),
        content: vec![ToolResultContent::text("72F and sunny")],
    };

    let items = vec![
        Ok(MultiTurnStreamItem::StreamAssistantItem(text(
            "Sure! Let me check the weather right away!",
        ))),
        Ok(MultiTurnStreamItem::ToolCall { tool_call }),
        Ok(MultiTurnStreamItem::ToolResult { tool_result }),
        Ok(MultiTurnStreamItem::StreamAssistantItem(text(
            "It's 72F and sunny in Tokyo.",
        ))),
        Ok(MultiTurnStreamItem::final_response(
            vec![AssistantContent::text("It's 72F and sunny in Tokyo.")],
            Usage::default(),
        )),
    ];

    let stats = collect_stream_stats(stream::iter(items), "test").await;

    assert_eq!(stats.tool_calls_in_stream, vec!["get_weather".to_string()]);
    assert_eq!(stats.tool_results_in_stream, 1);
    assert!(stats.got_final_response, "expected final response event");
    assert_eq!(
        stats.final_turn_text, "It's 72F and sunny in Tokyo.",
        "pre-tool assistant text should not be counted as final-turn text"
    );
    assert_eq!(
        stats.final_response_text.as_deref(),
        Some(stats.final_turn_text.as_str()),
        "final response text should match the final turn's streamed text"
    );
}
