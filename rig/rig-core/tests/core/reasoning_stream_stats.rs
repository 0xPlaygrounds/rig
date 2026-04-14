use futures::stream;
use rig::OneOrMany;
use rig::agent::MultiTurnStreamItem;
use rig::completion::Usage;
use rig::message::{ToolCall, ToolFunction, ToolResult, ToolResultContent};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent};

use crate::reasoning::collect_stream_stats;

#[tokio::test]
async fn collect_stream_stats_tracks_only_final_turn_text() {
    let internal_call_id = "call_internal_1".to_string();
    let tool_call = ToolCall::new(
        "tool_1".to_string(),
        ToolFunction::new(
            "get_weather".to_string(),
            serde_json::json!({ "city": "Tokyo" }),
        ),
    );
    let tool_result = ToolResult {
        id: "tool_1".to_string(),
        call_id: None,
        content: OneOrMany::one(ToolResultContent::text("72F and sunny")),
    };

    let items = vec![
        Ok(MultiTurnStreamItem::StreamAssistantItem(
            StreamedAssistantContent::<()>::text("Sure! Let me check the weather right away!"),
        )),
        Ok(MultiTurnStreamItem::StreamAssistantItem(
            StreamedAssistantContent::ToolCall {
                tool_call,
                internal_call_id: internal_call_id.clone(),
            },
        )),
        Ok(MultiTurnStreamItem::StreamUserItem(
            StreamedUserContent::tool_result(tool_result, internal_call_id),
        )),
        Ok(MultiTurnStreamItem::StreamAssistantItem(
            StreamedAssistantContent::<()>::text("It's 72F and sunny in Tokyo."),
        )),
        Ok(MultiTurnStreamItem::final_response(
            "It's 72F and sunny in Tokyo.",
            Usage::new(),
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
