#[cfg(feature = "openai")]
use crate::message::AssistantContent;
#[cfg(feature = "openai")]
use crate::streaming::{self, BlockClose, BlockKind, Delta, StreamEvent};
use bytes::Bytes;
#[cfg(feature = "openai")]
use futures::StreamExt;

pub(crate) fn sse_bytes_from_data_lines<T>(events: impl IntoIterator<Item = T>) -> Bytes
where
    T: AsRef<str>,
{
    Bytes::from(
        events
            .into_iter()
            .map(|event| format!("data: {}\n\n", event.as_ref()))
            .collect::<String>(),
    )
}

#[cfg(any(feature = "openai", feature = "copilot"))]
pub(crate) fn sse_bytes_from_json_events(events: &[serde_json::Value]) -> Bytes {
    Bytes::from(
        events
            .iter()
            .map(|event| {
                format!(
                    "data: {}\n\n",
                    serde_json::to_string(event).expect("event should serialize")
                )
            })
            .collect::<String>(),
    )
}

#[cfg(feature = "openai")]
pub(crate) async fn assert_zero_arg_tool_call_is_emitted(
    mut stream: streaming::StreamingCompletionResponse,
    expected_id: &str,
    expected_name: &str,
    expect_final_response: bool,
) {
    let mut saw_final = false;
    let mut collected_tool_calls = Vec::new();

    while let Some(chunk) = stream.next().await {
        match chunk.expect("stream item should be ok") {
            StreamEvent::BlockStart {
                kind: BlockKind::ToolCall,
                ..
            }
            | StreamEvent::BlockDelta {
                delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
                ..
            } => {}
            StreamEvent::Final(_) => saw_final = true,
            StreamEvent::BlockEnd {
                end: BlockClose::ToolCall(_),
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            } => {
                collected_tool_calls.push(tool_call);
            }
            _ => panic!("unexpected stream item while asserting zero-arg tool call"),
        }
    }

    if expect_final_response {
        assert!(saw_final, "stream should still yield a final response");
    } else {
        assert!(
            !saw_final,
            "a truncated stream must not synthesize a terminal record"
        );
    }

    assert_eq!(collected_tool_calls.len(), 1);
    assert_eq!(collected_tool_calls[0].id.explicit(), Some(expected_id));
    assert_eq!(collected_tool_calls[0].function.name, expected_name);
    assert_eq!(
        collected_tool_calls[0].function.arguments,
        serde_json::json!({})
    );
}
