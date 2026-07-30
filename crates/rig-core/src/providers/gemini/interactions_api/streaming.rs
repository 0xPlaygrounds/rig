use futures::Stream;
use std::pin::Pin;

use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, FunctionCallDelta, InteractionSseEvent, Step,
    TextDelta, ThoughtSummaryContent, ThoughtSummaryDelta,
};
use crate::completion::CompletionError;
use crate::streaming;
use serde_json::{Map, Value};

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>> + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>>>>;

/// Map an Interactions `step.start` payload onto a raw streaming choice. Pure.
pub fn step_start_to_choice(step: Step) -> Option<streaming::RawStreamingChoice> {
    match step {
        Step::ModelOutput { content } => content.into_iter().find_map(content_to_choice),
        Step::FunctionCall(FunctionCallContent {
            name,
            arguments,
            id,
        }) => {
            let name = name?;
            let call_id = id.unwrap_or_else(|| name.clone());
            Some(streaming::RawStreamingChoice::ToolCall(
                streaming::RawStreamingToolCall::new(
                    name.clone(),
                    name,
                    arguments.unwrap_or(Value::Object(Map::new())),
                )
                .with_call_id(call_id),
            ))
        }
        _ => None,
    }
}

/// Map one Interactions output content block onto a raw streaming choice. Pure.
pub fn content_to_choice(content: Content) -> Option<streaming::RawStreamingChoice> {
    match content {
        Content::Text(text) if !text.text.is_empty() => {
            Some(streaming::RawStreamingChoice::Message(text.text))
        }
        Content::FunctionCall(content) => step_start_to_choice(Step::FunctionCall(content)),
        _ => None,
    }
}

/// Map an Interactions `step.delta` payload onto a raw streaming choice. Pure.
pub fn content_delta_to_choice(delta: ContentDelta) -> Option<streaming::RawStreamingChoice> {
    match delta {
        ContentDelta::Text(TextDelta {
            text: Some(text), ..
        }) => Some(streaming::RawStreamingChoice::Message(text)),
        ContentDelta::FunctionCall(FunctionCallDelta {
            name,
            arguments,
            id,
        }) => {
            let name = name?;
            let call_id = id.unwrap_or_else(|| name.clone());
            Some(streaming::RawStreamingChoice::ToolCall(
                streaming::RawStreamingToolCall::new(
                    name.clone(),
                    name,
                    arguments.unwrap_or(Value::Object(Map::new())),
                )
                .with_call_id(call_id),
            ))
        }
        ContentDelta::ThoughtSummary(ThoughtSummaryDelta { content }) => {
            let text = match content {
                ThoughtSummaryContent::Text(text) => text.text,
                _ => return None,
            };
            Some(streaming::RawStreamingChoice::ReasoningDelta {
                id: None,
                reasoning: text,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_stream_final_has_model_version() {
        let response = streaming::StreamFinal::new("gemini", crate::completion::Usage::default())
            .with_model("gemini-2.5-pro-preview-05-06");

        assert_eq!(
            response.model.as_deref(),
            Some("gemini-2.5-pro-preview-05-06")
        );

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: streaming::StreamFinal = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.model.as_deref(),
            Some("gemini-2.5-pro-preview-05-06")
        );
    }

    #[test]
    fn test_content_delta_text_event() {
        let event_json = json!({
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "type": "text",
                "text": "Hello"
            }
        });

        let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
        let InteractionSseEvent::StepDelta { delta, .. } = event else {
            panic!("expected step delta");
        };

        let choice = content_delta_to_choice(delta).expect("choice should exist");
        match choice {
            crate::streaming::RawStreamingChoice::Message(text) => {
                assert_eq!(text, "Hello");
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }

    #[test]
    fn test_content_delta_function_call_event() {
        let event_json = json!({
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "type": "function_call",
                "name": "get_weather",
                "arguments": {"location": "Paris"},
                "id": "call-1"
            }
        });

        let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
        let InteractionSseEvent::StepDelta { delta, .. } = event else {
            panic!("expected step delta");
        };

        let choice = content_delta_to_choice(delta).expect("choice should exist");
        match choice {
            crate::streaming::RawStreamingChoice::ToolCall(call) => {
                assert_eq!(call.name, "get_weather");
                assert_eq!(call.call_id.as_deref(), Some("call-1"));
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }
}
