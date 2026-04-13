use async_stream::stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use super::InteractionsCompletionModel;
use super::create_request_body;
use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, FunctionCallDelta, Interaction,
    InteractionSseEvent, InteractionUsage, TextContent, TextDelta, ThoughtContent,
    ThoughtSignatureDelta, ThoughtSummaryContent, ThoughtSummaryDelta,
};
use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::Request;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::message::ReasoningContent;
use crate::streaming;
use crate::telemetry::SpanCombinator;
use serde_json::{Map, Value};

#[derive(Debug, Default)]
struct ThoughtState {
    text: String,
    signature: Option<String>,
}

/// Final metadata yielded by an Interactions streaming response.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StreamingCompletionResponse {
    pub usage: Option<InteractionUsage>,
    pub interaction: Option<Interaction>,
}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>> + Send>>;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>>>>;

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        self.usage.as_ref().and_then(|usage| usage.token_usage())
    }
}

impl<T> InteractionsCompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "interactions_streaming",
                gen_ai.operation.name = "interactions_streaming",
                gen_ai.provider.name = "gcp.gemini",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request = create_request_body(self.model.clone(), completion_request, Some(true))?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::streaming",
                "Gemini interactions streaming request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post_sse("/v1beta/interactions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let mut event_source = GenericEventSource::new(self.client.clone(), req);

        let stream = stream! {
            let mut final_interaction: Option<Interaction> = None;
            let mut final_usage: Option<InteractionUsage> = None;
            let mut thought_states: HashMap<i32, ThoughtState> = HashMap::new();

            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::debug!("SSE connection opened");
                        continue;
                    }
                    Ok(Event::Message(message)) => {
                        if message.data.trim().is_empty() {
                            continue;
                        }

                        let data = match serde_json::from_str::<InteractionSseEvent>(&message.data)
                        {
                            Ok(data) => data,
                            Err(err) => {
                                tracing::debug!(
                                    "Failed to deserialize interactions SSE event: {err}"
                                );
                                continue;
                            }
                        };

                        match data {
                            InteractionSseEvent::ContentDelta { index, delta, .. } => {
                                if let Some(choice) =
                                    content_delta_to_choice(index, delta, &mut thought_states)
                                {
                                    yield Ok(choice);
                                }
                            }
                            InteractionSseEvent::ContentStart { index, content, .. } => {
                                for choice in content_start_to_choices(
                                    index,
                                    content,
                                    &mut thought_states,
                                ) {
                                    yield Ok(choice);
                                }
                            }
                            InteractionSseEvent::ContentStop { index, .. } => {
                                if let Some(choice) =
                                    content_stop_to_choice(index, &mut thought_states)
                                {
                                    yield Ok(choice);
                                }
                            }
                            InteractionSseEvent::InteractionComplete { interaction, .. } => {
                                let span = tracing::Span::current();
                                span.record("gen_ai.response.id", &interaction.id);
                                if let Some(model) = interaction.model.clone() {
                                    span.record("gen_ai.response.model", model);
                                }

                                if let Some(usage) = interaction.usage.clone() {
                                    span.record_token_usage(&usage);
                                    final_usage = Some(usage);
                                }
                                final_interaction = Some(interaction);
                            }
                            InteractionSseEvent::Error { error, .. } => {
                                yield Err(CompletionError::ProviderError(error.message));
                                break;
                            }
                            _ => continue,
                        }
                    }
                    Err(crate::http_client::Error::StreamEnded) => {
                        break;
                    }
                    Err(error) => {
                        tracing::error!(?error, "SSE error");
                        yield Err(CompletionError::ProviderError(error.to_string()));
                        break;
                    }
                }
            }

            event_source.close();

            yield Ok(streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                usage: final_usage.or_else(|| final_interaction.as_ref().and_then(|i| i.usage.clone())),
                interaction: final_interaction,
            }));
        }
        .instrument(span);

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
    }
}

pub(crate) fn stream_interaction_events<T>(
    client: super::InteractionsClient<T>,
    request: Request<Vec<u8>>,
) -> InteractionEventStream
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + 'static,
{
    let mut event_source = GenericEventSource::new(client.clone(), request);

    let stream = stream! {
        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => continue,
                Ok(Event::Message(message)) => {
                    if message.data.trim().is_empty() {
                        continue;
                    }

                    let data = serde_json::from_str::<InteractionSseEvent>(&message.data);
                    let Ok(data) = data else {
                        let err = data.unwrap_err();
                        tracing::debug!("Failed to deserialize interactions SSE event: {err}");
                        continue;
                    };

                    yield Ok(data);
                }
                Err(crate::http_client::Error::StreamEnded) => break,
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    yield Err(CompletionError::ProviderError(error.to_string()));
                    break;
                }
            }
        }

        event_source.close();
    };

    Box::pin(stream)
}

fn extend_thought_state_with_summary(
    thought_state: &mut ThoughtState,
    summary: impl IntoIterator<Item = ThoughtSummaryContent>,
) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
    summary
        .into_iter()
        .filter_map(|content| match content {
            ThoughtSummaryContent::Text(text) if !text.text.is_empty() => Some(text.text),
            _ => None,
        })
        .map(|text| {
            thought_state.text.push_str(&text);
            streaming::RawStreamingChoice::ReasoningDelta {
                id: None,
                reasoning: text,
            }
        })
        .collect()
}

fn content_start_to_choices(
    index: i32,
    content: Content,
    thought_states: &mut HashMap<i32, ThoughtState>,
) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
    match content {
        Content::Text(TextContent { text, .. }) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![streaming::RawStreamingChoice::Message(text)]
            }
        }
        Content::FunctionCall(FunctionCallContent {
            name,
            arguments,
            id,
        }) => {
            let Some(name) = name else {
                return Vec::new();
            };
            let call_id = id.unwrap_or_else(|| name.clone());
            vec![streaming::RawStreamingChoice::ToolCall(
                streaming::RawStreamingToolCall::new(
                    name.clone(),
                    name,
                    arguments.unwrap_or(Value::Object(Map::new())),
                )
                .with_call_id(call_id),
            )]
        }
        Content::Thought(ThoughtContent { summary, signature }) => {
            let mut thought_state = ThoughtState {
                text: String::new(),
                signature,
            };
            let choices =
                extend_thought_state_with_summary(&mut thought_state, summary.unwrap_or_default());
            thought_states.insert(index, thought_state);
            choices
        }
        _ => Vec::new(),
    }
}

fn content_delta_to_choice(
    index: i32,
    delta: ContentDelta,
    thought_states: &mut HashMap<i32, ThoughtState>,
) -> Option<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
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
            let thought_state = thought_states.entry(index).or_default();
            extend_thought_state_with_summary(thought_state, [content])
                .into_iter()
                .next()
        }
        ContentDelta::ThoughtSignature(ThoughtSignatureDelta { signature }) => {
            let thought_state = thought_states.entry(index).or_default();
            thought_state
                .signature
                .get_or_insert_with(String::new)
                .push_str(&signature);
            None
        }
        _ => None,
    }
}

fn content_stop_to_choice(
    index: i32,
    thought_states: &mut HashMap<i32, ThoughtState>,
) -> Option<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
    let thought_state = thought_states.remove(&index)?;
    if thought_state.text.is_empty() && thought_state.signature.is_none() {
        return None;
    }

    Some(streaming::RawStreamingChoice::Reasoning {
        id: None,
        content: ReasoningContent::Text {
            text: thought_state.text,
            signature: thought_state.signature,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::RawStreamingChoice;
    use serde_json::json;

    #[test]
    fn test_content_delta_text_event() {
        let event_json = json!({
            "event_type": "content.delta",
            "index": 0,
            "delta": {
                "type": "text",
                "text": "Hello"
            }
        });

        let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
        let InteractionSseEvent::ContentDelta { delta, .. } = event else {
            panic!("expected content delta");
        };

        let choice =
            content_delta_to_choice(0, delta, &mut HashMap::new()).expect("choice should exist");
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
            "event_type": "content.delta",
            "index": 0,
            "delta": {
                "type": "function_call",
                "name": "get_weather",
                "arguments": {"location": "Paris"},
                "id": "call-1"
            }
        });

        let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
        let InteractionSseEvent::ContentDelta { delta, .. } = event else {
            panic!("expected content delta");
        };

        let choice =
            content_delta_to_choice(0, delta, &mut HashMap::new()).expect("choice should exist");
        match choice {
            crate::streaming::RawStreamingChoice::ToolCall(call) => {
                assert_eq!(call.name, "get_weather");
                assert_eq!(call.call_id.as_deref(), Some("call-1"));
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }

    #[test]
    fn test_thought_summary_and_signature_emit_final_signed_reasoning() {
        let mut thought_states = HashMap::new();

        let start_choices = content_start_to_choices(
            0,
            Content::Thought(ThoughtContent {
                summary: None,
                signature: None,
            }),
            &mut thought_states,
        );
        assert!(start_choices.is_empty());

        let summary_choice = content_delta_to_choice(
            0,
            ContentDelta::ThoughtSummary(ThoughtSummaryDelta {
                content: ThoughtSummaryContent::Text(TextContent {
                    text: "thinking...".to_string(),
                    annotations: None,
                }),
            }),
            &mut thought_states,
        )
        .expect("summary delta should yield reasoning");
        match summary_choice {
            RawStreamingChoice::ReasoningDelta { id, reasoning } => {
                assert_eq!(id, None);
                assert_eq!(reasoning, "thinking...");
            }
            other => panic!("unexpected choice: {other:?}"),
        }

        assert!(
            content_delta_to_choice(
                0,
                ContentDelta::ThoughtSignature(ThoughtSignatureDelta {
                    signature: "sig-123".to_string(),
                }),
                &mut thought_states,
            )
            .is_none()
        );

        let stop_choice =
            content_stop_to_choice(0, &mut thought_states).expect("stop should emit reasoning");
        match stop_choice {
            RawStreamingChoice::Reasoning { id, content } => {
                assert_eq!(id, None);
                match content {
                    ReasoningContent::Text { text, signature } => {
                        assert_eq!(text, "thinking...");
                        assert_eq!(signature.as_deref(), Some("sig-123"));
                    }
                    other => panic!("unexpected reasoning content: {other:?}"),
                }
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }

    #[test]
    fn test_signature_only_thought_emits_final_signed_reasoning() {
        let mut thought_states = HashMap::new();

        assert!(
            content_delta_to_choice(
                0,
                ContentDelta::ThoughtSignature(ThoughtSignatureDelta {
                    signature: "sig-only".to_string(),
                }),
                &mut thought_states,
            )
            .is_none()
        );

        let stop_choice =
            content_stop_to_choice(0, &mut thought_states).expect("stop should emit reasoning");
        match stop_choice {
            RawStreamingChoice::Reasoning { id, content } => {
                assert_eq!(id, None);
                match content {
                    ReasoningContent::Text { text, signature } => {
                        assert!(text.is_empty());
                        assert_eq!(signature.as_deref(), Some("sig-only"));
                    }
                    other => panic!("unexpected reasoning content: {other:?}"),
                }
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }

    #[test]
    fn test_content_stop_without_tracked_thought_emits_nothing() {
        assert!(content_stop_to_choice(0, &mut HashMap::new()).is_none());
    }

    #[test]
    fn test_content_start_thought_seeds_summary_and_signature() {
        let mut thought_states = HashMap::new();

        let start_choices = content_start_to_choices(
            0,
            Content::Thought(ThoughtContent {
                summary: Some(vec![ThoughtSummaryContent::Text(TextContent {
                    text: "seeded".to_string(),
                    annotations: None,
                })]),
                signature: Some("sig-seeded".to_string()),
            }),
            &mut thought_states,
        );

        assert_eq!(start_choices.len(), 1);
        match &start_choices[0] {
            RawStreamingChoice::ReasoningDelta { id, reasoning } => {
                assert_eq!(id, &None);
                assert_eq!(reasoning, "seeded");
            }
            other => panic!("unexpected choice: {other:?}"),
        }

        let stop_choice =
            content_stop_to_choice(0, &mut thought_states).expect("stop should emit reasoning");
        match stop_choice {
            RawStreamingChoice::Reasoning { id, content } => {
                assert_eq!(id, None);
                match content {
                    ReasoningContent::Text { text, signature } => {
                        assert_eq!(text, "seeded");
                        assert_eq!(signature.as_deref(), Some("sig-seeded"));
                    }
                    other => panic!("unexpected reasoning content: {other:?}"),
                }
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }
}
