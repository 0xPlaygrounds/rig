use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use super::InteractionsCompletionModel;
use super::create_request_body;
use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, FunctionCallDelta, Interaction,
    InteractionSseEvent, InteractionUsage, TextContent, TextDelta, ThoughtSummaryContent,
    ThoughtSummaryDelta,
};
use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::streaming;
use crate::telemetry::SpanCombinator;
use serde_json::{Map, Value};

/// Final metadata yielded by an Interactions streaming response.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StreamingCompletionResponse {
    pub usage: Option<InteractionUsage>,
    pub interaction: Option<Interaction>,
}

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

                        let data = serde_json::from_str::<InteractionSseEvent>(&message.data);

                        let Ok(data) = data else {
                            let err = data.unwrap_err();
                            tracing::debug!(
                                "Failed to deserialize interactions SSE event: {err}"
                            );
                            continue;
                        };

                        match data {
                            InteractionSseEvent::ContentDelta { delta, .. } => {
                                if let Some(choice) = content_delta_to_choice(delta) {
                                    yield Ok(choice);
                                }
                            }
                            InteractionSseEvent::ContentStart { content, .. } => {
                                if let Some(choice) = content_start_to_choice(content) {
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

fn content_start_to_choice(
    content: Content,
) -> Option<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
    match content {
        Content::Text(TextContent { text, .. }) => {
            if text.is_empty() {
                None
            } else {
                Some(streaming::RawStreamingChoice::Message(text))
            }
        }
        Content::FunctionCall(FunctionCallContent {
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

fn content_delta_to_choice(
    delta: ContentDelta,
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
