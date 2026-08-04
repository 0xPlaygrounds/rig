use async_stream::stream;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tracing::{Level, enabled};
use tracing_futures::Instrument;

use super::InteractionsCompletionModel;
use super::create_request_body;
use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, FunctionCallDelta, Interaction,
    InteractionSseEvent, InteractionUsage, Step, TextDelta, ThoughtSummaryContent,
    ThoughtSummaryDelta,
};
use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::Request;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::streaming;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use serde_json::{Map, Value};

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>> + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>>>>;

impl<T> InteractionsCompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + 'static,
{
    /// Execute a streaming interaction and preserve the terminal native interaction.
    pub async fn raw_stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::RawStreamingResult<Interaction>, CompletionError> {
        let span = CompletionSpanBuilder::new(
            "gcp.gemini",
            &self.model,
            CompletionOperation::InteractionsStreaming,
        )
        .system_instructions(
            completion_request.preamble.as_deref(),
            completion_request.record_telemetry_content,
        )
        .build();

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
                            InteractionSseEvent::StepDelta { delta, .. } => {
                                if let Some(choice) = content_delta_to_choice(delta) {
                                    yield choice.try_map_final(|_| {
                                        Err(CompletionError::ResponseError(
                                            "Gemini emitted an unexpected terminal interaction delta".to_owned(),
                                        ))
                                    });
                                }
                            }
                            InteractionSseEvent::StepStart { step, .. } => {
                                if let Some(choice) = step_start_to_choice(step) {
                                    yield choice.try_map_final(|_| {
                                        Err(CompletionError::ResponseError(
                                            "Gemini emitted an unexpected terminal interaction start".to_owned(),
                                        ))
                                    });
                                }
                            }
                            InteractionSseEvent::InteractionCompleted { interaction, .. } => {
                                let span = tracing::Span::current();
                                span.record("gen_ai.response.id", &interaction.id);
                                if let Some(model) = interaction.model.clone() {
                                    span.record("gen_ai.response.model", model);
                                }

                                if let Some(usage) = interaction.usage.clone() {
                                    span.record_token_usage(&usage.token_usage());
                                    final_usage = Some(usage);
                                }
                                final_interaction = Some(interaction);
                            }
                            InteractionSseEvent::Error { .. } => {
                                // Preserve the full provider error payload (code +
                                // message) by reusing the raw SSE event JSON, matching
                                // the SSE path's `completion_error_from_body`. The error
                                // arrives over an established stream, so there is no HTTP
                                // status to attach (status: None).
                                yield Err(crate::provider_response::completion_error_from_body(
                                    message.data,
                                ));
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
                        yield Err(CompletionError::from_stream_transport(error));
                        break;
                    }
                }
            }

            event_source.close();

            let mut interaction = final_interaction.unwrap_or_default();
            if interaction.usage.is_none() {
                interaction.usage = final_usage;
            }
            yield Ok(streaming::RawStreamingChoice::FinalResponse(interaction));
        }
        .instrument(span);

        Ok(Box::pin(stream))
    }

    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let stream = self.raw_stream(completion_request).await?;
        let stream = streaming::normalize_stream(stream, |interaction| {
            let mut response = streaming::StreamFinal::new("gemini", interaction.token_usage());
            if !interaction.id.is_empty() {
                response = response.with_message_id(interaction.id);
            }
            response.model = interaction.model;
            response.finish_reason = interaction
                .status
                .as_ref()
                .map(super::map_interaction_finish_reason);
            Ok(response)
        });
        Ok(streaming::StreamingCompletionResponse::stream(stream))
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
                        let Err(err) = data else {
                            continue;
                        };
                        tracing::debug!("Failed to deserialize interactions SSE event: {err}");
                        continue;
                    };

                    yield Ok(data);
                }
                Err(crate::http_client::Error::StreamEnded) => break,
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    yield Err(CompletionError::from_stream_transport(error));
                    break;
                }
            }
        }

        event_source.close();
    };

    Box::pin(stream)
}

fn step_start_to_choice(step: Step) -> Option<streaming::RawStreamingChoice> {
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

fn content_to_choice(content: Content) -> Option<streaming::RawStreamingChoice> {
    match content {
        Content::Text(text) if !text.text.is_empty() => {
            Some(streaming::RawStreamingChoice::Message(text.text))
        }
        Content::FunctionCall(content) => step_start_to_choice(Step::FunctionCall(content)),
        _ => None,
    }
}

fn content_delta_to_choice(delta: ContentDelta) -> Option<streaming::RawStreamingChoice> {
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
