use async_stream::stream;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing_futures::Instrument;

use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, FunctionCallDelta, Interaction,
    InteractionSseEvent, InteractionUsage, Step, TextDelta, ThoughtSummaryContent,
    ThoughtSummaryDelta,
};
use crate::completion::CompletionError;
use crate::http_client::sse::{BoxedEventSource, Event};
use crate::streaming;
use crate::telemetry::SpanCombinator;
use serde_json::{Map, Value};

/// A live stream of raw Gemini Interactions events.
///
/// The erased SSE transport remains private. Use [`Self::next`] for the
/// default pin-free loop or the [`Stream`] implementation for combinators.
#[must_use = "streams do nothing unless polled"]
pub struct InteractionEventStream {
    inner: BoxedEventSource,
    done: bool,
}

impl InteractionEventStream {
    /// Pull the next decoded provider event without caller pinning or a
    /// [`StreamExt`] import.
    pub async fn next(&mut self) -> Option<Result<InteractionSseEvent, CompletionError>> {
        StreamExt::next(self).await
    }
}

impl Stream for InteractionEventStream {
    type Item = Result<InteractionSseEvent, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        if stream.done {
            return Poll::Ready(None);
        }

        loop {
            match stream.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None | Some(Err(crate::http_client::Error::StreamEnded))) => {
                    stream.done = true;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Ok(Event::Open))) => continue,
                Poll::Ready(Some(Ok(Event::Message(message)))) => {
                    if message.data.trim().is_empty() {
                        continue;
                    }

                    match serde_json::from_str::<InteractionSseEvent>(&message.data) {
                        Ok(event) => return Poll::Ready(Some(Ok(event))),
                        Err(error) => {
                            tracing::debug!(
                                "Failed to deserialize interactions SSE event: {error}"
                            );
                        }
                    }
                }
                Poll::Ready(Some(Err(error))) => {
                    stream.done = true;
                    tracing::error!(?error, "SSE error");
                    return Poll::Ready(Some(Err(CompletionError::from_stream_transport(error))));
                }
            }
        }
    }
}

/// Drive an Interactions SSE stream into the normalized streaming response.
///
/// The transport-free half of the deleted
/// `InteractionsCompletionModel::stream`: identical event handling, identical
/// terminal [`StreamFinal`](streaming::StreamFinal) assembly, identical span
/// recording — only the event source is now the concrete
/// [`BoxedEventSource`] rather than an `H: HttpClientExt` generic.
pub(crate) fn interaction_completion_stream(
    event_source: BoxedEventSource,
    span: tracing::Span,
) -> streaming::CompletionStream {
    let mut event_source = event_source;

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

                    let data = match serde_json::from_str::<InteractionSseEvent>(&message.data) {
                        Ok(data) => data,
                        Err(err) => {
                            tracing::debug!("Failed to deserialize interactions SSE event: {err}");
                            continue;
                        }
                    };

                    match data {
                        InteractionSseEvent::StepDelta { delta, .. } => {
                            if let Some(choice) = content_delta_to_choice(delta) {
                                yield Ok(choice);
                            }
                        }
                        InteractionSseEvent::StepStart { step, .. } => {
                            if let Some(choice) = step_start_to_choice(step) {
                                yield Ok(choice);
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

        // The Interactions API has no `FinishReason` field; use
        // `interaction.status` for lifecycle state.
        let usage = final_usage
            .or_else(|| final_interaction.as_ref().and_then(|i| i.usage.clone()))
            .as_ref()
            .map(InteractionUsage::token_usage)
            .unwrap_or_default();
        let mut final_response = streaming::StreamFinal::new("gemini", usage);
        if let Some(interaction) = final_interaction.as_ref() {
            if !interaction.id.is_empty() {
                final_response = final_response.with_message_id(interaction.id.clone());
            }
            if let Some(model) = interaction.model.as_deref() {
                final_response = final_response.with_model(model);
            }
        }
        yield Ok(streaming::RawStreamingChoice::FinalResponse(final_response));
    }
    .instrument(span);

    streaming::CompletionStream::from_stream(stream)
}

/// Decode an Interactions SSE stream into raw [`InteractionSseEvent`]s.
///
/// The transport-free replacement for the deleted
/// `stream_interaction_events`, which was generic over `H: HttpClientExt`.
pub(crate) fn interaction_event_stream(event_source: BoxedEventSource) -> InteractionEventStream {
    InteractionEventStream {
        inner: event_source,
        done: false,
    }
}

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

    #[tokio::test]
    async fn raw_event_stream_transport_error_is_terminal() {
        let source = futures::stream::iter([
            Err(crate::http_client::Error::InvalidStatusCode(
                http::StatusCode::BAD_GATEWAY,
            )),
            Ok(Event::Open),
        ]);
        let mut stream = interaction_event_stream(Box::pin(source));

        assert!(matches!(stream.next().await, Some(Err(_))));
        assert!(stream.next().await.is_none());
    }
}
