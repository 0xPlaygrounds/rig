use async_stream::stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tracing::{Level, enabled};
use tracing_futures::Instrument;

use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, FunctionCallDelta, Interaction,
    InteractionSseEvent, InteractionUsage, Step, TextDelta, ThoughtSummaryContent,
    ThoughtSummaryDelta, map_interaction_status,
};
use super::{InteractionsCompletionModel, PROVIDER_NAME, create_request_body};
use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::Request;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::providers::gemini::streaming::shared_parts;
use crate::providers::internal::adapter::{AdapterOutput, WireAdapter, WireFrame, run_wire_stream};
use crate::providers::internal::wire::{self, WireEvent};
use crate::streaming;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use serde_json::{Map, Value};

/// The `event_type` values this client models on the Interactions SSE wire.
///
/// [`wire::classify_tagged_frame`] dispatches on this list: a frame whose
/// `event_type` is outside it classifies `Unknown` (driver policy: warn +
/// skip), while a listed value must pass the full [`InteractionSseEvent`]
/// decode or classify `Corrupt`. There is no untagged serde fallback — policy
/// lives in the classify layer, never in serde.
const KNOWN_EVENT_TYPES: &[&str] = &[
    "interaction.created",
    "interaction.completed",
    "interaction.status_update",
    "step.start",
    "step.delta",
    "step.stop",
    "error",
];

/// Classify one Interactions SSE frame. The single classify site for both
/// consumers of this wire: the completion adapter below and the raw
/// [`stream_interaction_events`] surface.
fn classify_interaction_frame(data: &str) -> WireEvent<InteractionSseEvent> {
    wire::classify_tagged_frame(data, "event_type", |event_type| {
        KNOWN_EVENT_TYPES.contains(&event_type)
    })
}

/// Final metadata yielded by an Interactions streaming response.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StreamingCompletionResponse {
    pub usage: Option<InteractionUsage>,
    pub interaction: Option<Interaction>,
    /// Resolved model identifier (e.g. `gemini-2.5-pro-preview-05-06`), extracted from
    /// `Interaction.model`. The Interactions API has no `FinishReason` field; use
    /// `interaction.status` for lifecycle state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>> + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>>>>;

impl From<&StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: &StreamingCompletionResponse) -> crate::completion::Usage {
        value
            .usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default()
    }
}

impl From<StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: StreamingCompletionResponse) -> crate::completion::Usage {
        (&value).into()
    }
}

/// Normalize the Interactions API's terminal streaming record.
///
/// The finish reason comes from the completed interaction's lifecycle status —
/// the API has no `finishReason` field — and is absent when the stream ended
/// without one.
fn map_stream_final(
    response: StreamingCompletionResponse,
) -> Result<streaming::StreamFinal, CompletionError> {
    let usage = (&response).into();
    let interaction = response.interaction.as_ref();
    let finish_reason = interaction
        .and_then(|interaction| interaction.status.as_ref())
        .map(map_interaction_status);
    let message_id = interaction
        .map(|interaction| interaction.id.as_str())
        .filter(|id| !id.is_empty());

    Ok(streaming::StreamFinal::new(PROVIDER_NAME, usage)
        .with_optional_finish_reason(finish_reason)
        .with_optional_response_id(message_id)
        .with_optional_model(response.model_version.as_deref()))
}

impl<T> InteractionsCompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + 'static,
{
    /// Open an Interactions stream whose terminal record stays provider-native.
    ///
    /// The normalized [`CompletionModel::stream`](crate::completion::CompletionModel::stream)
    /// delegates here and maps only the terminal record, so both paths open
    /// exactly one stream over the same request, telemetry, and error handling.
    pub async fn raw_stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::RawStreamingResult<StreamingCompletionResponse>, CompletionError> {
        let span = CompletionSpanBuilder::new(
            PROVIDER_NAME,
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

        let event_source = GenericEventSource::new(self.client.clone(), req);

        // Transport layer: SSE events → `WireFrame`s. Byte splitting and
        // framing only — classification and policy live downstream.
        let transport = stream! {
            let mut event_source = Box::pin(event_source);
            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::debug!("SSE connection opened");
                    }
                    Ok(Event::Message(message)) => {
                        if message.data.trim().is_empty() {
                            continue;
                        }
                        yield Ok(WireFrame::Text(message.data));
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
        };

        let stream: streaming::RawStreamingResult<StreamingCompletionResponse> =
            Box::pin(run_wire_stream(transport, InteractionsAdapter::default()).instrument(span));

        Ok(stream)
    }

    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let inner = self.raw_stream(completion_request).await?;

        Ok(streaming::StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            streaming::normalize_stream(inner, map_stream_final),
        ))
    }
}

/// The Gemini Interactions SSE wire as a [`WireAdapter`].
///
/// Frame-triage policy (warn on `Unknown`, in-band `Err` on `Corrupt`) lives
/// in [`run_wire_stream`], not here — this ends the wire's former
/// debug-log-and-skip handling of every decode failure.
#[derive(Default)]
struct InteractionsAdapter {
    /// A provider `error` event ended the turn; later frames are dead — the
    /// provider aborted, and interpreting more output (or a terminal) would
    /// dress the failure up as a completed turn.
    failed: bool,
}

impl WireAdapter for InteractionsAdapter {
    type Frame = WireFrame;
    type Event = InteractionSseEvent;
    type Response = StreamingCompletionResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<InteractionSseEvent> {
        classify_interaction_frame(&frame.as_str())
    }

    fn interpret(&mut self, event: InteractionSseEvent, out: &mut AdapterOutput<Self::Response>) {
        if self.failed {
            return;
        }

        match event {
            InteractionSseEvent::StepDelta { delta, .. } => {
                if let Some(choice) = content_delta_to_choice(delta) {
                    out.push(Ok(choice));
                }
            }
            InteractionSseEvent::StepStart { step, .. } => {
                if let Some(choice) = step_start_to_choice(step) {
                    out.push(Ok(choice));
                }
            }
            InteractionSseEvent::InteractionCompleted { interaction, .. } => {
                let span = tracing::Span::current();
                span.record("gen_ai.response.id", &interaction.id);
                if let Some(model) = interaction.model.clone() {
                    span.record("gen_ai.response.model", model);
                }
                if let Some(usage) = interaction.usage.as_ref() {
                    span.record_token_usage(&crate::completion::Usage::from(usage));
                }

                // Only a genuine `interaction.completed` event counts as the
                // provider completing the turn; the driver stops consuming
                // after the terminal record. EOF without one is truncation and
                // synthesizes nothing (see `finish`).
                let model_version = interaction.model.clone();
                out.push(Ok(streaming::RawStreamingChoice::FinalResponse(
                    StreamingCompletionResponse {
                        usage: interaction.usage.clone(),
                        interaction: Some(interaction),
                        model_version,
                    },
                )));
            }
            event @ InteractionSseEvent::Error { .. } => {
                // Preserve the provider error payload (code + message) as the
                // error body, matching the blocking path's
                // `completion_error_from_body`. The event is re-serialized
                // from its decoded form — the modeled fields survive. The
                // error arrives over an established stream, so there is no
                // HTTP status to attach (status: None).
                self.failed = true;
                let body = serde_json::to_string(&event).unwrap_or_default();
                out.push(Err(crate::provider_response::completion_error_from_body(
                    body,
                )));
            }
            InteractionSseEvent::InteractionCreated { .. }
            | InteractionSseEvent::InteractionStatusUpdate { .. }
            | InteractionSseEvent::StepStop { .. } => {}
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput<Self::Response>) {
        // EOF without `interaction.completed` is truncation: no terminal
        // record may be synthesized — it would report a successful completion
        // for a turn the provider aborted.
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

                    // Same frame-triage table as the completion path's
                    // `run_wire_stream` driver — this surface yields typed
                    // events rather than grammar events, so the policy is
                    // restated here against the same classify site.
                    match classify_interaction_frame(&message.data) {
                        WireEvent::Known(event) => yield Ok(event),
                        WireEvent::Unknown { event_type, value } => {
                            tracing::warn!(
                                event_type,
                                payload = %value,
                                "skipping unrecognized stream event"
                            );
                        }
                        WireEvent::Corrupt(error) => {
                            yield Err(CompletionError::JsonError(error));
                        }
                    }
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

fn step_start_to_choice(
    step: Step,
) -> Option<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
    match step {
        Step::ModelOutput { content } => content.into_iter().find_map(content_to_choice),
        Step::FunctionCall(FunctionCallContent {
            name,
            arguments,
            id,
        }) => {
            let name = name?;
            let call_id = id.unwrap_or_else(|| name.clone());
            Some(shared_parts::function_call(
                name,
                arguments.unwrap_or(Value::Object(Map::new())),
                Some(call_id),
                None,
            ))
        }
        _ => None,
    }
}

fn content_to_choice(
    content: Content,
) -> Option<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
    match content {
        Content::Text(text) if !text.text.is_empty() => {
            Some(streaming::RawStreamingChoice::Message(text.text))
        }
        Content::FunctionCall(content) => step_start_to_choice(Step::FunctionCall(content)),
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
            Some(shared_parts::function_call(
                name,
                arguments.unwrap_or(Value::Object(Map::new())),
                Some(call_id),
                None,
            ))
        }
        ContentDelta::ThoughtSummary(ThoughtSummaryDelta { content }) => {
            let text = match content {
                ThoughtSummaryContent::Text(text) => text.text,
                _ => return None,
            };
            Some(shared_parts::reasoning_delta(text))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_streaming_completion_response_has_model_version() {
        let response = StreamingCompletionResponse {
            usage: None,
            interaction: None,
            model_version: Some("gemini-2.5-pro-preview-05-06".to_string()),
        };

        assert_eq!(
            response.model_version.as_deref(),
            Some("gemini-2.5-pro-preview-05-06")
        );

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: StreamingCompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.model_version.as_deref(),
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

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[tokio::test]
    async fn truncated_stream_does_not_synthesize_a_terminal_record() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::providers::gemini::Client;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // Content deltas then EOF without `interaction.completed`: the
        // truncated stream must deliver its content but never a synthesized
        // terminal record.
        let sse_bytes = bytes::Bytes::from(
            [r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}"#]
                .iter()
                .map(|event| format!("data: {event}\n\n"))
                .collect::<String>(),
        );

        let client = Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient { sse_bytes })
            .build()
            .expect("build client")
            .interactions_api();
        let model = client.completion_model("gemini-2.5-pro");
        let request = model.completion_request("hello").build();
        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut texts = Vec::new();
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item.expect("stream item should be Ok") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(_) => saw_terminal = true,
                _ => {}
            }
        }

        assert_eq!(texts, ["hi"]);
        assert!(
            !saw_terminal,
            "EOF without interaction.completed must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
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
