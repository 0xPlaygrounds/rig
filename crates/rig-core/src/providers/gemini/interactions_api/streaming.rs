use async_stream::stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tracing::{Level, enabled};
use tracing_futures::Instrument;

use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, FunctionCallDelta, Interaction,
    InteractionSseEvent, InteractionUsage, Step, TextDelta, ThoughtSignatureDelta,
    ThoughtSummaryContent, ThoughtSummaryDelta, map_interaction_status,
};
use super::{InteractionsCompletionModel, PROVIDER_NAME, create_request_body};
use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::Request;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::providers::gemini::streaming::shared_parts;
use crate::providers::internal::adapter::{
    AdapterOutput, TriagedFrame, WireAdapter, WireFrame, run_wire_stream, triage_frame,
};
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
    /// Thought-summary text since the last boundary (signed emission or
    /// non-thought output). The grammar requires a full `Reasoning` block to
    /// be the block's *completed* form, but the wire's `thought_signature`
    /// delta carries only the signature — so the adapter restates the
    /// accumulated text (mirroring the REST wire's `thoughtSignature`
    /// handling). Reset on non-thought output to mirror the accumulator's
    /// minted-id boundary.
    thought_buffer: String,
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
            InteractionSseEvent::StepDelta { delta, .. } => match delta {
                ContentDelta::ThoughtSummary(ThoughtSummaryDelta { content }) => {
                    if let ThoughtSummaryContent::Text(text) = content {
                        self.thought_buffer.push_str(&text.text);
                        out.push(Ok(shared_parts::reasoning_delta(text.text)));
                    }
                }
                ContentDelta::ThoughtSignature(ThoughtSignatureDelta { signature }) => {
                    // The signature closes the thinking block: emit the
                    // completed signed `Reasoning` — the full accumulated
                    // text restated with the signature — superseding the
                    // deltas it restates (same treatment as the REST wire's
                    // `thoughtSignature` chunk; the payload is an opaque
                    // provider string, passed through verbatim). A
                    // signature-only block (no preceding summary text) still
                    // emits, so the signature survives into chat history.
                    out.push(Ok(shared_parts::signed_reasoning(
                        std::mem::take(&mut self.thought_buffer),
                        signature,
                    )));
                }
                delta => {
                    if let Some(choice) = content_delta_to_choice(delta) {
                        // Non-thought output closes the open reasoning item
                        // (accumulator minted-id boundary).
                        self.thought_buffer.clear();
                        out.push(Ok(choice));
                    }
                }
            },
            InteractionSseEvent::StepStart { step, .. } => {
                if let Some(choice) = step_start_to_choice(step) {
                    // Non-thought output closes the open reasoning item
                    // (accumulator minted-id boundary).
                    self.thought_buffer.clear();
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

    fn is_finished(&self) -> bool {
        // A provider `error` event is the wire's own in-band terminal:
        // `interpret` already pushed the `Err` and gates itself on `failed`,
        // so the driver must stop reading rather than drain the rest of the
        // transport (and pass through post-error unknown frames).
        self.failed
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
                    // events rather than grammar events, so it applies the
                    // driver's factored per-frame policy against the same
                    // classify site instead of restating the table.
                    match triage_frame(classify_interaction_frame(&message.data)) {
                        Ok(TriagedFrame::Event(event)) => yield Ok(event),
                        // This surface yields typed interaction events, not
                        // grammar events — there is no raw passthrough item to
                        // carry an unknown frame on, so it stays a warned skip
                        // (the completion path surfaces Unknown via the
                        // driver's `RawStreamingChoice::Unknown` passthrough).
                        Ok(TriagedFrame::Unknown(_)) => {}
                        Err(error) => yield Err(error),
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
            // The wire's id when present; never the tool name — a
            // name-as-id fallback collides two same-tool calls in one turn.
            Some(shared_parts::function_call(
                name,
                arguments.unwrap_or(Value::Object(Map::new())),
                id,
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
            // The wire's id when present; never the tool name — a
            // name-as-id fallback collides two same-tool calls in one turn.
            Some(shared_parts::function_call(
                name,
                arguments.unwrap_or(Value::Object(Map::new())),
                id,
                None,
            ))
        }
        // Thought deltas (`thought_summary`, `thought_signature`) are
        // stateful — the adapter accumulates and restates them in
        // `interpret`, so they never reach this stateless mapping.
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

    /// Drive Interactions SSE frames through the full normalized path and
    /// collect what the consumer sees, in order.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    async fn drive_frames(
        frames: &[&str],
    ) -> (
        Vec<Result<crate::streaming::StreamedAssistantContent, String>>,
        crate::streaming::StreamingCompletionResponse,
    ) {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::providers::gemini::Client;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let sse_bytes = bytes::Bytes::from(
            frames
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

        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item.map_err(|error| error.to_string()));
        }
        (items, stream)
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[tokio::test]
    async fn provider_error_event_ends_the_stream_without_draining_later_frames() {
        use crate::streaming::StreamedAssistantContent;

        // A provider `error` event, then more frames: well-formed content, an
        // unknown frame, and a terminal `interaction.completed`. The error
        // must be the LAST item — the driver stops reading (`is_finished`),
        // so nothing after it is interpreted or passed through as `Unknown`.
        let (items, stream) = drive_frames(&[
            r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}"#,
            r#"{"event_type":"error","error":{"code":"internal","message":"boom"}}"#,
            r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"dead"}}"#,
            r#"{"event_type":"something.future","payload":{"x":1}}"#,
            r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
        ])
        .await;

        let error_position = items
            .iter()
            .position(|item| item.is_err())
            .expect("the provider error must reach the consumer");
        assert_eq!(
            error_position,
            items.len() - 1,
            "the in-band error must end the stream: no later text, Unknown passthrough, or terminal; got {items:?}"
        );
        assert!(
            items.iter().any(|item| matches!(
                item,
                Ok(StreamedAssistantContent::Text(text)) if text.text == "hi"
            )),
            "content before the error must survive"
        );
        assert!(stream.response.is_none());
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[tokio::test]
    async fn thought_signature_completes_the_accumulated_reasoning_block() {
        use crate::streaming::StreamedAssistantContent;

        // Text-then-signature: the signed block must restate the full
        // accumulated thought text and carry the signature; the aggregated
        // choice keeps it (superseding the deltas), alongside the later text.
        let (items, stream) = drive_frames(&[
            r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"think1 "}}}"#,
            r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"think2"}}}"#,
            r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig-abc"}}"#,
            r#"{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"answer"}}"#,
        ])
        .await;

        let signed = items
            .iter()
            .find_map(|item| match item {
                Ok(StreamedAssistantContent::Reasoning(reasoning)) => Some(reasoning.clone()),
                _ => None,
            })
            .expect("the signature must yield a completed Reasoning block");
        assert_eq!(
            signed.content,
            vec![crate::completion::message::ReasoningContent::Text {
                text: "think1 think2".to_string(),
                signature: Some("sig-abc".to_string()),
            }],
            "the signed block must restate the accumulated text with the signature"
        );

        // The aggregated choice keeps exactly one reasoning part carrying the
        // signature — the signed restatement superseded the deltas.
        let aggregated: Vec<_> = stream
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect();
        assert_eq!(aggregated.len(), 1, "got {:?}", stream.choice);
        assert_eq!(
            aggregated.first().map(|r| r.content.clone()),
            Some(signed.content)
        );
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    #[tokio::test]
    async fn signature_only_thought_still_carries_the_signature() {
        use crate::streaming::StreamedAssistantContent;

        // Signature with no preceding thought-summary text: the signature is
        // the provider's replay-validated payload and must still survive as a
        // signed (empty-text) Reasoning block.
        let (items, _stream) = drive_frames(&[
            r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig-only"}}"#,
            r#"{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"answer"}}"#,
        ])
        .await;

        let signed = items
            .iter()
            .find_map(|item| match item {
                Ok(StreamedAssistantContent::Reasoning(reasoning)) => Some(reasoning.clone()),
                _ => None,
            })
            .expect("a signature-only block must still yield a signed Reasoning");
        assert_eq!(
            signed.content,
            vec![crate::completion::message::ReasoningContent::Text {
                text: String::new(),
                signature: Some("sig-only".to_string()),
            }]
        );
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
