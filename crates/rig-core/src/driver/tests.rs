//! The driver's own tests, over a fake wire and a fake socket.
//!
//! These are the tests that used to live per provider or in
//! `http_client/sse/tests.rs`: the non-success funnel's four cells, the
//! observation fact order, the connect-error-is-the-only-item rule, paging,
//! and the property the whole model exists for — a unary reply and a
//! streamed reply of the same content fold to the same response.

use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;

use super::{Model, Transport};
use crate::completion::CompletionRequest;
use crate::error::{EncodeError, ProviderError};
use crate::http_client::framing::Framing;
use crate::message::{CallId, ToolName};
use crate::model::{ModelInfo, ModelList};
use crate::observe::{
    AdapterContext, AdapterEvent, AdapterUsage, AdapterVerdict, ObservationLog, Subject,
};
use crate::operation::{Completion, Finish, ModelListing, ModelPage, TextPart};
use crate::streaming::Streamed;
use crate::test_utils::{
    HttpErrorStreamingClient, MockHttpResponse, MockStreamingClient, NonSuccessStreamingClient,
    RecordingHttpClient, SequencedHttpClient, SequencedStreamingHttpClient,
};
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Mode, ObservationSink, Out, Wire, WireEvent,
    WireFrame,
};

/// Fold one reply of `wire` over `http`.
pub(crate) async fn call<W, H>(
    wire: &W,
    http: &H,
    request: crate::wire::Request<W>,
    context: Option<AdapterContext>,
) -> Result<crate::wire::Response<W>, ProviderError>
where
    W: Wire,
    H: Transport<W>,
{
    let model = Model::new(wire.clone(), http.clone());
    match context {
        Some(context) => model.call_observed(request, context).await,
        None => model.call(request).await,
    }
}

/// One streamed reply of `wire` over `http`.
pub(crate) fn stream<W, H>(
    wire: &W,
    http: &H,
    request: crate::wire::Request<W>,
    context: Option<AdapterContext>,
) -> Result<Streamed<W::Op>, ProviderError>
where
    W: Wire,
    H: Transport<W>,
{
    let model = Model::new(wire.clone(), http.clone());
    match context {
        Some(context) => model.stream_observed(request, context),
        None => model.stream(request),
    }
}

// ── the fake completion wire ────────────────────────────────────────────

/// A wire whose reply is either a whole message (`{"text":…}`) or a stream
/// of the same shape's deltas — the Anthropic/OpenAI split, in miniature.
#[derive(Clone, Debug, PartialEq)]
struct Echo {
    framing: Framing,
    request_id_header: Option<&'static str>,
    relaxed_content_type: bool,
}

impl Echo {
    fn unary() -> Self {
        Self {
            framing: Framing::Whole,
            request_id_header: Some("request-id"),
            relaxed_content_type: false,
        }
    }

    fn streaming() -> Self {
        Self {
            framing: Framing::Sse,
            request_id_header: Some("request-id"),
            relaxed_content_type: false,
        }
    }

    /// A wire whose *unary* reply is an event stream — the Responses
    /// endpoint's shape on a dialect that always streams.
    fn sse_unary() -> Self {
        Self::streaming()
    }

    fn relaxed(mut self) -> Self {
        self.relaxed_content_type = true;
        self
    }
}

/// One frame of the fake wire.
#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Frame {
    /// The whole message: the unary reply's shape.
    Message { text: String, usage: Usage },
    /// One text delta.
    Delta { text: String },
    /// The provider's own end of turn.
    Stop { usage: Usage },
    /// A whole tool call whose block the provider declared complete, ending
    /// the turn when it carries usage.
    Tool {
        name: String,
        arguments: String,
        usage: Option<Usage>,
    },
}

#[derive(Clone, Copy, serde::Deserialize)]
struct Usage {
    output_tokens: u64,
}

#[derive(Default)]
struct EchoDecoder<'id> {
    text: Option<TextPart<'id>>,
}

impl<'id> Decoder<'id, Completion> for EchoDecoder<'id> {
    type Event = Frame;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_tagged_frame(&frame.as_str(), "type", |kind| {
            matches!(kind, "message" | "delta" | "stop" | "tool")
        })
    }

    fn decode(
        &mut self,
        event: Frame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            // A whole message is the stream's text and end in one frame.
            Frame::Message { text, usage } => {
                self.push_text(&mut out, &text);
                return Ok(self.end(out, usage));
            }
            Frame::Delta { text } => self.push_text(&mut out, &text),
            Frame::Stop { usage } => return Ok(self.end(out, usage)),
            Frame::Tool {
                name,
                arguments,
                usage,
            } => {
                if let Some(part) = self.text.take() {
                    out.close_text(part);
                }
                let name = ToolName::new(name)
                    .map_err(|error| ProviderError::Response(error.to_string()))?;
                let part = out.call(CallId::from_wire("call_1"), name)?;
                out.push_arguments(&part, &arguments);
                out.close_call(part)?;
                if let Some(usage) = usage {
                    return Ok(self.end(out, usage));
                }
            }
        }
        Ok(Flow::More)
    }
}

impl<'id> EchoDecoder<'id> {
    fn push_text(&mut self, out: &mut Out<'id, Completion>, text: &str) {
        let part = self.text.get_or_insert_with(|| out.text());
        out.push_text(part, text);
    }

    fn end(&mut self, mut out: Out<'id, Completion>, usage: Usage) -> Flow {
        if let Some(part) = self.text.take() {
            out.close_text(part);
        }
        out.end(Finish {
            usage: crate::completion::Usage {
                output_tokens: Some(usage.output_tokens),
                ..crate::completion::Usage::default()
            },
            model: Some("echo-1".to_owned()),
            ..Finish::default()
        })
    }
}

impl EchoDecoder<'_> {
    fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(payload) else {
            return;
        };
        if let Some(tokens) = value
            .pointer("/usage/output_tokens")
            .and_then(|v| v.as_u64())
        {
            sink.emit(AdapterEvent::Usage {
                usage: AdapterUsage {
                    output_tokens: Some(tokens),
                    ..AdapterUsage::default()
                },
            });
        }
        sink.provider(
            AdapterVerdict {
                model: Some(sink.scrub("echo-1")),
                ..AdapterVerdict::default()
            },
            None,
        );
    }
}

impl Wire for Echo {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = EchoDecoder<'id>;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("echo").model("echo-1")
    }

    fn encode(&self, request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::to_vec(&serde_json::json!({
            "messages": request.chat_history.len(),
        }))?;
        let request =
            http::Request::post("https://echo.invalid/v1/messages").body(Body::Bytes(body))?;
        let encoded = Encoded::new(request, self.framing)
            .with_request_id_header(self.request_id_header)
            .with_projection(EchoDecoder::project);
        Ok(if self.relaxed_content_type {
            encoded.with_relaxed_content_type()
        } else {
            encoded
        })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        EchoDecoder::default()
    }
}

fn prompt() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: crate::NonEmpty::new(crate::message::Message::user("say hi")),
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

// ── the property the model exists for ───────────────────────────────────

const UNARY_BODY: &str = r#"{"type":"message","text":"hi there","usage":{"output_tokens":3}}"#;
const STREAM_BODY: &str = concat!(
    "data: {\"type\":\"delta\",\"text\":\"hi \"}\n\n",
    "data: {\"type\":\"delta\",\"text\":\"there\"}\n\n",
    "data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":3}}\n\n",
);

#[tokio::test]
async fn a_unary_reply_and_a_streamed_reply_fold_to_the_same_response() {
    let unary = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
    let buffered = unary.call(prompt()).await.expect("the reply decodes");

    let streaming = Model::new(
        Echo::streaming(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(STREAM_BODY.as_bytes()),
        },
    );
    let mut response = streaming.stream(prompt()).expect("the stream opens");
    while response.next().await.is_some() {}
    let streamed = response.finish().await.expect("the stream ended");

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.model, streamed.model);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(
        buffered.choice.first().and_then(|block| match block {
            crate::message::AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        }),
        Some("hi there")
    );
}

#[tokio::test]
async fn a_unary_reply_carries_its_body_as_raw() {
    let bound = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
    let response = bound.call(prompt()).await.expect("the reply decodes");
    assert_eq!(
        response.raw.pointer("/text").and_then(|text| text.as_str()),
        Some("hi there")
    );
}

// ── the non-success funnel: four cells ──────────────────────────────────

// ── the two paths read the same reply the same way ──────────────────────

/// A wire whose unary reply is an event stream: an SSE framer over a JSON
/// body yields no frames at all, so accepting the reply would fold to a
/// contentless success instead of naming the wrong endpoint.
#[tokio::test]
async fn a_unary_reply_that_is_not_the_event_stream_it_asked_for_fails_the_call() {
    let http = SequencedHttpClient::new([MockHttpResponse::success_typed(
        r#"{"error":"this endpoint speaks JSON"}"#,
        "application/json",
    )]);
    let error = call(&Echo::sse_unary(), &http, prompt(), None)
        .await
        .expect_err("a JSON reply to an SSE wire is the wrong endpoint");
    assert!(
        error.to_string().contains("content type"),
        "the error names what was wrong: {error}"
    );
}

/// The opt-out the gateway that replays Responses bodies without a content
/// type needs; it is a wire's declaration, not a reply's accident.
#[tokio::test]
async fn a_relaxed_wire_still_reads_a_unary_reply_that_names_no_content_type() {
    let http =
        SequencedHttpClient::new([MockHttpResponse::success(format!("data: {UNARY_BODY}\n\n"))]);
    let response = call(&Echo::sse_unary().relaxed(), &http, prompt(), None)
        .await
        .expect("the reply decodes");
    assert_eq!(text_of(&response), Some("hi there"));
}

/// Projection walks framed payloads, not the raw body: an SSE-framed unary
/// reply's facts are JSON *inside* `data:` lines, so projecting the body
/// would hand every projector a document it cannot parse and the trace
/// would silently lose its usage and verdict.
#[tokio::test]
async fn a_unary_reply_projects_the_same_facts_a_streamed_one_does() {
    let body = format!("data: {UNARY_BODY}\n\n");
    let (unary_log, unary_context) = observed();
    let http = SequencedHttpClient::new([MockHttpResponse::success_typed(
        body.clone(),
        "text/event-stream",
    )]);
    call(&Echo::sse_unary(), &http, prompt(), Some(unary_context))
        .await
        .expect("the reply decodes");

    let (streamed_log, streamed_context) = observed();
    let http = MockStreamingClient {
        sse_bytes: Bytes::from(body),
    };
    let frames = stream(&Echo::streaming(), &http, prompt(), Some(streamed_context))
        .expect("the stream opens");
    let _: Vec<_> = frames.collect().await;

    let facts = |log: &ObservationLog| {
        events(log)
            .into_iter()
            .filter(|event| event != "finished")
            .collect::<Vec<_>>()
    };
    assert_eq!(facts(&unary_log), facts(&streamed_log));
    assert!(
        facts(&unary_log).contains(&"usage".to_owned()),
        "the projection ran: {:?}",
        facts(&unary_log)
    );
}

/// A transport that reports the reply as an error, with a request-id header.
#[tokio::test]
async fn a_transport_reported_failure_preserves_status_headers_and_request_id() {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("7"));
    headers.insert("request-id", http::HeaderValue::from_static("req_1"));
    let http = RecordingHttpClient::with_error_headers(
        http::StatusCode::TOO_MANY_REQUESTS,
        "slow down",
        headers,
    );
    let error = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect_err("a 429 fails the call");
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(error.provider_request_id(), Some("req_1"));
    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get("retry-after"))
            .and_then(|value| value.to_str().ok()),
        Some("7")
    );
}

/// A transport that hands the non-success *response* back instead.
#[tokio::test]
async fn a_non_success_response_preserves_status_headers_and_request_id() {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("11"));
    headers.insert("request-id", http::HeaderValue::from_static("req_2"));
    let http = SequencedHttpClient::new([MockHttpResponse::error_with_headers(
        http::StatusCode::SERVICE_UNAVAILABLE,
        "down",
        headers,
    )]);
    let error = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect_err("a 503 fails the call");
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_request_id(), Some("req_2"));
    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get("retry-after"))
            .and_then(|value| value.to_str().ok()),
        Some("11")
    );
}

#[tokio::test]
async fn an_undecodable_body_fails_the_call_as_a_json_error() {
    let http = RecordingHttpClient::new("not json at all");
    let error = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect_err("a non-JSON reply fails the call");
    assert!(
        matches!(error, ProviderError::Json(_)),
        "expected a JSON error, got {error:?}"
    );
}

/// A reply that framed to nothing never reached the provider's end: it is
/// truncated, never an empty answer.
#[tokio::test]
async fn a_reply_with_no_frames_is_truncated() {
    let http = RecordingHttpClient::new("");
    let error = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect_err("an empty body is not an answer");
    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
}

// ── streaming semantics ─────────────────────────────────────────────────

#[tokio::test]
async fn a_connect_failure_is_the_streams_only_item() {
    let http = HttpErrorStreamingClient::new(http::StatusCode::UNAUTHORIZED, "{\"error\":\"no\"}");
    let frames = stream(&Echo::streaming(), &http, prompt(), None)
        .expect("opening the stream is not an encode error");
    let items: Vec<_> = frames.collect().await;
    assert_eq!(items.len(), 1);
    let error = items
        .into_iter()
        .next()
        .expect("one item")
        .expect_err("the item is the connect failure");
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::UNAUTHORIZED)
    );
}

/// A streaming transport that hands the non-success *response* back as `Ok`:
/// the driver rejects it on status, and the reply's status, headers, request
/// id and body are the error — the only item.
#[tokio::test]
async fn a_non_success_streaming_response_is_rejected_as_the_streams_only_item() {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("13"));
    headers.insert("request-id", http::HeaderValue::from_static("req_3"));
    let http = NonSuccessStreamingClient {
        status: http::StatusCode::SERVICE_UNAVAILABLE,
        headers,
        body: Bytes::from_static(b"{\"error\":\"down\"}"),
    };
    let frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = frames.collect().await;
    assert_eq!(items.len(), 1, "nothing follows the rejection: {items:?}");
    let error = items
        .into_iter()
        .next()
        .expect("one item")
        .expect_err("the item is the rejected reply");
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_request_id(), Some("req_3"));
    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get("retry-after"))
            .and_then(|value| value.to_str().ok()),
        Some("13")
    );
    assert!(
        error
            .provider_response_body()
            .is_some_and(|body| body.contains("down")),
        "the reply body is the error: {error:?}"
    );
}

#[tokio::test]
async fn a_transport_failure_mid_stream_ends_it_without_a_terminal() {
    let http = SequencedStreamingHttpClient::new(vec![
        Ok(Bytes::from_static(
            b"data: {\"type\":\"delta\",\"text\":\"hi\"}\n\n",
        )),
        Err(crate::http_client::Error::StreamEnded),
    ]);
    let mut frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = (&mut frames).collect().await;
    assert!(
        items.last().is_some_and(|item| item.is_err()),
        "the transport error is the last item"
    );
    assert!(
        frames.finish().await.is_err(),
        "truncation never fabricates an end"
    );
}

#[tokio::test]
async fn a_corrupt_frame_ends_the_reply() {
    let body = concat!(
        "data: {\"type\":\"delta\",\"text\":\"hi\"}\n\n",
        "data: {\"type\":\"delta\"}\n\n",
        "data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":1}}\n\n",
    );
    let http = MockStreamingClient {
        sse_bytes: Bytes::from(body),
    };
    let frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = frames.collect().await;
    assert_eq!(items.iter().filter(|item| item.is_err()).count(), 1);
    assert!(
        items.last().is_some_and(Result::is_err),
        "nothing follows the corrupt frame: {items:?}"
    );
}

#[tokio::test]
async fn a_heartbeat_frame_never_reaches_the_decoder() {
    let body = concat!(
        ": keep-alive\n\n",
        "data:  \n\n",
        "data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":0}}\n\n",
    );
    let http = MockStreamingClient {
        sse_bytes: Bytes::from(body),
    };
    let mut frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = (&mut frames).collect().await;
    assert!(items.is_empty(), "only the end is a frame: {items:?}");
    frames.finish().await.expect("the end still ends the reply");
}

#[tokio::test]
async fn a_streams_terminal_carries_the_transport_request_id() {
    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(STREAM_BODY.as_bytes()),
    };
    // `MockStreamingClient` reports no request-id header, so a wire that
    // names one still yields `None` rather than failing.
    let mut frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    while frames.next().await.is_some() {}
    let response = frames.finish().await.expect("the stream ended");
    assert_eq!(response.provider_request_id, None);
}

// ── observation ────────────────────────────────────────────────────────

fn observed() -> (Arc<ObservationLog>, AdapterContext) {
    let log = Arc::new(ObservationLog::with_capacity(64));
    let context = AdapterContext::new(log.clone(), Subject::default(), "driver-test".to_owned());
    (log, context)
}

/// The names of the adapter events a trace recorded, in order.
fn events(log: &ObservationLog) -> Vec<String> {
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            crate::observe::Action::Adapter { observation } => Some(
                serde_json::to_value(&observation.event)
                    .ok()?
                    .get("event")?
                    .as_str()?
                    .to_owned(),
            ),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_streams_observation_facts_arrive_in_the_old_order() {
    let (log, context) = observed();
    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(STREAM_BODY.as_bytes()),
    };
    let frames =
        stream(&Echo::streaming(), &http, prompt(), Some(context)).expect("the stream opens");
    let _: Vec<_> = frames.collect().await;
    assert_eq!(
        events(&log),
        vec![
            "started", "response",
            // Each frame's payload is projected before it is classified.
            "provider", "provider", "usage", "provider", "finished",
        ]
    );
}

#[tokio::test]
async fn a_unary_call_closes_its_attempt_as_decoded() {
    let (log, context) = observed();
    let http = RecordingHttpClient::new(UNARY_BODY);
    call(&Echo::unary(), &http, prompt(), Some(context))
        .await
        .expect("the reply decodes");
    assert_eq!(
        events(&log),
        vec!["started", "response", "usage", "provider", "finished"]
    );
}

/// How an attempt ended, when it did.
fn ending(log: &ObservationLog) -> Option<AdapterEvent> {
    log.trace()
        .observations
        .iter()
        .find_map(|observation| match &observation.action {
            crate::observe::Action::Adapter { observation }
                if matches!(observation.event, AdapterEvent::Finished { .. }) =>
            {
                Some(observation.event.clone())
            }
            _ => None,
        })
}

/// A malformed complete tool input is a decode defect that ends the reply,
/// so the attempt ends as a decode error in both modes.
#[tokio::test]
async fn a_malformed_tool_input_ends_the_attempt_as_a_decode_error() {
    let decode_error = AdapterEvent::Finished {
        ending: crate::observe::AdapterEnding::Error {
            boundary: crate::observe::AdapterErrorBoundary::Decode,
            kind: "response".into(),
            status: None,
            retryable: false,
        },
    };

    let (log, context) = observed();
    let http = RecordingHttpClient::new(
        r#"{"type":"tool","name":"add","arguments":"{not json","usage":{"output_tokens":1}}"#,
    );
    let error = call(&Echo::unary(), &http, prompt(), Some(context))
        .await
        .expect_err("the defect fails a whole reply");
    assert!(
        matches!(error, ProviderError::MalformedToolInput(_)),
        "{error:?}"
    );
    assert_eq!(ending(&log), Some(decode_error.clone()));

    let (log, context) = observed();
    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(
            b"data: {\"type\":\"tool\",\"name\":\"add\",\"arguments\":\"{not json\"}\n\n\
              data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":1}}\n\n",
        ),
    };
    let items: Vec<_> = Model::new(Echo::streaming(), http)
        .stream_observed(prompt(), context)
        .expect("the stream opens")
        .collect()
        .await;
    assert!(
        matches!(
            items.last(),
            Some(Err(ProviderError::MalformedToolInput(_)))
        ),
        "the defect ends the stream: {items:?}"
    );
    assert_eq!(ending(&log), Some(decode_error));
}

#[tokio::test]
async fn a_failed_unary_call_projects_the_reply_it_failed_on() {
    let (log, context) = observed();
    let http = RecordingHttpClient::with_error_response(
        http::StatusCode::BAD_REQUEST,
        r#"{"usage":{"output_tokens":0}}"#,
    );
    call(&Echo::unary(), &http, prompt(), Some(context))
        .await
        .expect_err("a 400 fails the call");
    let events = events(&log);
    assert!(
        events.contains(&"usage".to_owned()),
        "the failed reply's facts are still projected: {events:?}"
    );
    assert_eq!(events.last().map(String::as_str), Some("finished"));
}

// ── paging ─────────────────────────────────────────────────────────────

/// A model-listing wire whose first reply names a second page.
#[derive(Clone, Debug, PartialEq)]
struct Catalogue;

#[derive(Default)]
struct CatalogueDecoder;

#[derive(serde::Deserialize)]
struct Page {
    data: Vec<String>,
    #[serde(default)]
    next: Option<String>,
}

impl<'id> Decoder<'id, ModelListing> for CatalogueDecoder {
    type Event = Page;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        match serde_json::from_str(&frame.as_str()) {
            Ok(page) => WireEvent::Known(page),
            Err(error) => WireEvent::Corrupt(error),
        }
    }

    fn decode(&mut self, page: Page, out: Out<'id, ModelListing>) -> Result<Flow, ProviderError> {
        Ok(out.end(ModelPage {
            models: ModelList::new(page.data.into_iter().map(ModelInfo::from_id).collect()),
            next: page.next,
        }))
    }
}

impl Wire for Catalogue {
    type Op = ModelListing;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = CatalogueDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("echo")
    }

    fn encode(&self, cursor: Option<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let uri = match cursor {
            Some(cursor) => format!("https://echo.invalid/v1/models?after={cursor}"),
            None => "https://echo.invalid/v1/models".to_owned(),
        };
        let request = http::Request::get(uri).body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        CatalogueDecoder
    }
}

#[tokio::test]
async fn a_paged_listing_follows_every_continuation() {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(r#"{"data":["a","b"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["c"]}"#),
    ]);
    let bound = Model::new(Catalogue, http.clone());
    let models = bound.list().await.expect("both pages decode");
    assert_eq!(
        models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c"]
    );
    let paths: Vec<_> = http
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    assert_eq!(
        paths,
        vec![
            "https://echo.invalid/v1/models".to_owned(),
            "https://echo.invalid/v1/models?after=b".to_owned(),
        ]
    );
}

/// One call reads one page, whatever cursor it names.
#[tokio::test]
async fn a_listing_call_reads_one_page() {
    let http = SequencedHttpClient::new([MockHttpResponse::success(
        r#"{"data":["a","b"],"next":"b"}"#,
    )]);
    let page = Model::new(Catalogue, http.clone())
        .call(None)
        .await
        .expect("the page decodes");
    assert_eq!(page.models.len(), 2);
    assert_eq!(page.next.as_deref(), Some("b"));
    assert_eq!(http.requests().len(), 1);
}

#[tokio::test]
async fn a_listing_that_repeats_its_cursor_stops_after_the_repeated_page() {
    // The second page names the cursor that fetched it, so the next request
    // would be identical to the one just answered (rig#2334).
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(r#"{"data":["a"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["b"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["never"]}"#),
    ]);
    let bound = Model::new(Catalogue, http.clone());
    let models = bound.list().await.expect("the fetched pages decode");
    assert_eq!(
        models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(
        http.requests().len(),
        2,
        "the repeated cursor is not re-sent"
    );
}

#[tokio::test]
async fn a_listing_whose_cursor_keeps_changing_stops_at_the_page_ceiling() {
    let pages = (0..super::MAX_CONTINUATION_PAGES + 5).map(|page| {
        MockHttpResponse::success(format!(r#"{{"data":["m{page}"],"next":"c{}"}}"#, page + 1))
    });
    let http = SequencedHttpClient::new(pages);
    let bound = Model::new(Catalogue, http.clone());
    let models = bound.list().await.expect("the fetched pages decode");
    assert_eq!(models.len(), super::MAX_CONTINUATION_PAGES);
    assert_eq!(http.requests().len(), super::MAX_CONTINUATION_PAGES);
}

/// Every warning a body emits, by message.
fn warnings_of(body: impl std::future::Future<Output = ()>) -> Vec<String> {
    #[derive(Clone, Default)]
    struct Warnings(Arc<std::sync::Mutex<Vec<String>>>);
    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Warnings {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if *event.metadata().level() == tracing::Level::WARN {
                let mut fields = Vec::new();
                event.record(&mut Visit(&mut fields));
                if let Ok(mut warnings) = self.0.lock() {
                    warnings.extend(
                        fields
                            .into_iter()
                            .filter(|(name, _)| name == "message")
                            .map(|(_, message)| message),
                    );
                }
            }
        }
    }
    use tracing_subscriber::layer::SubscriberExt;
    let warnings = Warnings::default();
    let subscriber = tracing_subscriber::registry().with(warnings.clone());
    let _guard = tracing::subscriber::set_default(subscriber);
    futures::executor::block_on(body);
    warnings
        .0
        .lock()
        .map(|warnings| warnings.clone())
        .unwrap_or_default()
}

/// A repeated cursor stops after the repeated page and a cursor that keeps
/// changing stops at the ceiling, each with its warning.
#[test]
fn a_listing_warns_when_a_cursor_guard_stops_it() {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(r#"{"data":["a"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["b"],"next":"b"}"#),
    ]);
    let warnings = warnings_of(async {
        let models = Model::new(Catalogue, http.clone())
            .list()
            .await
            .expect("the listing decodes");
        assert_eq!(models.len(), 2, "the repeated page is the last");
    });
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("repeated its pagination cursor")),
        "{warnings:?}"
    );

    let pages = (0..super::MAX_CONTINUATION_PAGES + 5).map(|page| {
        MockHttpResponse::success(format!(r#"{{"data":["m{page}"],"next":"c{}"}}"#, page + 1))
    });
    let http = SequencedHttpClient::new(pages);
    let warnings = warnings_of(async {
        Model::new(Catalogue, http.clone())
            .list()
            .await
            .expect("the listing decodes");
    });
    assert!(
        warnings
            .iter()
            .any(|warning| warning.contains("hit its page ceiling")),
        "{warnings:?}"
    );
}

/// An embedding reply over HTTP streams no events, and finishes to what a
/// call folds.
#[tokio::test]
async fn a_streamed_embedding_finishes_to_the_call() {
    const REPLY: &str = r#"{"object":"list","model":"text-embedding-3-small","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]},{"object":"embedding","index":1,"embedding":[0.3,0.4]}],"usage":{"prompt_tokens":4,"total_tokens":4}}"#;
    let wire = crate::providers::openai::wire::OpenAIConfig::new("sk-test")
        .embedding("text-embedding-3-small", None);
    let texts = || vec!["a".to_owned(), "b".to_owned()];
    let reply = || SequencedHttpClient::new([MockHttpResponse::success(REPLY)]);

    let called = Model::new(wire.clone(), reply())
        .call(texts())
        .await
        .expect("the reply decodes");
    let mut stream = Model::new(wire, reply())
        .stream(texts())
        .expect("the embedding opens");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("the reply decodes"));
    }
    assert!(events.is_empty(), "the reply's end is the response");
    let streamed = stream.finish().await.expect("the reply folds");
    assert_eq!(
        serde_json::to_value(&streamed).expect("json"),
        serde_json::to_value(&called).expect("json")
    );
    assert_eq!(
        streamed
            .embeddings
            .iter()
            .map(|embedding| embedding.document.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
}

// ── telemetry ──────────────────────────────────────────────────────────

/// A completion's span is named for the mode the call runs in.
#[test]
fn the_completion_operation_names_its_span_by_mode() {
    use tracing::subscriber::with_default;
    use tracing_subscriber::layer::SubscriberExt;

    let recorded = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
    let subscriber = tracing_subscriber::registry().with(RecordFields {
        recorded: recorded.clone(),
    });
    with_default(subscriber, || {
        let unary = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
        futures::executor::block_on(unary.call(prompt())).expect("the reply decodes");
        let streaming = Model::new(
            Echo::streaming(),
            MockStreamingClient {
                sse_bytes: Bytes::from_static(STREAM_BODY.as_bytes()),
            },
        );
        let stream = streaming.stream(prompt()).expect("the stream opens");
        futures::executor::block_on(stream.collect::<Vec<_>>());
    });
    let names: Vec<_> = recorded
        .lock()
        .expect("no panic held the lock")
        .iter()
        .filter(|(field, _)| field == "gen_ai.operation.name")
        .map(|(_, value)| value.clone())
        .collect();
    assert_eq!(names, vec!["chat", "chat_streaming"]);
}

#[tokio::test]
async fn the_driver_records_the_folded_responses_metadata() {
    use tracing::subscriber::with_default;
    use tracing_subscriber::layer::SubscriberExt;

    let recorded = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
    let layer = RecordFields {
        recorded: recorded.clone(),
    };
    let subscriber = tracing_subscriber::registry().with(layer);
    let bound = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
    let response = with_default(subscriber, || {
        futures::executor::block_on(bound.call(prompt()))
    })
    .expect("the reply decodes");
    assert_eq!(response.model.as_deref(), Some("echo-1"));
    let recorded = recorded.lock().expect("no panic held the lock").clone();
    assert!(
        recorded
            .iter()
            .any(|(field, value)| field == "gen_ai.response.model" && value == "echo-1"),
        "expected the response model on the span: {recorded:?}"
    );
    assert!(
        recorded
            .iter()
            .any(|(field, _)| field == "gen_ai.usage.output_tokens"),
        "expected usage on the span: {recorded:?}"
    );
}

/// A streamed embedding records its response on the span when it
/// finishes, as a call does.
#[test]
fn a_streamed_embedding_records_its_response_on_the_span() {
    use tracing::subscriber::with_default;
    use tracing_subscriber::layer::SubscriberExt;

    const REPLY: &str = r#"{"object":"list","model":"text-embedding-3-small","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"usage":{"prompt_tokens":2,"total_tokens":2}}"#;
    let recorded = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
    let layer = RecordFields {
        recorded: recorded.clone(),
    };
    let subscriber = tracing_subscriber::registry().with(layer);
    let wire = crate::providers::openai::wire::OpenAIConfig::new("sk-test")
        .embedding("text-embedding-3-small", None);
    let model = Model::new(
        wire,
        SequencedHttpClient::new([MockHttpResponse::success(REPLY)]),
    );
    with_default(subscriber, || {
        let stream = model
            .stream(vec!["a".to_owned()])
            .expect("the embedding opens");
        futures::executor::block_on(stream.finish()).expect("the reply folds")
    });
    let recorded = recorded.lock().expect("no panic held the lock").clone();
    assert!(
        recorded
            .iter()
            .any(|(field, value)| field == "gen_ai.response.model"
                && value == "text-embedding-3-small"),
        "expected the response model on the span: {recorded:?}"
    );
}

#[tokio::test]
async fn the_span_names_the_requests_model_override_not_the_wires() {
    use tracing::subscriber::with_default;
    use tracing_subscriber::layer::SubscriberExt;

    let recorded = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
    let layer = RecordFields {
        recorded: recorded.clone(),
    };
    let subscriber = tracing_subscriber::registry().with(layer);
    let bound = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
    let request = CompletionRequest {
        model: Some("echo-override".to_owned()),
        ..prompt()
    };
    with_default(subscriber, || {
        futures::executor::block_on(bound.call(request))
    })
    .expect("the reply decodes");
    let recorded = recorded.lock().expect("no panic held the lock").clone();
    assert!(
        recorded
            .iter()
            .any(|(field, value)| field == "gen_ai.request.model" && value == "echo-override"),
        "expected the override on the span: {recorded:?}"
    );
}

/// Captures every field recorded on a span, so the driver's telemetry is
/// asserted through `tracing` rather than through its own call sites.
struct RecordFields {
    recorded: Arc<std::sync::Mutex<Vec<(String, String)>>>,
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for RecordFields {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut recorded = self
            .recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        attrs.record(&mut Visit(&mut recorded));
    }

    fn on_record(
        &self,
        _id: &tracing::Id,
        values: &tracing::span::Record<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut recorded = self
            .recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        values.record(&mut Visit(&mut recorded));
    }
}

struct Visit<'a>(&'a mut Vec<(String, String)>);
impl tracing::field::Visit for Visit<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().to_owned(), format!("{value:?}")));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.push((field.name().to_owned(), value.to_owned()));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.push((field.name().to_owned(), value.to_string()));
    }
}

/// The text a folded response carries, for the assertions below.
fn text_of(response: &crate::completion::CompletionResponse) -> Option<&str> {
    response.choice.first().and_then(|block| match block {
        crate::message::AssistantContent::Text(text) => Some(text.text.as_str()),
        _ => None,
    })
}

/// A stream opens one byte-body request. A wire that encodes a multipart
/// body for a stream fails before anything is sent, as a request that could
/// not be built: the stream's only item.
#[test]
fn a_stream_the_driver_cannot_send_is_a_request_failure() {
    #[derive(Clone)]
    struct Multipart;
    impl Wire for Multipart {
        type Op = Completion;
        type Payload = Encoded;
        type Frame = WireFrame;
        type Decoder<'id> = EchoDecoder<'id>;
        fn describe(&self) -> Descriptor<'_> {
            Descriptor::new("echo")
        }
        fn encode(&self, _request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
            let request = http::Request::post("https://echo.invalid/a")
                .body(Body::Multipart(crate::http_client::MultipartForm::new()))?;
            Ok(Encoded::new(request, Framing::Sse))
        }
        fn decoder<'id>(&self) -> Self::Decoder<'id> {
            EchoDecoder::default()
        }
    }

    let http = crate::test_utils::RecordingHttpClient::new("");
    let items: Vec<_> = futures::executor::block_on(
        stream(&Multipart, &http, prompt(), None)
            .expect("the request encodes")
            .collect::<Vec<_>>(),
    );
    let [Err(error)] = items.as_slice() else {
        panic!("the failure is the stream's only item: {items:?}");
    };
    assert_eq!(
        error.to_string(),
        "RequestError: a multipart request cannot open a streamed reply"
    );
    assert_eq!(error.kind(), crate::error::ErrorKind::Request);
    assert_eq!(
        error.boundary(),
        crate::observe::AdapterErrorBoundary::Request
    );
    assert!(!error.is_retryable());
    assert!(http.requests().is_empty(), "nothing was sent");
}
