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

use super::{Local, Model, Opened, Transport};
use crate::completion::CompletionRequest;
use crate::error::{EncodeError, ProviderError};
use crate::http_client::framing::Framing;
use crate::model::{ModelInfo, ModelList};
use crate::observe::{
    AdapterContext, AdapterEvent, AdapterUsage, AdapterVerdict, ObservationLog, Subject,
};
use crate::operation::{Completion, Events, ModelListing};
use crate::streaming::{StreamEvent, StreamFinal};
use crate::test_utils::{
    HttpErrorStreamingClient, MockHttpResponse, MockStreamingClient, NonSuccessStreamingClient,
    RecordingHttpClient, SequencedHttpClient, SequencedStreamingHttpClient,
};
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{
    Body, Decoder, Encoded, Fold, Mode, ObservationSink, Operation, Output, Reply, Sink, Wire,
    WireEvent, WireFrame,
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
) -> Result<crate::streaming::Streamed<W::Op>, ProviderError>
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
    /// A whole tool call, its arguments as the wire spelled them.
    Call { arguments: String },
    /// The provider's own end of turn.
    Stop { usage: Usage },
}

#[derive(Clone, Copy, serde::Deserialize)]
struct Usage {
    output_tokens: u64,
}

#[derive(Default)]
struct EchoDecoder;

impl Decoder<Completion> for EchoDecoder {
    type Event = Frame;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_tagged_frame(&frame.as_str(), "type", |kind| {
            matches!(kind, "message" | "delta" | "call" | "stop")
        })
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<Completion>) {
        match event {
            // The unary shape synthesizes the stream's events rather than
            // carrying a second content mapping.
            Frame::Message { text, usage } => {
                out.text(text);
                terminal(out, usage);
            }
            Frame::Delta { text } => out.text(text),
            Frame::Call { arguments } => {
                let id = crate::streaming::BlockId::wire("call_1");
                out.tool_name(&id, "lookup");
                out.tool_arguments(&id, arguments);
                out.tool_end(
                    id,
                    crate::streaming::ToolCallEnd::new(
                        crate::streaming::UnparseableToolInput::Error,
                    ),
                );
            }
            Frame::Stop { usage } => terminal(out, usage),
        }
    }

    fn project(&self, payload: &[u8], sink: &mut dyn ObservationSink) {
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

fn terminal(out: &mut Output<Completion>, usage: Usage) {
    out.close_active_blocks();
    out.final_record(
        StreamFinal::new(
            "echo",
            crate::completion::Usage {
                output_tokens: Some(usage.output_tokens),
                ..crate::completion::Usage::default()
            },
            serde_json::json!({}),
        )
        .with_model("echo-1"),
    );
}

impl Wire for Echo {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder = EchoDecoder;

    fn name(&self) -> &str {
        "echo"
    }

    fn id(&self) -> Option<&str> {
        Some("echo-1")
    }

    fn encode(&self, request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::to_vec(&serde_json::json!({
            "messages": request.chat_history.len(),
        }))?;
        let request =
            http::Request::post("https://echo.invalid/v1/messages").body(Body::Bytes(body))?;
        let encoded =
            Encoded::new(request, self.framing).with_request_id_header(self.request_id_header);
        Ok(if self.relaxed_content_type {
            encoded.with_relaxed_content_type()
        } else {
            encoded
        })
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        EchoDecoder
    }
}

fn prompt() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![crate::message::Message::user("say hi")],
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
    let streamed = response
        .finish()
        .expect("the stream produced a terminal record");

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

/// The driver itself has no empty-turn policy: a reply that framed to
/// nothing folds to a response with nothing in it. Whether that is an
/// answer or a defect is the decoder's to say, from the [`Mode`] it was
/// built for — see
/// [`the_mode_a_decoder_is_built_for_decides_what_its_eof_means`].
#[tokio::test]
async fn a_reply_with_no_frames_folds_to_an_empty_response_with_no_terminal() {
    let http = RecordingHttpClient::new("");
    let response = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect("an empty body yields an empty choice");
    assert!(response.choice.is_empty());
    assert_eq!(response.finish_reason(), None);
    assert_eq!(response.usage, crate::completion::Usage::default());
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
    let frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = frames.collect().await;
    let terminals = items
        .iter()
        .filter(|item| matches!(item, Ok(StreamEvent::Final(_))))
        .count();
    assert_eq!(terminals, 0, "truncation never fabricates a terminal");
    assert!(
        items.last().is_some_and(|item| item.is_err()),
        "the transport error is the last item"
    );
}

#[tokio::test]
async fn a_corrupt_frame_surfaces_in_band_and_the_stream_keeps_consuming() {
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
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, Ok(StreamEvent::Final(_))))
            .count(),
        1,
        "a genuine terminal after a corrupt frame still completes the stream"
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
    let frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = frames.collect().await;
    assert!(items.iter().all(|item| item.is_ok()));
    assert_eq!(items.len(), 1, "only the terminal is a frame");
}

#[tokio::test]
async fn a_streams_terminal_carries_the_transport_request_id() {
    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(STREAM_BODY.as_bytes()),
    };
    // `MockStreamingClient` reports no request-id header, so a wire that
    // names one still yields `None` rather than failing.
    let frames = stream(&Echo::streaming(), &http, prompt(), None).expect("the stream opens");
    let items: Vec<_> = frames.collect().await;
    let terminal = items.iter().find_map(|item| match item {
        Ok(StreamEvent::Final(terminal)) => Some(terminal),
        _ => None,
    });
    assert_eq!(
        terminal.and_then(|terminal| terminal.provider_request_id.as_deref()),
        None
    );
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

/// The ending the trace recorded for the attempt, serialized.
fn ending(log: &ObservationLog) -> Option<serde_json::Value> {
    log.trace()
        .observations
        .iter()
        .find_map(|observation| match &observation.action {
            crate::observe::Action::Adapter { observation } => {
                serde_json::to_value(&observation.event)
                    .ok()
                    .filter(|event| {
                        event.get("event").and_then(|kind| kind.as_str()) == Some("finished")
                    })
                    .and_then(|event| event.get("ending").cloned())
            }
            _ => None,
        })
}

/// A malformed tool input is the sink's error item, so the decoder reports
/// it and the attempt ends as an error on both surfaces, with the same
/// ending a whole reply always recorded for it.
#[tokio::test]
async fn a_malformed_tool_input_ends_the_observed_attempt_as_an_error_on_both_surfaces() {
    const CALL: &str = r#"{"type":"call","arguments":"{not json"}"#;
    let (unary_log, unary_context) = observed();
    let http = RecordingHttpClient::new(CALL);
    call(&Echo::unary(), &http, prompt(), Some(unary_context))
        .await
        .expect_err("a malformed tool input fails the call");

    let (streamed_log, streamed_context) = observed();
    let http = MockStreamingClient {
        sse_bytes: Bytes::from(format!(
            "data: {CALL}\n\ndata: {{\"type\":\"stop\",\"usage\":{{\"output_tokens\":1}}}}\n\n"
        )),
    };
    let stream = stream(&Echo::streaming(), &http, prompt(), Some(streamed_context))
        .expect("the stream opens");
    let items: Vec<_> = stream.collect().await;
    assert_eq!(items.iter().filter(|item| item.is_err()).count(), 1);
    assert!(
        items
            .iter()
            .any(|item| matches!(item, Ok(StreamEvent::Final(_)))),
        "the terminal still completes the stream: {items:?}"
    );

    let unary = ending(&unary_log).expect("the unary attempt ended");
    let streamed = ending(&streamed_log).expect("the streamed attempt ended");
    assert_eq!(
        unary.get("ending").and_then(|e| e.as_str()),
        Some("error"),
        "{unary}"
    );
    assert_eq!(streamed, unary);
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
struct CatalogueDecoder {
    next: Option<String>,
}

#[derive(serde::Deserialize)]
struct Page {
    data: Vec<String>,
    #[serde(default)]
    next: Option<String>,
}

impl Decoder<ModelListing> for CatalogueDecoder {
    type Event = Page;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        match serde_json::from_str(&frame.as_str()) {
            Ok(page) => WireEvent::Known(page),
            Err(error) => WireEvent::Corrupt(error),
        }
    }

    fn interpret(&mut self, page: Self::Event, out: &mut Output<ModelListing>) {
        self.next = page.next;
        out.push(Ok(ModelList::new(
            page.data.into_iter().map(ModelInfo::from_id).collect(),
        )));
    }

    fn cursor(&self) -> Option<String> {
        self.next.clone()
    }
}

impl Wire for Catalogue {
    type Op = ModelListing;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder = CatalogueDecoder;

    fn name(&self) -> &str {
        "echo"
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = http::Request::get("https://echo.invalid/v1/models").body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn page(&self, cursor: &str) -> Result<Encoded, EncodeError> {
        let request = http::Request::get(format!("https://echo.invalid/v1/models?after={cursor}"))
            .body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        CatalogueDecoder::default()
    }
}

#[tokio::test]
async fn a_paged_listing_follows_every_continuation() {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(r#"{"data":["a","b"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["c"]}"#),
    ]);
    let bound = Model::new(Catalogue, http.clone());
    let models = bound.call(()).await.expect("both pages decode");
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
    let models = bound.call(()).await.expect("the fetched pages decode");
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
    let models = bound.call(()).await.expect("the fetched pages decode");
    assert_eq!(models.len(), super::MAX_CONTINUATION_PAGES);
    assert_eq!(http.requests().len(), super::MAX_CONTINUATION_PAGES);
}

// ── telemetry ──────────────────────────────────────────────────────────

#[test]
fn the_completion_operation_names_its_span_by_mode() {
    assert_eq!(
        Completion::telemetry(Mode::Unary),
        crate::telemetry::GenAiOperation::Chat
    );
    assert_eq!(
        Completion::telemetry(Mode::Streaming),
        crate::telemetry::GenAiOperation::ChatStreaming
    );
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

// ── the mode a decoder is built for ─────────────────────────────────────

/// A wire whose decoder runs the whole-reply guard the two real ones run
/// (`providers::openai::wire::chat`, `providers::gemini::streaming`): a
/// reply that delivered no content and named no terminal is the provider
/// answering with nothing when it arrived whole, and truncation when it was
/// streamed. The [`Mode`] it was built for is the only thing that tells the
/// two apart, so this wire is what pins that `call` and `stream` each hand
/// [`Wire::decoder`] the mode they actually are.
#[derive(Clone, Debug, PartialEq)]
struct Guarded(Framing);

struct GuardedDecoder {
    /// This reply arrives whole, so its EOF ends an answer.
    whole: bool,
}

impl Decoder<Completion> for GuardedDecoder {
    type Event = ();

    fn classify(&self, _frame: WireFrame) -> WireEvent<()> {
        WireEvent::Known(())
    }

    /// Every frame decodes and none of them delivers content: the state the
    /// guard exists for, reached without a second frame vocabulary.
    fn interpret(&mut self, _event: (), _out: &mut Output<Completion>) {}

    fn finish(&mut self, out: &mut Output<Completion>) {
        if self.whole {
            out.error(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
    }
}

impl Wire for Guarded {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder = GuardedDecoder;

    fn name(&self) -> &str {
        "guarded"
    }

    fn encode(&self, _request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = http::Request::post("https://echo.invalid/v1/messages")
            .body(Body::Bytes(Vec::new()))?;
        Ok(Encoded::new(request, self.0))
    }

    fn decoder(&self, mode: Mode) -> Self::Decoder {
        GuardedDecoder {
            whole: mode == Mode::Unary,
        }
    }
}

#[tokio::test]
async fn the_mode_a_decoder_is_built_for_decides_what_its_eof_means() {
    let unary = Model::new(Guarded(Framing::Whole), RecordingHttpClient::new("{}"));
    let error = unary
        .call(prompt())
        .await
        .expect_err("a whole reply that delivered nothing is not an answer");
    assert!(
        matches!(&error, ProviderError::Response(message)
            if message == crate::message::EMPTY_RESPONSE_ERROR),
        "expected the empty-reply error, got {error:?}"
    );

    let streaming = Model::new(
        Guarded(Framing::Sse),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(b"data: {}\n\n"),
        },
    );
    let mut response = streaming.stream(prompt()).expect("the stream opens");
    let mut errors = Vec::new();
    while let Some(item) = response.next().await {
        if let Err(error) = item {
            errors.push(error);
        }
    }
    assert!(
        errors.is_empty(),
        "a streamed reply's EOF is truncation — reported by carrying no \
         terminal record, not by an error: {errors:?}"
    );
}

/// A stream opens one byte-body request. A wire that encodes a batch or a
/// multipart body for a stream fails before anything is sent, as a request
/// that could not be built.
#[test]
fn a_stream_the_driver_cannot_send_is_a_request_failure() {
    #[derive(Clone)]
    struct Batch;
    impl Wire for Batch {
        type Op = Completion;
        type Payload = Encoded;
        type Frame = WireFrame;
        type Decoder = EchoDecoder;
        fn name(&self) -> &str {
            "echo"
        }
        fn encode(&self, _request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
            let one =
                || http::Request::post("https://echo.invalid/a").body(Body::Bytes(Vec::new()));
            let mut encoded = Encoded::new(one()?, Framing::Sse);
            encoded.requests.push(one()?);
            Ok(encoded)
        }
        fn decoder(&self, _mode: Mode) -> Self::Decoder {
            EchoDecoder
        }
    }
    #[derive(Clone)]
    struct Multipart;
    impl Wire for Multipart {
        type Op = Completion;
        type Payload = Encoded;
        type Frame = WireFrame;
        type Decoder = EchoDecoder;
        fn name(&self) -> &str {
            "echo"
        }
        fn encode(&self, _request: CompletionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
            let request = http::Request::post("https://echo.invalid/a")
                .body(Body::Multipart(crate::http_client::MultipartForm::new()))?;
            Ok(Encoded::new(request, Framing::Sse))
        }
        fn decoder(&self, _mode: Mode) -> Self::Decoder {
            EchoDecoder
        }
    }

    let http = crate::test_utils::RecordingHttpClient::new("");
    for (error, message) in [
        (
            stream(&Batch, &http, prompt(), None).err(),
            "RequestError: a streamed reply takes exactly one request, not 2",
        ),
        (
            stream(&Multipart, &http, prompt(), None).err(),
            "RequestError: a multipart request cannot open a streamed reply",
        ),
    ] {
        let error = error.expect("the stream must not open");
        assert_eq!(error.to_string(), message);
        assert_eq!(error.kind(), crate::error::ErrorKind::Request);
        assert_eq!(
            error.boundary(),
            crate::observe::AdapterErrorBoundary::Request
        );
        assert!(!error.is_retryable());
    }
}

// ── every operation streams ─────────────────────────────────────────────

/// The ids a listing folded to, in order.
fn ids(models: &ModelList) -> Vec<&str> {
    models.iter().map(|model| model.id.as_str()).collect()
}

/// Every page's events arrive on the stream, the cursor is followed as a
/// call follows it, and the finished stream is the call's response.
#[tokio::test]
async fn a_paged_listing_streamed_yields_every_page_and_finishes_to_what_call_gives() {
    let script = || {
        SequencedHttpClient::new([
            MockHttpResponse::success(r#"{"data":["a","b"],"next":"b"}"#),
            MockHttpResponse::success(r#"{"data":["c"]}"#),
        ])
    };
    let called = Model::new(Catalogue, script())
        .call(())
        .await
        .expect("both pages decode");

    let http = script();
    let mut stream = Model::new(Catalogue, http.clone())
        .stream(())
        .expect("the stream opens");
    let mut pages = Vec::new();
    while let Some(page) = stream.next().await {
        pages.push(page.expect("a page decodes"));
    }
    assert_eq!(
        pages.iter().map(ids).collect::<Vec<_>>(),
        vec![vec!["a", "b"], vec!["c"]],
        "each page is one event"
    );
    let streamed = stream.finish().expect("the pages fold");
    assert_eq!(ids(&streamed), ids(&called));
    assert_eq!(http.requests().len(), 2);
}

/// The cursor guards hold on a stream: a repeated cursor stops after the
/// repeated page and a cursor that keeps changing stops at the ceiling.
#[tokio::test]
async fn a_streamed_listing_keeps_the_cursor_guards() {
    let repeating = SequencedHttpClient::new([
        MockHttpResponse::success(r#"{"data":["a"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["b"],"next":"b"}"#),
        MockHttpResponse::success(r#"{"data":["never"]}"#),
    ]);
    let mut stream = Model::new(Catalogue, repeating.clone())
        .stream(())
        .expect("the stream opens");
    while stream.next().await.is_some() {}
    assert_eq!(
        ids(&stream.finish().expect("the pages fold")),
        vec!["a", "b"]
    );
    assert_eq!(
        repeating.requests().len(),
        2,
        "the repeated cursor is not re-sent"
    );

    let endless = SequencedHttpClient::new((0..super::MAX_CONTINUATION_PAGES + 5).map(|page| {
        MockHttpResponse::success(format!(r#"{{"data":["m{page}"],"next":"c{}"}}"#, page + 1))
    }));
    let mut stream = Model::new(Catalogue, endless.clone())
        .stream(())
        .expect("the stream opens");
    let mut pages = 0;
    while stream.next().await.is_some() {
        pages += 1;
    }
    assert_eq!(pages, super::MAX_CONTINUATION_PAGES);
    assert_eq!(endless.requests().len(), super::MAX_CONTINUATION_PAGES);
}

/// An operation whose reply is one whole document streams that document
/// as one event, and finishes to what the call gives.
#[tokio::test]
async fn an_embedding_streamed_yields_one_event_and_finishes_to_what_call_gives() {
    use crate::providers::openai::{OpenAI, TEXT_EMBEDDING_3_SMALL};

    const REPLY: &str = r#"{"object":"list","model":"text-embedding-3-small","data":[{"object":"embedding","index":0,"embedding":[0.5,0.25]}],"usage":{"prompt_tokens":1,"total_tokens":1}}"#;
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(REPLY),
        MockHttpResponse::success(REPLY),
    ]);
    let model = Model::new(
        OpenAI::new("test-key").embedding(TEXT_EMBEDDING_3_SMALL, None),
        http,
    );
    let called = model
        .call(vec!["hi".to_owned()])
        .await
        .expect("the reply decodes");

    let mut stream = model
        .stream(vec!["hi".to_owned()])
        .expect("the stream opens");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("the reply decodes"));
    }
    assert_eq!(events.len(), 1, "one whole reply is one event");
    let streamed = stream.finish().expect("the reply folds");
    assert_eq!(
        serde_json::to_value(&streamed).expect("json"),
        serde_json::to_value(&called).expect("json")
    );
}

// ── an operation the crate never heard of ───────────────────────────────

#[derive(Clone, Debug, PartialEq)]
struct VideoRequest {
    frames: usize,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
struct SkeletonFrame {
    index: usize,
    last: bool,
}

#[derive(Debug, Default, PartialEq, serde::Serialize)]
struct PoseTrack {
    frames: Vec<SkeletonFrame>,
}

struct PoseEstimation;

impl Operation for PoseEstimation {
    type Request = VideoRequest;
    type Event = SkeletonFrame;
    type Response = PoseTrack;
    type Capabilities = ();
    type Output = Events<Self>;
    type Fold = PoseTrack;
    type Telemetry = ();

    const NAME: &'static str = "pose_estimation";

    fn is_terminal(frame: &SkeletonFrame) -> bool {
        frame.last
    }

    fn fold<W: Wire<Op = Self>>(_: &VideoRequest, _: &W, _: Mode) -> PoseTrack {
        PoseTrack::default()
    }

    fn telemetry(_: Mode) {}
}

impl Fold<PoseEstimation> for PoseTrack {
    fn absorb(&mut self, frame: &SkeletonFrame) -> Result<(), ProviderError> {
        self.frames.push(frame.clone());
        Ok(())
    }

    fn finish(self, _: Reply) -> Result<PoseTrack, ProviderError> {
        Ok(self)
    }
}

/// The runtime behind the wire: one skeleton per frame of the video.
#[derive(Clone)]
struct PoseRuntime;

impl Transport<Local<PoseEstimation>> for PoseRuntime {
    fn send(
        &self,
        request: VideoRequest,
        _: Mode,
        _: Option<super::Observation>,
    ) -> Result<
        impl Future<Output = Opened<VideoRequest, SkeletonFrame>> + WasmCompatSend + 'static + use<>,
        ProviderError,
    > {
        let frames: Vec<_> = (0..request.frames)
            .map(|index| {
                Ok(SkeletonFrame {
                    index,
                    last: index + 1 == request.frames,
                })
            })
            .collect();
        Ok(async move { Opened::new(futures::stream::iter(frames)) })
    }
}

/// An operation defined outside the crate calls and streams through the one
/// driver, typed and erased, and the erased call can be spawned.
#[tokio::test]
async fn an_operation_the_crate_never_heard_of_calls_and_streams_typed_and_erased() {
    let model = Model::new(Local::<PoseEstimation>::new("pose"), PoseRuntime);
    let video = VideoRequest { frames: 3 };

    let called = model.call(video.clone()).await.expect("the track folds");
    assert_eq!(called.frames.len(), 3);
    assert!(called.frames.last().is_some_and(|frame| frame.last));

    let mut stream = model.stream(video.clone()).expect("the stream opens");
    let mut frames = Vec::new();
    while let Some(frame) = stream.next().await {
        frames.push(frame.expect("a skeleton"));
    }
    assert_eq!(frames, called.frames, "the stream yields every frame");
    assert_eq!(stream.finish().expect("the track folds"), called);

    let erased = model.erase();
    let spawned = tokio::spawn(erased.call(video.clone()))
        .await
        .expect("the spawned call runs")
        .expect("the track folds");
    assert_eq!(spawned, called);

    let mut stream = erased.stream(video).expect("the erased stream opens");
    let mut frames = Vec::new();
    while let Some(frame) = stream.next().await {
        frames.push(frame.expect("a skeleton"));
    }
    assert_eq!(frames, called.frames);
    assert_eq!(stream.finish().expect("the track folds"), called);
}
