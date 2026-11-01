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
use serde_json::json;

use super::{Model, Transport};
use crate::completion::CompletionRequest;
use crate::error::{EncodeError, ProviderError};
use crate::http_client::framing::Framing;
use crate::message::{CallId, ToolName};
use crate::observe::{AdapterContext, AdapterVerdict, ObservationLog, Subject};
use crate::operation::{Block, Completion, Finish};
use crate::streaming::Streamed;
use crate::test_utils::{
    MockHttpResponse, MockStreamingClient, NonSuccessStreamingClient, PacedStreamingClient,
    RecordingHttpClient, SequencedHttpClient, SilentStreamingClient, TraceCapture,
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

/// One observed reply of `wire` over `http` in `mode`: the usage
/// observations it emitted, in order, and the response's usage.
///
/// # Panics
///
/// When the reply does not fold.
pub(crate) async fn observed_usage<W, T>(
    wire: W,
    http: T,
    mode: crate::wire::Mode,
) -> (Vec<crate::observe::AdapterUsage>, crate::completion::Usage)
where
    W: crate::wire::Wire<Op = crate::operation::Completion>,
    T: Transport<W>,
{
    use crate::observe::{Action, AdapterContext, AdapterEvent, ObservationLog, Subject};
    let log = std::sync::Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "call");
    let model = Model::new(wire, http);
    let request = crate::completion::CompletionRequest::new("hi");
    let response = match mode {
        crate::wire::Mode::Unary => model.call_observed(request, context).await,
        crate::wire::Mode::Streaming => match model.stream_observed(request, context) {
            Ok(stream) => stream.finish().await,
            Err(error) => Err(error),
        },
    }
    .unwrap_or_else(|error| panic!("the reply folds: {error}"));
    let observed = log
        .trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::Adapter { observation } => match &observation.event {
                AdapterEvent::Usage { usage } => Some(usage.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    (observed, response.usage)
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
    /// The provider's in-band failure, ending the stream without an end.
    Error { message: String },
}

#[derive(Clone, Copy, serde::Deserialize)]
struct Usage {
    output_tokens: u64,
}

#[derive(Default)]
struct EchoDecoder<'id> {
    brand: std::marker::PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id> Decoder<'id, Completion> for EchoDecoder<'id> {
    type Event = Frame;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_tagged_frame(&frame.as_str(), "type", |kind| {
            matches!(kind, "message" | "delta" | "stop" | "tool" | "error")
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
                out.run(Block::Text, &text)?;
                return self.end(out, usage);
            }
            Frame::Delta { text } => {
                out.run(Block::Text, &text)?;
            }
            Frame::Stop { usage } => return self.end(out, usage),
            Frame::Error { message } => return Err(ProviderError::Provider(message)),
            Frame::Tool {
                name,
                arguments,
                usage,
            } => {
                out.end_run()?;
                let name = ToolName::new(name)
                    .map_err(|error| ProviderError::Response(error.to_string()))?;
                let index = out.fresh_index();
                out.whole(
                    index,
                    Block::Call {
                        id: CallId::from_wire("call_1"),
                        name,
                    },
                    serde_json::Value::Null,
                    &arguments,
                )?;
                if let Some(usage) = usage {
                    return self.end(out, usage);
                }
            }
        }
        Ok(Flow::More)
    }
}

impl<'id> EchoDecoder<'id> {
    fn end(&mut self, mut out: Out<'id, Completion>, usage: Usage) -> Result<Flow, ProviderError> {
        out.end_run()?;
        Ok(out.end(Finish {
            usage: crate::completion::Usage {
                output_tokens: Some(usage.output_tokens),
                ..crate::completion::Usage::default()
            },
            model: Some("echo-1".to_owned()),
            ..Finish::default()
        }))
    }
}

impl EchoDecoder<'_> {
    fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        if serde_json::from_slice::<serde_json::Value>(payload).is_err() {
            return;
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

impl crate::completion::ReplayTarget for Echo {
    fn map_options(
        &self,
        _request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::test_utils::refuse_options(fields)
    }

    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("echo.chat")
    }

    fn provider(&self) -> &str {
        "echo"
    }

    fn model(&self) -> &str {
        "echo-1"
    }

    fn accepts(&self, _model: &str) -> crate::completion::Accepts {
        crate::completion::Accepts::ALL
    }
}

/// The unary `message` document the fake wire's frames add up to: delta
/// texts append, a stop gives the usage, a whole message or call is kept.
#[derive(Default)]
struct EchoDocument(Option<serde_json::Map<String, serde_json::Value>>);

impl crate::wire::document::Serves<Completion> for EchoDocument {}

impl crate::wire::document::Reassemble<Frame> for EchoDocument {
    fn absorb(&mut self, frame: &Frame) {
        let document = self.0.get_or_insert_with(|| {
            serde_json::Map::from_iter([
                ("type".to_owned(), json!("message")),
                ("text".to_owned(), json!("")),
            ])
        });
        let usage = |usage: &Usage| json!({"output_tokens": usage.output_tokens});
        match frame {
            Frame::Message { text, usage: used } => {
                *document = serde_json::Map::from_iter([
                    ("type".to_owned(), json!("message")),
                    ("text".to_owned(), json!(text)),
                    ("usage".to_owned(), usage(used)),
                ]);
            }
            Frame::Delta { text } => {
                if let Some(serde_json::Value::String(held)) = document.get_mut("text") {
                    held.push_str(text);
                }
            }
            Frame::Stop { usage: used } => {
                document.insert("usage".to_owned(), usage(used));
            }
            Frame::Tool {
                name,
                arguments,
                usage: used,
            } => {
                document.insert("name".to_owned(), json!(name));
                document.insert("arguments".to_owned(), json!(arguments));
                if let Some(used) = used {
                    document.insert("usage".to_owned(), usage(used));
                }
            }
            Frame::Error { .. } => {}
        }
    }

    fn finish(self) -> serde_json::Value {
        self.0
            .map_or(serde_json::Value::Null, serde_json::Value::Object)
    }
}

impl Wire for Echo {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Event = Frame;
    type Decoder<'id> = EchoDecoder<'id>;
    type Reassembler = EchoDocument;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("echo").model("echo-1").replay(self)
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
    CompletionRequest::new("say hi")
}

// ── the property the model exists for ───────────────────────────────────

const UNARY_BODY: &str = r#"{"type":"message","text":"hi there","usage":{"output_tokens":3}}"#;

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

// ── streaming semantics ─────────────────────────────────────────────────

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

/// A transport whose reply sends `frames`, then fails.
#[derive(Clone)]
struct Broken(Vec<&'static str>);

impl Transport<Echo> for Broken {
    fn send(&self, _payload: Encoded, _exchange: super::Exchange) -> super::Opening<WireFrame> {
        let frames = self
            .0
            .iter()
            .map(|frame| Ok(WireFrame::Text((*frame).to_owned())))
            .chain([Err(ProviderError::Response(
                "the connection dropped".to_owned(),
            ))])
            .collect::<Vec<_>>();
        super::Opening::ready(super::Opened::new(futures::stream::iter(frames)))
    }
}

/// A transport whose reply sends `frames`, each `gap` after the last, then
/// sends nothing more, without ending.
#[derive(Clone)]
struct Paced {
    frames: Vec<&'static str>,
    gap: std::time::Duration,
}

impl Transport<Echo> for Paced {
    fn send(&self, _payload: Encoded, _exchange: super::Exchange) -> super::Opening<WireFrame> {
        let gap = self.gap;
        let frames = futures::stream::iter(self.frames.clone())
            .then(move |frame| async move {
                tokio::time::sleep(gap).await;
                Ok(WireFrame::Text(frame.to_owned()))
            })
            .chain(futures::stream::pending());
        super::Opening::ready(super::Opened::new(frames))
    }
}

/// A streamed reply that keeps sending within the idle timeout ends as the
/// provider ends it however long it takes; one that falls silent fails as
/// a retryable stall.
#[tokio::test]
async fn a_streamed_reply_that_falls_silent_stalls() {
    let idle = std::time::Duration::from_millis(200);
    let delta = r#"{"type":"delta","text":"hi"}"#;
    let stop = r#"{"type":"stop","usage":{"output_tokens":1}}"#;
    let steady = Paced {
        frames: vec![delta, delta, delta, delta, stop],
        gap: idle / 4,
    };
    let response = Model::new(Echo::streaming(), steady)
        .with_idle_timeout(Some(idle))
        .stream(prompt())
        .expect("the stream opens")
        .finish()
        .await;
    assert!(response.is_ok(), "a steady reply ends: {response:?}");

    let silent = Paced {
        frames: vec![delta],
        gap: idle / 4,
    };
    let stalled = Model::new(Echo::streaming(), silent)
        .with_idle_timeout(Some(idle))
        .stream(prompt())
        .expect("the stream opens")
        .finish()
        .await;
    match stalled {
        Err(error @ ProviderError::Stalled(after)) => {
            assert_eq!(after, idle);
            assert!(error.is_retryable(), "a stall is retried");
        }
        other => panic!("a silent reply stalls: {other:?}"),
    }
}

/// An event stream that sends only comments, as a provider pings while it
/// sends nothing, stalls an idle timeout after it opened, however often it
/// pings.
#[tokio::test]
async fn comments_alone_stall_a_streamed_reply() {
    let idle = std::time::Duration::from_millis(400);
    let pinging = PacedStreamingClient {
        chunks: vec![": keep-alive\n\n"; 64],
        gap: idle / 8,
    };
    let started = std::time::Instant::now();
    let stalled = Model::new(Echo::streaming(), pinging)
        .with_idle_timeout(Some(idle))
        .stream(prompt())
        .expect("the stream opens")
        .finish()
        .await;
    let took = started.elapsed();
    assert!(
        matches!(stalled, Err(ProviderError::Stalled(after)) if after == idle),
        "a reply of comments stalls: {stalled:?}"
    );
    assert!(took < idle * 2, "stalled {took:?} after it opened");
}

/// A reply that sends a frame now and then, each within the idle timeout,
/// never stalls, however long it runs in all and whatever comes between.
#[tokio::test]
async fn frames_within_the_idle_timeout_never_stall() {
    let idle = std::time::Duration::from_millis(200);
    let mut chunks = Vec::new();
    for _ in 0..4 {
        chunks.extend([": keep-alive\n\n", ": keep-alive\n\n"]);
        chunks.push("data: {\"type\":\"delta\",\"text\":\"hi\"}\n\n");
    }
    chunks.push("data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":1}}\n\n");
    let trickle = PacedStreamingClient {
        chunks,
        gap: idle / 4,
    };
    let response = Model::new(Echo::streaming(), trickle)
        .with_idle_timeout(Some(idle))
        .stream(prompt())
        .expect("the stream opens")
        .finish()
        .await;
    assert!(response.is_ok(), "a trickling reply ends: {response:?}");
}

/// A reader that pauses between frames does not move the stall: it comes
/// the idle timeout after the last bytes, however long the reader took.
#[tokio::test]
async fn a_slow_reader_stalls_an_idle_timeout_after_the_last_bytes() {
    let idle = std::time::Duration::from_millis(400);
    let once = PacedStreamingClient {
        chunks: vec!["data: {\"type\":\"delta\",\"text\":\"hi\"}\n\n"],
        gap: std::time::Duration::ZERO,
    };
    let mut stream = Model::new(Echo::streaming(), once)
        .with_idle_timeout(Some(idle))
        .stream(prompt())
        .expect("the stream opens");
    let started = std::time::Instant::now();
    assert!(
        matches!(stream.next().await, Some(Ok(_))),
        "the delta comes"
    );
    tokio::time::sleep(idle * 3 / 4).await;
    let mut last = None;
    while let Some(item) = stream.next().await {
        last = Some(item);
    }
    let took = started.elapsed();
    assert!(
        matches!(last, Some(Err(ProviderError::Stalled(_)))),
        "a silent reply stalls: {last:?}"
    );
    // The bytes came at once, so the stall is due at about 400 ms; timing
    // from when the reader came back would take 700 ms.
    assert!(took < idle + idle / 2, "stalled {took:?} after the bytes");
}

/// An idle timeout longer than a browser's timer takes waits indefinitely,
/// where a browser would otherwise fire it at once.
#[tokio::test]
async fn an_idle_timeout_past_the_timer_waits_indefinitely() {
    let delta = r#"{"type":"delta","text":"hi"}"#;
    let stop = r#"{"type":"stop","usage":{"output_tokens":1}}"#;
    let steady = Paced {
        frames: vec![delta, stop],
        gap: std::time::Duration::from_millis(10),
    };
    let response = Model::new(Echo::streaming(), steady)
        .with_idle_timeout(Some(std::time::Duration::MAX))
        .stream(prompt())
        .expect("the stream opens")
        .finish()
        .await;
    assert!(response.is_ok(), "the reply ends: {response:?}");
}

/// A streamed reply that never opens stalls too, and its attempt ends on
/// the stall, not as dropped.
#[tokio::test]
async fn a_reply_that_never_opens_ends_its_attempt_on_the_stall() {
    use crate::observe::{Action, AdapterEnding, AdapterEvent};
    let idle = std::time::Duration::from_millis(100);
    let (log, context) = observed();
    let stalled = match Model::new(Echo::streaming(), SilentStreamingClient)
        .with_idle_timeout(Some(idle))
        .stream_observed(prompt(), context)
    {
        Ok(stream) => stream.finish().await,
        Err(error) => Err(error),
    };
    assert!(
        matches!(stalled, Err(ProviderError::Stalled(after)) if after == idle),
        "a reply that never opens stalls: {stalled:?}"
    );
    let endings: Vec<AdapterEnding> = log
        .trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::Adapter { observation } => match &observation.event {
                AdapterEvent::Finished { ending } => Some(ending.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(
        matches!(
            endings.as_slice(),
            [AdapterEnding::Error {
                retryable: true,
                ..
            }]
        ),
        "the attempt ends once, on the stall: {endings:?}"
    );
}

/// A streamed reply's `raw` is what its reassembler rebuilt: the whole
/// document at the provider's end, and the document so far on a stream cut
/// short or failed by its transport, which `partial()` then carries.
#[tokio::test]
async fn a_streamed_reply_records_its_document_whole_cut_or_failed() {
    let delta = r#"{"type":"delta","text":"hi "}"#;
    let whole = MockStreamingClient {
        sse_bytes: Bytes::from(format!(
            "data: {delta}\n\ndata: {{\"type\":\"delta\",\"text\":\"there\"}}\n\n\
             data: {{\"type\":\"stop\",\"usage\":{{\"output_tokens\":3}}}}\n\n"
        )),
    };
    let response = Model::new(Echo::streaming(), whole)
        .stream(prompt())
        .expect("the stream opens")
        .finish()
        .await
        .expect("the reply ends");
    assert_eq!(
        response.raw,
        serde_json::from_str::<serde_json::Value>(UNARY_BODY).expect("the unary body is JSON"),
        "the stream rebuilds the unary document"
    );

    let so_far = json!({"type": "message", "text": "hi "});
    let cut = MockStreamingClient {
        sse_bytes: Bytes::from(format!("data: {delta}\n\n")),
    };
    let mut stream = Model::new(Echo::streaming(), cut)
        .stream(prompt())
        .expect("the stream opens");
    while stream.next().await.is_some() {}
    assert_eq!(
        stream.partial().raw,
        so_far,
        "a cut stream keeps the document so far"
    );

    let mut stream = Model::new(Echo::streaming(), Broken(vec![delta]))
        .stream(prompt())
        .expect("the stream opens");
    let items: Vec<_> = (&mut stream).collect().await;
    assert!(
        items.last().is_some_and(Result::is_err),
        "the transport failure ends the stream: {items:?}"
    );
    assert_eq!(
        stream.partial().raw,
        so_far,
        "a failed stream keeps the document so far"
    );
}

/// A caller that stops polling leaves the reassembler unfinished, so the
/// reply records no `raw`.
#[tokio::test]
async fn a_stream_the_caller_stopped_polling_records_no_raw() {
    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(
            b"data: {\"type\":\"delta\",\"text\":\"hi \"}\n\n\
              data: {\"type\":\"delta\",\"text\":\"there\"}\n\n",
        ),
    };
    let mut stream = Model::new(Echo::streaming(), http)
        .stream(prompt())
        .expect("the stream opens");
    let _first = stream.next().await;
    assert!(stream.partial().raw.is_null());
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

/// A tool call whose input is not JSON never fails the reply: the call is
/// kept with its raw text, in both modes.
#[tokio::test]
async fn a_malformed_tool_input_is_kept_with_its_raw_text() {
    let http = RecordingHttpClient::new(
        r#"{"type":"tool","name":"add","arguments":"{not json","usage":{"output_tokens":1}}"#,
    );
    let response = call(&Echo::unary(), &http, prompt(), None)
        .await
        .expect("a malformed call does not fail the reply");
    let call_of = |response: &crate::completion::CompletionResponse| {
        response
            .tool_calls()
            .map(|call| {
                (
                    call.function.arguments_value(),
                    call.function.invalid_arguments.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        call_of(&response),
        [(serde_json::json!({}), Some("{not json".to_owned()))]
    );

    let http = MockStreamingClient {
        sse_bytes: Bytes::from_static(
            b"data: {\"type\":\"tool\",\"name\":\"add\",\"arguments\":\"{not json\"}\n\n\
              data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":1}}\n\n",
        ),
    };
    let streamed = Model::new(Echo::streaming(), http)
        .stream(prompt())
        .expect("the stream opens")
        .finish()
        .await
        .expect("a malformed call does not fail the stream");
    assert_eq!(call_of(&streamed), call_of(&response));
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
        events.contains(&"provider".to_owned()),
        "the failed reply's facts are still projected: {events:?}"
    );
    assert_eq!(events.last().map(String::as_str), Some("finished"));
}

/// The driver observes a reply's usage once, when it ends, from the usage
/// the decoder normalized into its end: unary or streamed, it is the
/// response's usage, with no invented total.
#[tokio::test]
async fn a_reply_observes_its_finish_usage_once() {
    let streamed = MockStreamingClient {
        sse_bytes: Bytes::from_static(
            b"data: {\"type\":\"delta\",\"text\":\"hi\"}\n\n\
              data: {\"type\":\"stop\",\"usage\":{\"output_tokens\":3}}\n\n",
        ),
    };
    for (observed, usage) in [
        crate::driver::tests::observed_usage(
            Echo::unary(),
            RecordingHttpClient::new(UNARY_BODY),
            Mode::Unary,
        )
        .await,
        crate::driver::tests::observed_usage(Echo::streaming(), streamed, Mode::Streaming).await,
    ] {
        let expected = crate::observe::AdapterUsage {
            output_tokens: Some(3),
            ..Default::default()
        };
        assert_eq!(observed, std::slice::from_ref(&expected));
        assert_eq!(expected, crate::observe::AdapterUsage::from(&usage));
    }
}

/// A stream that carries usage and then fails has no end, so it observes
/// no usage, and its attempt closes on the failure, not as `Terminal`.
/// Before usage was observed from the end, the raw-bytes projector reported
/// the failing stream's usage frame.
#[tokio::test]
async fn a_failed_stream_observes_no_usage() {
    use crate::observe::{Action, AdapterEnding, AdapterEvent};
    // Each frame carries usage the old projector would have reported.
    let usage_frame =
        "data: {\"type\":\"delta\",\"text\":\"hi\",\"usage\":{\"output_tokens\":3}}\n\n";
    let in_band =
        format!("{usage_frame}data: {{\"type\":\"error\",\"message\":\"overloaded\"}}\n\n");
    for (sse, ended) in [
        (
            in_band,
            (|ending: &AdapterEnding| matches!(ending, AdapterEnding::Error { .. }))
                as fn(&AdapterEnding) -> bool,
        ),
        (usage_frame.to_owned(), |ending| {
            matches!(ending, AdapterEnding::Eof { after: 1 })
        }),
    ] {
        let (log, context) = observed();
        let http = MockStreamingClient {
            sse_bytes: Bytes::from(sse),
        };
        let result = match Model::new(Echo::streaming(), http).stream_observed(prompt(), context) {
            Ok(stream) => stream.finish().await,
            Err(error) => Err(error),
        };
        assert!(result.is_err(), "the stream fails");
        let adapter: Vec<AdapterEvent> = log
            .trace()
            .observations
            .iter()
            .filter_map(|observation| match &observation.action {
                Action::Adapter { observation } => Some(observation.event.clone()),
                _ => None,
            })
            .collect();
        assert!(
            !adapter
                .iter()
                .any(|event| matches!(event, AdapterEvent::Usage { .. })),
            "a failed stream observes no usage: {adapter:?}"
        );
        let endings: Vec<&AdapterEnding> = adapter
            .iter()
            .filter_map(|event| match event {
                AdapterEvent::Finished { ending } => Some(ending),
                _ => None,
            })
            .collect();
        assert!(
            matches!(endings.as_slice(), [ending] if ended(ending)),
            "the attempt closes once, on the failure: {endings:?}"
        );
    }
}

// ── paging ─────────────────────────────────────────────────────────────

// ── telemetry ──────────────────────────────────────────────────────────

#[tokio::test]
async fn the_driver_records_the_folded_responses_metadata() {
    use tracing::subscriber::with_default;

    let capture = TraceCapture::default();
    let bound = Model::new(Echo::unary(), RecordingHttpClient::new(UNARY_BODY));
    let response = with_default(capture.subscriber(), || {
        futures::executor::block_on(bound.call(prompt()))
    })
    .expect("the reply decodes");
    assert_eq!(response.model(), Some("echo-1"));
    assert!(
        capture
            .values_of("gen_ai.response.model")
            .contains(&json!("echo-1")),
        "expected the response model on the span"
    );
    assert!(
        !capture.values_of("gen_ai.usage.output_tokens").is_empty(),
        "expected usage on the span"
    );
}

/// A streamed embedding records its response on the span when it
/// finishes, as a call does.
#[test]
fn a_streamed_embedding_records_its_response_on_the_span() {
    use tracing::subscriber::with_default;

    const REPLY: &str = r#"{"object":"list","model":"text-embedding-3-small","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"usage":{"prompt_tokens":2,"total_tokens":2}}"#;
    let capture = TraceCapture::default();
    let wire = crate::providers::openai::wire::OpenAIConfig::new("sk-test")
        .embedding("text-embedding-3-small", None);
    let model = Model::new(
        wire,
        SequencedHttpClient::new([MockHttpResponse::success(REPLY)]),
    );
    with_default(capture.subscriber(), || {
        let stream = model
            .stream(vec!["a".to_owned()])
            .expect("the embedding opens");
        futures::executor::block_on(stream.finish()).expect("the reply folds")
    });
    assert!(
        capture
            .values_of("gen_ai.response.model")
            .contains(&json!("text-embedding-3-small")),
        "expected the response model on the span"
    );
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
    #[derive(Clone, Debug)]
    struct Multipart;
    impl crate::completion::ReplayTarget for Multipart {
        fn map_options(
            &self,
            _request: &crate::completion::CompletionRequest,
            fields: crate::completion::options::OptionFields<'_>,
        ) -> crate::completion::options::OptionMap {
            crate::test_utils::refuse_options(fields)
        }

        fn api(&self) -> crate::message::Api {
            crate::message::Api::from_static("echo.chat")
        }
        fn provider(&self) -> &str {
            "echo"
        }
        fn model(&self) -> &str {
            ""
        }

        fn accepts(&self, _model: &str) -> crate::completion::Accepts {
            crate::completion::Accepts::ALL
        }
    }
    impl Wire for Multipart {
        type Op = Completion;
        type Payload = Encoded;
        type Frame = WireFrame;
        type Event = Frame;
        type Decoder<'id> = EchoDecoder<'id>;
        type Reassembler = EchoDocument;
        fn describe(&self) -> Descriptor<'_> {
            Descriptor::new("echo").replay(self)
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

// ── request validation ─────────────────────────────────────────────────

/// An erased model observed under a context records the adapter events the
/// model it was made from records, called or streamed.
#[tokio::test]
async fn an_erased_model_is_observed_as_the_model_it_was_made_from() {
    let reply = r#"{"type":"text","text":"hi","usage":{"output_tokens":1}}"#;
    let direct = {
        let (log, context) = observed();
        let _ = Model::new(Echo::unary(), RecordingHttpClient::new(reply))
            .call_observed(prompt(), context)
            .await;
        events(&log)
    };
    let erased = {
        let (log, context) = observed();
        let _ = Model::new(Echo::unary(), RecordingHttpClient::new(reply))
            .erase()
            .call_observed(prompt(), context)
            .await;
        events(&log)
    };
    assert!(!direct.is_empty(), "the direct call is observed");
    assert_eq!(direct, erased);

    let streamed = |erase: bool| async move {
        let (log, context) = observed();
        let model = Model::new(Echo::streaming(), RecordingHttpClient::new(reply));
        let opened = if erase {
            model.erase().stream_observed(prompt(), context)
        } else {
            model.stream_observed(prompt(), context)
        };
        if let Ok(stream) = opened {
            let _ = stream.finish().await;
        }
        events(&log)
    };
    let direct = streamed(false).await;
    assert!(!direct.is_empty(), "the direct stream is observed");
    assert_eq!(direct, streamed(true).await);
}
