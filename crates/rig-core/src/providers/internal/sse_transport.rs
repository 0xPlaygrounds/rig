//! The shared SSE transport preamble: `GenericEventSource` events →
//! [`WireFrame`]s.
//!
//! Every SSE-backed streaming wire opens with the same loop — log the `Open`
//! event, skip payload-less heartbeat frames, yield each `data:` payload as a
//! text frame, end on `StreamEnded`, and surface transport errors in-band —
//! so the loop is stated once here. Byte splitting and framing only:
//! classification and policy live downstream in the wire adapters and
//! [`run_wire_stream`](super::adapter::run_wire_stream).
//!
//! The per-provider deltas are the variation points: the `Open` log level
//! ([`OpenLog`]), whether `StreamEnded` is a normal end or an error
//! (Anthropic's historical behavior), whether transport errors are logged,
//! and a per-frame triage closure that owns heartbeat/`[DONE]` filtering and
//! any in-band provider-error pre-filter.

use async_stream::stream;
use futures::{Stream, StreamExt};
use tracing_futures::Instrument;

use super::adapter::{WireAdapter, WireFrame, run_wire_stream};
use crate::completion::CompletionError;
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::wasm_compat::WasmCompatSend;

/// How the transport logs the SSE `Open` event (a per-provider historical
/// delta, preserved exactly).
#[derive(Clone, Copy)]
pub(crate) enum OpenLog {
    Silent,
    Trace,
    Debug,
}

/// One frame's disposition, decided by the caller's triage closure.
pub(crate) enum FrameDisposition {
    /// Not a wire frame (heartbeat, `[DONE]` on wires that drop it): skip.
    Skip,
    /// A payload frame: yield it as [`WireFrame::Text`].
    Frame(String),
    /// An in-band terminal provider error (the wire's error envelope,
    /// detected pre-classification exactly as an HTTP failure would be):
    /// yield the error and end the transport.
    Fail(CompletionError),
}

/// Skip-blank triage shared by wires with no `[DONE]` sentinel and no
/// in-band error envelope: heartbeats carry no payload and are not wire
/// frames; everything else passes through untrimmed.
pub(crate) fn skip_blank_frames(data: String) -> FrameDisposition {
    if data.trim().is_empty() {
        FrameDisposition::Skip
    } else {
        FrameDisposition::Frame(data)
    }
}

/// Triage shared by wires whose heartbeats and `[DONE]` sentinel are both
/// dropped at the transport: trim the payload, skip blanks and `[DONE]`,
/// yield everything else trimmed.
pub(crate) fn skip_blank_and_done(data: String) -> FrameDisposition {
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        FrameDisposition::Skip
    } else {
        FrameDisposition::Frame(data.to_owned())
    }
}

/// The per-wire transport deltas.
pub(crate) struct SseTransportOptions {
    pub open_log: OpenLog,
    /// `false`: `StreamEnded` is the normal end of the stream (break).
    /// `true` (Anthropic): `StreamEnded` maps through
    /// [`CompletionError::from_stream_transport`] like any other transport
    /// error — its historical loop had no separate `StreamEnded` arm.
    pub stream_ended_is_error: bool,
    /// Whether transport errors are logged (`error!(?error, "SSE error")`)
    /// before being yielded in-band. Anthropic historically did not log.
    pub log_transport_errors: bool,
}

/// Run the SSE transport preamble: drain `event_source` into a stream of
/// [`WireFrame`]s for [`run_wire_stream`](super::adapter::run_wire_stream),
/// closing the event source when the loop ends.
pub(crate) fn sse_frames<HttpClient, RequestBody, F>(
    event_source: GenericEventSource<HttpClient, RequestBody>,
    options: SseTransportOptions,
    mut triage: F,
) -> impl Stream<Item = Result<WireFrame, CompletionError>>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
    F: FnMut(String) -> FrameDisposition + WasmCompatSend + 'static,
{
    stream! {
        let mut event_source = Box::pin(event_source);
        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => match options.open_log {
                    OpenLog::Silent => {}
                    OpenLog::Trace => tracing::trace!("SSE connection opened"),
                    OpenLog::Debug => tracing::debug!("SSE connection opened"),
                },
                Ok(Event::Message(message)) => match triage(message.data) {
                    FrameDisposition::Skip => {}
                    FrameDisposition::Frame(data) => yield Ok(WireFrame::Text(data)),
                    FrameDisposition::Fail(error) => {
                        yield Err(error);
                        break;
                    }
                },
                Err(crate::http_client::Error::StreamEnded)
                    if !options.stream_ended_is_error =>
                {
                    break;
                }
                Err(error) => {
                    if options.log_transport_errors {
                        tracing::error!(?error, "SSE error");
                    }
                    yield Err(CompletionError::from_stream_transport(error));
                    break;
                }
            }
        }
        // Ensure event source is closed when stream ends
        event_source.close();
    }
}

/// Open an SSE-backed wire stream: build the event source, run the transport
/// preamble ([`sse_frames`]) with the wire's options and triage, and drive the
/// frames through the shared adapter driver
/// ([`run_wire_stream`](super::adapter::run_wire_stream)) under `span`.
pub(crate) fn open_wire_stream<HttpClient, RequestBody, A, F>(
    event_source: GenericEventSource<HttpClient, RequestBody>,
    options: SseTransportOptions,
    triage: F,
    adapter: A,
    span: tracing::Span,
) -> crate::streaming::RawStreamingResult<A::Response>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
    A: WireAdapter<Frame = WireFrame> + WasmCompatSend + 'static,
    A::Event: WasmCompatSend,
    A::Response: WasmCompatSend + 'static,
    F: FnMut(String) -> FrameDisposition + WasmCompatSend + 'static,
{
    // Transport layer: SSE events → `WireFrame`s. Byte splitting, framing,
    // and any in-band provider-error pre-filter carried by `triage` —
    // classification and policy live downstream.
    let transport = sse_frames(event_source, options, triage);

    Box::pin(run_wire_stream(transport, adapter).instrument(span))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_client::{self, sse::BoxedStream};
    use bytes::Bytes;
    use futures::StreamExt;
    use std::future::{self, Future};

    /// A streaming mock that yields the given chunk results in order, so
    /// transport errors (e.g. `StreamEnded`) can be injected mid-stream.
    #[derive(Clone)]
    struct ChunkedStreamingClient {
        chunks: Vec<Result<Bytes, StubError>>,
    }

    /// `http_client::Error` isn't `Clone`; carry a cloneable discriminant.
    #[derive(Clone)]
    enum StubError {
        StreamEnded,
    }

    impl http_client::HttpClientExt for ChunkedStreamingClient {
        fn send<T, U>(
            &self,
            _req: http::Request<T>,
        ) -> impl Future<Output = http_client::Result<http::Response<http_client::LazyBody<U>>>>
        + WasmCompatSend
        + 'static
        where
            T: Into<Bytes> + WasmCompatSend,
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            future::ready(Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::NOT_IMPLEMENTED,
            )))
        }

        fn send_multipart<U>(
            &self,
            _req: http::Request<http_client::MultipartForm>,
        ) -> impl Future<Output = http_client::Result<http::Response<http_client::LazyBody<U>>>>
        + WasmCompatSend
        + 'static
        where
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            future::ready(Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::NOT_IMPLEMENTED,
            )))
        }

        fn send_streaming<T>(
            &self,
            _req: http::Request<T>,
        ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>> + WasmCompatSend
        where
            T: Into<Bytes> + WasmCompatSend,
        {
            let chunks = self.chunks.clone();
            async move {
                let byte_stream = futures::stream::iter(chunks.into_iter().map(|chunk| {
                    chunk.map_err(|StubError::StreamEnded| http_client::Error::StreamEnded)
                }));
                let boxed_stream: BoxedStream = Box::pin(byte_stream);

                http::Response::builder()
                    .status(http::StatusCode::OK)
                    .header(http::header::CONTENT_TYPE, "text/event-stream")
                    .body(boxed_stream)
                    .map_err(http_client::Error::Protocol)
            }
        }
    }

    async fn collect_frames(
        chunks: Vec<Result<Bytes, StubError>>,
        options: SseTransportOptions,
        triage: impl FnMut(String) -> FrameDisposition + WasmCompatSend + 'static,
    ) -> Vec<Result<WireFrame, CompletionError>> {
        let req = http::Request::builder()
            .method(http::Method::POST)
            .uri("https://example.com/stream")
            .body(Vec::<u8>::new())
            .expect("request should build");
        let event_source = GenericEventSource::new(ChunkedStreamingClient { chunks }, req);
        sse_frames(event_source, options, triage).collect().await
    }

    fn data_chunk(events: &[&str]) -> Result<Bytes, StubError> {
        Ok(Bytes::from(
            events
                .iter()
                .map(|event| format!("data: {event}\n\n"))
                .collect::<String>(),
        ))
    }

    fn text_frames(frames: &[Result<WireFrame, CompletionError>]) -> Vec<String> {
        frames
            .iter()
            .map(|frame| match frame {
                Ok(WireFrame::Text(text)) => text.clone(),
                other => panic!("expected text frame, got {other:?}"),
            })
            .collect()
    }

    fn default_options() -> SseTransportOptions {
        SseTransportOptions {
            open_log: OpenLog::Silent,
            stream_ended_is_error: false,
            log_transport_errors: false,
        }
    }

    #[tokio::test]
    async fn yields_frames_and_skips_blanks() {
        let frames = collect_frames(
            vec![data_chunk(&["one", " ", "two"])],
            default_options(),
            skip_blank_frames,
        )
        .await;

        assert_eq!(text_frames(&frames), vec!["one", "two"]);
    }

    #[tokio::test]
    async fn stream_ended_is_a_normal_end_by_default() {
        let frames = collect_frames(
            vec![data_chunk(&["one"]), Err(StubError::StreamEnded)],
            default_options(),
            skip_blank_frames,
        )
        .await;

        assert_eq!(text_frames(&frames), vec!["one"]);
    }

    #[tokio::test]
    async fn stream_ended_surfaces_as_error_when_flagged() {
        let frames = collect_frames(
            vec![data_chunk(&["one"]), Err(StubError::StreamEnded)],
            SseTransportOptions {
                stream_ended_is_error: true,
                ..default_options()
            },
            skip_blank_frames,
        )
        .await;

        assert_eq!(frames.len(), 2, "frame then error, got {frames:?}");
        assert!(matches!(frames[0], Ok(WireFrame::Text(_))));
        assert!(frames[1].is_err(), "StreamEnded must surface in-band");
    }

    #[tokio::test]
    async fn fail_disposition_yields_error_and_ends_the_stream() {
        let frames = collect_frames(
            vec![data_chunk(&["one", "boom", "after"])],
            default_options(),
            |data| {
                if data == "boom" {
                    FrameDisposition::Fail(CompletionError::ProviderError("boom".into()))
                } else {
                    FrameDisposition::Frame(data)
                }
            },
        )
        .await;

        assert_eq!(frames.len(), 2, "frame then error, got {frames:?}");
        assert!(matches!(frames[0], Ok(WireFrame::Text(ref t)) if t == "one"));
        assert!(frames[1].is_err());
    }
}
