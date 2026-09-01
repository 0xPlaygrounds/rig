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

fn data_chunk(events: &[&str]) -> Bytes {
    Bytes::from(
        events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
    )
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
        vec![Ok(data_chunk(&["one", " ", "two"]))],
        default_options(),
        skip_blank_frames,
    )
    .await;

    assert_eq!(text_frames(&frames), vec!["one", "two"]);
}

#[tokio::test]
async fn stream_ended_is_a_normal_end_by_default() {
    let frames = collect_frames(
        vec![Ok(data_chunk(&["one"])), Err(StubError::StreamEnded)],
        default_options(),
        skip_blank_frames,
    )
    .await;

    assert_eq!(text_frames(&frames), vec!["one"]);
}

#[tokio::test]
async fn stream_ended_surfaces_as_error_when_flagged() {
    let frames = collect_frames(
        vec![Ok(data_chunk(&["one"])), Err(StubError::StreamEnded)],
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
        vec![Ok(data_chunk(&["one", "boom", "after"]))],
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
