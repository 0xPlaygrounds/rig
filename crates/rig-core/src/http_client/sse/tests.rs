use super::*;
use crate::http_client::{self, HttpClientExt};
use futures::StreamExt;
use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Arc, Mutex};

/// One scripted connection: `Err` to fail the connect outright, else the
/// request-id header value and body chunks the connection delivers.
type ScriptedConnection =
    Result<(Option<&'static str>, Vec<StreamResult<Bytes>>), http_client::Error>;

/// Scripted connection outcomes: each `send_streaming` call pops one
/// [`ScriptedConnection`].
#[derive(Clone)]
struct SequencedStreamingClient {
    connections: Arc<Mutex<VecDeque<ScriptedConnection>>>,
}

impl SequencedStreamingClient {
    fn new(connections: impl IntoIterator<Item = ScriptedConnection>) -> Self {
        Self {
            connections: Arc::new(Mutex::new(connections.into_iter().collect())),
        }
    }
}

impl HttpClientExt for SequencedStreamingClient {
    fn send<T, U>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<http_client::LazyBody<U>>>>
    + WasmCompatSend
    + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::InvalidStatusCode(
            StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_multipart<U>(
        &self,
        _req: Request<crate::http_client::MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<http_client::LazyBody<U>>>>
    + WasmCompatSend
    + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::InvalidStatusCode(
            StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let next = self
            .connections
            .lock()
            .expect("scripted connections")
            .pop_front();
        async move {
            let (request_id, chunks) =
                next.expect("a scripted connection should remain for each connect")?;
            let boxed: BoxedStream = Box::pin(futures::stream::iter(chunks));
            let mut builder = Response::builder()
                .status(StatusCode::OK)
                .header(http::header::CONTENT_TYPE, "text/event-stream");
            if let Some(id) = request_id {
                builder = builder.header("x-request-id", id);
            }
            builder.body(boxed).map_err(http_client::Error::Protocol)
        }
    }
}

/// The retry number advances across reconnects, so a bounded policy
/// actually terminates. One arm now serves the initial connect and every
/// reconnect, distinguished only by the retry history it carries; were
/// that history dropped on the way into a reconnect, the policy would see
/// attempt 1 forever and `max_retries` would never be reached.
///
/// A unit test rather than a cassette test: the behavior under test is the
/// state machine's own accounting, and no provider traffic can express
/// "the third connect attempt is refused".
#[tokio::test]
async fn a_bounded_retry_policy_stops_after_its_last_reconnect() {
    // Four scripted failures for a policy that allows two retries: the
    // fourth stays unused unless the numbering regresses, and the client
    // panics past the end rather than silently looping.
    let client = SequencedStreamingClient::new(
        std::iter::repeat_with(|| Err(http_client::Error::StreamEnded)).take(4),
    );
    let req = Request::builder()
        .uri("http://mock.invalid/stream")
        .body(Vec::<u8>::new())
        .expect("request should build");
    let mut source = GenericEventSource::new(client, req);
    source.retry_policy = ExponentialBackoff::new(
        Duration::from_millis(1),
        1.,
        Some(Duration::from_millis(1)),
        Some(2),
    );
    let mut source = Box::pin(source);

    let mut failures = 0;
    while let Some(item) = source.next().await {
        assert!(item.is_err(), "every scripted connect fails");
        failures += 1;
    }

    assert_eq!(
        failures, 3,
        "the initial connect plus two retries, then the policy declines"
    );
}

/// Regression (rig#2265): after a mid-stream failure and reconnect, the
/// slot must describe the connection that is now open — a reconnect whose
/// response omits the header resets it to `None` instead of leaking the
/// first connection's id.
#[tokio::test]
async fn reconnect_replaces_request_id_slot_including_with_none() {
    let client = SequencedStreamingClient::new([
        Ok((
            Some("req-first-connection"),
            vec![
                Ok(Bytes::from_static(b"data: one\n\n")),
                Err(http_client::Error::StreamEnded),
            ],
        )),
        Ok((None, vec![Ok(Bytes::from_static(b"data: two\n\n"))])),
    ]);
    let req = Request::builder()
        .uri("http://mock.invalid/stream")
        .body(Vec::<u8>::new())
        .expect("request should build");
    let (source, slot) = GenericEventSource::new(client, req).capture_request_id("x-request-id");
    let mut source = Box::pin(source);

    let mut messages = Vec::new();
    let mut checked_first_connection = false;
    while let Some(item) = source.next().await {
        if let Ok(Event::Message(message)) = item {
            if !checked_first_connection {
                assert_eq!(
                    slot.lock().expect("slot").as_deref(),
                    Some("req-first-connection"),
                    "the first connection's id is captured at connect"
                );
                checked_first_connection = true;
            }
            messages.push(message.data);
        }
    }

    assert_eq!(messages, ["one", "two"], "both connections delivered data");
    assert_eq!(
        slot.lock().expect("slot").as_deref(),
        None,
        "the reconnect omitted the header, so the slot must not retain the \
             first connection's id"
    );
}
