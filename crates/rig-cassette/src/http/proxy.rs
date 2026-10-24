//! The recording proxy: a local server that sends each request it receives
//! on to the provider through the traffic recorder and streams the reply
//! back, so a proxied recording keeps its exchanges, its ledger lines and
//! its refusals exactly as a [`direct_http`](super::ProviderCassette::direct_http)
//! one does ([`DirectRecorder::record_exchange`]).
//!
//! The proxy reaches the provider through [`Raw`], rig-reqwest's shared
//! client with every status answered by its own headers and byte stream, so
//! an error reply of any size or encoding reaches the client and the cassette
//! unchanged. The recorder writes what a reply created to the ledger as an
//! event stream's events arrive, and a body read whole when it ends; the
//! reply is passed back one chunk behind, so the chunk that completes it goes
//! out only after that, and no client reads a resource before it is on disk.
//! A websocket
//! upgrade reaches the provider as a plain request without its hop-by-hop
//! `connection` and `upgrade` headers: a rejected handshake is recorded like
//! any reply, and an accepted one cannot be recorded.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri, header};
use axum::response::Response;
use axum::{Router, routing::any};
use futures::{Stream, StreamExt, stream};
use rig_core::http_client::{
    self, BoxedStream, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;
use rig_reqwest::ReqwestClient;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use super::{DirectRecorder, is_hop_by_hop_header};
use crate::traffic;

/// A running proxy: accept on `base_url`, forward to the provider's origin.
pub(crate) struct Proxy {
    pub(crate) base_url: String,
    task: JoinHandle<()>,
}

/// Where the proxy forwards and the recorded client it forwards through.
struct Forward {
    origin: String,
    http: DynHttpClient,
}

impl Proxy {
    /// Listen on a local port and forward every request to `origin`, each
    /// exchange kept by `recorder`.
    pub(crate) async fn start(origin: String, recorder: DirectRecorder) -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}", listener.local_addr()?);
        let http = traffic::recorded(
            DynHttpClient::new(Raw::default()),
            traffic::Sink::Cassette(recorder),
        );
        // A request reaches the provider whole, however large (an upload, a
        // long conversation): no limit of the proxy's own.
        let app = Router::new()
            .fallback(any(forward))
            .layer(DefaultBodyLimit::disable())
            .with_state(Arc::new(Forward { origin, http }));
        let task = tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("the recording proxy stopped: {error}");
            }
        });
        Ok(Self { base_url, task })
    }
}

impl Drop for Proxy {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn forward(
    State(forward): State<Arc<Forward>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let target = format!(
        "{}{}",
        forward.origin,
        uri.path_and_query().map_or("/", |path| path.as_str())
    );
    let mut request = http_client::Request::builder().method(method).uri(target);
    for (name, value) in &headers {
        if name != header::HOST && !is_hop_by_hop_header(name.as_str()) {
            request = request.header(name, value);
        }
    }
    let request = match request.body(body) {
        Ok(request) => request,
        Err(error) => return failure(format!("the proxy could not build the request: {error}")),
    };
    match forward.http.send_streaming(request).await {
        Ok(response) => {
            let (parts, body) = response.into_parts();
            let mut reply = Response::new(Body::from_stream(behind(body)));
            *reply.status_mut() = parts.status;
            // The body is passed on byte for byte, so its length still holds.
            copy_headers(&parts.headers, reply.headers_mut(), |name| {
                name == header::CONTENT_LENGTH
            });
            reply
        }
        Err(error) => failure(format!("the proxy could not reach the provider: {error}")),
    }
}

/// Copy the end-to-end headers of `from`, and the hop-by-hop ones `keep`
/// accepts.
fn copy_headers(from: &HeaderMap, to: &mut HeaderMap, keep: impl Fn(&header::HeaderName) -> bool) {
    for (name, value) in from {
        if keep(name) || !is_hop_by_hop_header(name.as_str()) {
            to.append(name, value.clone());
        }
    }
}

fn failure(message: String) -> Response {
    let mut reply = Response::new(Body::from(message));
    *reply.status_mut() = StatusCode::BAD_GATEWAY;
    reply
}

/// rig-reqwest's shared client as the proxy forwards through it: a streamed
/// request answers every status with its headers and its body as the
/// provider sent it, where [`ReqwestClient`] turns a non-success reply into
/// an error holding at most 64 KiB of decoded text. The proxy only streams;
/// a unary or multipart request is [`ReqwestClient`]'s.
#[derive(Clone, Default)]
struct Raw(ReqwestClient);

impl HttpClientExt for Raw {
    fn send<T, U>(
        &self,
        req: http_client::Request<T>,
    ) -> impl Future<Output = http_client::Result<http_client::Response<LazyBody<U>>>>
    + WasmCompatSend
    + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.0.send(req)
    }

    fn send_multipart<U>(
        &self,
        req: http_client::Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<http_client::Response<LazyBody<U>>>>
    + WasmCompatSend
    + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.0.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: http_client::Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let (parts, body) = req.into_parts();
        let request = self.0.inner().map(|client| {
            client
                .request(parts.method, parts.uri.to_string())
                .headers(parts.headers)
                .body(body.into())
        });
        async move {
            let request = request.ok_or_else(|| {
                http_client::Error::instance(std::io::Error::other(
                    "reqwest could not build its shared client",
                ))
            })?;
            let mut response = request.send().await.map_err(http_client::Error::instance)?;
            let mut head = http_client::Response::builder()
                .status(response.status())
                .version(response.version());
            if let Some(headers) = head.headers_mut() {
                *headers = std::mem::take(response.headers_mut());
            }
            let body: BoxedStream = Box::pin(
                response
                    .bytes_stream()
                    .map(|chunk| chunk.map_err(http_client::Error::instance)),
            );
            head.body(body).map_err(http_client::Error::Protocol)
        }
    }
}

/// `body`'s chunks one behind: each is passed on when the next arrives, the
/// last when the stream ends, and a failure after the chunks before it.
fn behind(body: BoxedStream) -> impl Stream<Item = http_client::Result<Bytes>> {
    struct Behind {
        body: BoxedStream,
        held: Option<Bytes>,
        failed: Option<http_client::Error>,
        ended: bool,
    }
    let start = Behind {
        body,
        held: None,
        failed: None,
        ended: false,
    };
    stream::unfold(start, |mut state| async move {
        if let Some(error) = state.failed.take() {
            state.ended = true;
            return Some((Err(error), state));
        }
        if state.ended {
            return None;
        }
        loop {
            match state.body.next().await {
                Some(Ok(chunk)) => {
                    if let Some(previous) = state.held.replace(chunk) {
                        return Some((Ok(previous), state));
                    }
                }
                Some(Err(error)) => match state.held.take() {
                    Some(previous) => {
                        state.failed = Some(error);
                        return Some((Ok(previous), state));
                    }
                    None => {
                        state.ended = true;
                        return Some((Err(error), state));
                    }
                },
                None => {
                    state.ended = true;
                    return state.held.take().map(|chunk| (Ok(chunk), state));
                }
            }
        }
    })
}

#[cfg(test)]
mod tests;
