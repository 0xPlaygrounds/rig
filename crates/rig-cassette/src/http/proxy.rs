//! The recording proxy, and the transport both recording modes reach the
//! provider through.
//!
//! The proxy is a local server that sends each request it receives on to the
//! provider through the traffic recorder and streams the reply back, so a
//! proxied recording keeps its exchanges, its ledger lines and its refusals
//! exactly as a [`direct_http`](super::ProviderCassette::direct_http) one
//! does ([`DirectRecorder::record_exchange`]). It reads every reply to its end
//! in a task of its own, whatever the client does: a reply with no body (a
//! 204, a 304, a `HEAD`) and an error the client stops reading are still
//! recorded whole. Only an event stream ends when its client goes, as a
//! provider's reader stops at its final event.
//!
//! [`Raw`] follows no redirect and decodes nothing, in every build whatever
//! reqwest features other crates turn on, and answers every status with its
//! own headers and bytes, so a 3xx or an error reply of any size reaches the
//! client and the cassette as the provider sent it. A redirect is
//! not followed while recording: each hop the client follows is an exchange
//! of its own, with its status and `location`, and one to the provider
//! itself goes on as its path, so the client follows it through the proxy.
//! The recorder asks for every reply uncompressed (`accept-encoding:
//! identity`); one that comes compressed all the same is kept by its size
//! only. The recorder writes what a reply created to the ledger before the
//! proxy hands a chunk on: an event stream's resources as each event arrives,
//! so an event stream is passed on chunk by chunk; a body read whole when it
//! ends, so that reply is passed on one chunk behind and the chunk that
//! completes it goes out only after the ledger line. A websocket
//! upgrade reaches the provider as a plain request without its hop-by-hop
//! `connection` and `upgrade` headers: a rejected handshake is recorded like
//! any reply, and an accepted one cannot be recorded.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri, header};
use axum::response::Response;
use axum::{Router, routing::any};
use futures::{StreamExt, stream};
use rig_core::http_client::{
    self, BoxedStream, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;
use rig_reqwest::reqwest;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::is_hop_by_hop_header;
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
    /// Listen on a local port and forward every request to `origin` through
    /// `http`, the session's recorded [`Raw`] transport.
    pub(crate) async fn start(origin: String, http: DynHttpClient) -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}", listener.local_addr()?);
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
    let mut request = http_client::Request::builder()
        .method(method)
        .uri(target.as_str());
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
            let handing = Handing::of(&parts.headers);
            let mut reply = Response::new(Body::from_stream(drained(body, handing)));
            *reply.status_mut() = parts.status;
            // The body is passed on byte for byte, so its length still holds.
            copy_headers(&parts.headers, reply.headers_mut(), |name| {
                name == header::CONTENT_LENGTH
            });
            // A redirect to the provider itself goes on as its path, so the
            // client follows it through the proxy and it is recorded.
            if let Some(location) = reply.headers().get(header::LOCATION)
                && let Ok(location) = location.to_str()
                && let Some(local) = super::local_location(&forward.origin, location)
                && let Ok(local) = axum::http::HeaderValue::from_str(&local)
            {
                reply.headers_mut().insert(header::LOCATION, local);
            }
            reply
        }
        // A transport error names the URL, which can carry a key.
        Err(error) => failure(
            crate::scrub::Scrubber::for_request(&target, &headers)
                .scrub_text(&format!("the proxy could not reach the provider: {error}")),
        ),
    }
}

/// How the proxy hands a reply to its client.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Handing {
    /// An event stream: each chunk at once, since the recorder wrote what
    /// each event created first, until the client goes.
    Events,
    /// Any other body: one chunk behind, read to its end even when the
    /// client went first.
    Body,
}

impl Handing {
    fn of(headers: &HeaderMap) -> Self {
        if traffic::is_event_stream(headers) {
            Self::Events
        } else {
            Self::Body
        }
    }
}

/// `body`, read by a task of its own and handed to the client as it can take
/// it, as `handing` says. A reply handed one chunk behind sends its last
/// chunk after the recorder saw it end, so the ledger line for what it
/// created comes first. An event stream ends when its client goes, even
/// while the provider sends nothing, such as after its final event on a
/// connection it keeps open.
fn drained(
    mut body: BoxedStream,
    handing: Handing,
) -> impl futures::Stream<Item = http_client::Result<Bytes>> {
    let (sender, receiver) = mpsc::channel(16);
    let ends_with_client = handing == Handing::Events;
    tokio::spawn(async move {
        let mut held: Option<Bytes> = None;
        let mut client_gone = false;
        loop {
            let item = if ends_with_client {
                tokio::select! {
                    () = sender.closed() => return,
                    item = body.next() => item,
                }
            } else {
                body.next().await
            };
            match item {
                None => break,
                Some(Ok(chunk)) if handing == Handing::Events => {
                    if sender.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                }
                Some(Ok(chunk)) => {
                    if let Some(previous) = held.replace(chunk)
                        && !client_gone
                    {
                        client_gone = sender.send(Ok(previous)).await.is_err();
                    }
                }
                Some(Err(error)) => {
                    if let Some(previous) = held.take() {
                        let _ = sender.send(Ok(previous)).await;
                    }
                    let _ = sender.send(Err(error)).await;
                    return;
                }
            }
        }
        if let Some(last) = held {
            let _ = sender.send(Ok(last)).await;
        }
    });
    stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|item| (item, receiver))
    })
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

/// A recording session's transport: a reqwest client of its own that follows
/// no redirect and decodes no body in any build, whose every request, unary,
/// multipart or streamed, answers every status, a 3xx included, with its
/// headers and its body as the provider sent it.
#[derive(Clone)]
pub(crate) struct Raw {
    client: reqwest::Client,
}

impl Raw {
    /// The transport over a new client; fails when reqwest cannot build one.
    pub(crate) fn new() -> std::io::Result<Self> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            // The same in every build: a plugin that turns on reqwest's
            // decompression would otherwise decode a reply compressed anyway.
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .build()
            .map_err(std::io::Error::other)?;
        Ok(Self { client })
    }

    fn request(&self, parts: axum::http::request::Parts) -> reqwest::RequestBuilder {
        self.client
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
    }
}

/// `response`'s status, version and headers.
fn head(response: &mut reqwest::Response) -> axum::http::response::Builder {
    let mut head = http_client::Response::builder()
        .status(response.status())
        .version(response.version());
    if let Some(headers) = head.headers_mut() {
        *headers = std::mem::take(response.headers_mut());
    }
    head
}

/// `request`'s reply, its body read whole when awaited.
async fn unary<U>(
    request: http_client::Result<reqwest::RequestBuilder>,
) -> http_client::Result<http_client::Response<LazyBody<U>>>
where
    U: From<Bytes> + WasmCompatSend + 'static,
{
    let mut response = request?
        .send()
        .await
        .map_err(http_client::Error::instance)?;
    let head = head(&mut response);
    let body: LazyBody<U> = Box::pin(async move {
        response
            .bytes()
            .await
            .map(U::from)
            .map_err(http_client::Error::instance)
    });
    head.body(body).map_err(http_client::Error::Protocol)
}

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
        let (parts, body) = req.into_parts();
        unary(Ok(self.request(parts).body(body.into())))
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
        let (parts, form) = req.into_parts();
        let request = self.request(parts);
        unary(rig_reqwest::multipart_form(form).map(|form| request.multipart(form)))
    }

    fn send_streaming<T>(
        &self,
        req: http_client::Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let (parts, body) = req.into_parts();
        let request = self.request(parts).body(body.into());
        async move {
            let mut response = request.send().await.map_err(http_client::Error::instance)?;
            let head = head(&mut response);
            let body: BoxedStream = Box::pin(
                response
                    .bytes_stream()
                    .map(|chunk| chunk.map_err(http_client::Error::instance)),
            );
            head.body(body).map_err(http_client::Error::Protocol)
        }
    }
}

/// `inner` with rig-reqwest's error semantics: a non-success reply becomes
/// [`rig_reqwest::non_success_body_error`], the error replay's rig-reqwest
/// client gives for the same reply. The reply is still read to its end, so
/// the recorder beneath it keeps the exact bytes the provider sent.
pub(crate) struct ProviderErrors(pub(crate) DynHttpClient);

/// `sent`'s reply, or the provider error of a non-success one.
async fn checked<U>(
    sent: impl Future<Output = http_client::Result<http_client::Response<LazyBody<Bytes>>>>,
) -> http_client::Result<http_client::Response<LazyBody<U>>>
where
    U: From<Bytes> + WasmCompatSend + 'static,
{
    let response = sent.await?;
    if !response.status().is_success() {
        let (parts, body) = response.into_parts();
        let mut body = stream::once(body);
        return Err(
            rig_reqwest::non_success_body_error(parts.status, parts.headers, &mut body).await,
        );
    }
    Ok(response.map(|body| -> LazyBody<U> { Box::pin(async move { body.await.map(U::from) }) }))
}

impl HttpClientExt for ProviderErrors {
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
        checked(self.0.send::<T, Bytes>(req))
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
        checked(self.0.send_multipart::<Bytes>(req))
    }

    fn send_streaming<T>(
        &self,
        req: http_client::Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let sent = self.0.send_streaming(req);
        async move {
            let response = sent.await?;
            if response.status().is_success() {
                return Ok(response);
            }
            let (parts, mut body) = response.into_parts();
            let error =
                rig_reqwest::non_success_body_error(parts.status, parts.headers, &mut body).await;
            // The recorder keeps a reply only when it was read to its end.
            while let Some(Ok(_)) = body.next().await {}
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests;
