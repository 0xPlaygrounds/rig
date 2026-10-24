//! The recording proxy: a local server that sends each request it receives
//! on to the provider through the traffic recorder and streams the reply
//! back, so a proxied recording keeps its exchanges, its ledger lines and
//! its refusals exactly as a [`direct_http`](super::ProviderCassette::direct_http)
//! one does ([`DirectRecorder::record_exchange`]).
//!
//! A reply is passed back one chunk behind: the chunk that completes it goes
//! out only after the recorder saw the reply end, so the ledger line for what
//! it created is on disk before the client can read all of it. A websocket
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
use rig_core::http_client::{self, BoxedStream, DynHttpClient, HttpClientExt};
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
            DynHttpClient::new(rig_reqwest::ReqwestClient::default()),
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
        Err(error) => match error.non_success_status() {
            Some(status) => {
                let body = error.non_success_body().unwrap_or_default().to_owned();
                let mut reply = Response::new(Body::from(body));
                *reply.status_mut() = status;
                if let Some(headers) = error.non_success_headers() {
                    copy_headers(headers, reply.headers_mut(), |_| false);
                }
                reply
            }
            None => failure(format!("the proxy could not reach the provider: {error}")),
        },
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
