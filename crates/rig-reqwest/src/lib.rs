#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! The bundled reqwest HTTP transport for Rig.
//!
//! [`ReqwestClient::default()`] is the process-wide client: built once on
//! first use and shared by every clone, so every provider client made with
//! it shares one connection pool. Construction never fails. On a host with
//! no CA certificates, the `rustls` build trusts the bundled Mozilla roots
//! instead. When reqwest still cannot build its client, every send reports
//! that build failure in-band.
//!
//! Native requests and bodies enter the captured Tokio context on each poll
//! ([`rig_http::runtime`]), using a lazy fallback when no runtime is current. Callers retain ownership;
//! dropping an operation cancels local work, not accepted remote work. Hosts
//! must keep their runtime driven with I/O and timers enabled until operations
//! finish. Missing drivers can panic; a stopped runtime causes I/O failure.
//!
//! ```no_run
//! use rig_reqwest::ReqwestClient;
//!
//! let shared = ReqwestClient::default();
//! let custom = ReqwestClient::from(rig_reqwest::reqwest::Client::new());
//! # let _ = (shared, custom);
//! ```

pub use reqwest;

/// A reqwest client implementing [`HttpClientExt`].
///
/// [`Default`] is the process-wide client, built once and shared by every
/// clone; `From<reqwest::Client>` wraps a configured client and keeps its
/// connection pool. Non-success responses retain at most 64 KiB of body bytes,
/// with a truncation marker only after observing excess data. Partial diagnostic
/// text survives body read failures; the client's timeout policy governs reads.
#[derive(Clone, Debug)]
pub struct ReqwestClient(Built);

/// The process-wide client, erased behind [`DynHttpClient`].
pub fn shared() -> DynHttpClient {
    DynHttpClient::new(ReqwestClient::default())
}

/// What building the client produced: the client, or why reqwest refused.
#[derive(Clone, Debug)]
enum Built {
    Client(Arc<reqwest::Client>),
    /// The shared client could not be built: every send reports this, so
    /// the failure surfaces where the first request is made rather than
    /// where the client was named.
    Failed(Arc<reqwest::Error>),
}

impl Default for ReqwestClient {
    /// The process-wide client, built on first use. It never panics: a
    /// client reqwest cannot build reports the build error on every send.
    fn default() -> Self {
        fn build() -> ReqwestClient {
            let built = reqwest::Client::builder().build();
            #[cfg(all(feature = "rustls", not(target_family = "wasm")))]
            let built = built.or_else(with_bundled_roots);
            ReqwestClient(match built {
                Ok(client) => Built::Client(Arc::new(client)),
                Err(error) => Built::Failed(Arc::new(error)),
            })
        }
        #[cfg(not(target_family = "wasm"))]
        {
            static SHARED: std::sync::LazyLock<ReqwestClient> = std::sync::LazyLock::new(build);
            SHARED.clone()
        }
        #[cfg(target_family = "wasm")]
        {
            thread_local! {
                static SHARED: ReqwestClient = build();
            }
            SHARED.with(Clone::clone)
        }
    }
}

impl ReqwestClient {
    /// The reqwest client, or `None` for the shared client when reqwest
    /// could not build it.
    pub fn inner(&self) -> Option<&reqwest::Client> {
        match &self.0 {
            Built::Client(client) => Some(client),
            Built::Failed(_) => None,
        }
    }

    /// Whether `self` and `other` are clones of one client.
    #[cfg(test)]
    fn same(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Built::Client(a), Built::Client(b)) => Arc::ptr_eq(a, b),
            (Built::Failed(a), Built::Failed(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// The shared client trusting only the bundled Mozilla roots, after the
/// default build failed with `error`. That build differs only in reading
/// the system's root store, which a host without CA certificates lacks.
/// Returns `error` when this build fails too.
#[cfg(all(feature = "rustls", not(target_family = "wasm")))]
fn with_bundled_roots(error: reqwest::Error) -> reqwest::Result<reqwest::Client> {
    let roots = webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .filter_map(|der| reqwest::Certificate::from_der(der).ok());
    let Ok(client) = reqwest::Client::builder().tls_certs_only(roots).build() else {
        return Err(error);
    };
    tracing::warn!(
        "the system's CA certificates could not be loaded ({}), so rig trusts its bundled \
         Mozilla roots; install the system's CA certificates, or set SSL_CERT_FILE or \
         SSL_CERT_DIR, to trust those instead",
        Chain(&error)
    );
    Ok(client)
}

/// The error every send on an unbuilt client reports.
fn unbuilt(error: &Arc<reqwest::Error>) -> Error {
    Error::unbuilt(TransportBuildError(Arc::clone(error)))
}

impl From<reqwest::Client> for ReqwestClient {
    fn from(client: reqwest::Client) -> Self {
        Self(Built::Client(Arc::new(client)))
    }
}

use bytes::Bytes;
use rig_http::http_client::{
    DynHttpClient, Error, HttpClientExt, LazyBody, MultipartForm, Request, Response, Result,
    StreamingResponse, multipart::PartContent,
};
use rig_http::wasm_compat::*;
use std::pin::Pin;
use std::sync::Arc;

/// A transport build failure that displays the source chain and retains the
/// original reqwest error as its source.
#[derive(Debug)]
struct TransportBuildError(Arc<reqwest::Error>);

impl std::fmt::Display for TransportBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "could not build the bundled reqwest transport: {}",
            Chain(&*self.0)
        )
    }
}

/// An error followed by each of its sources, separated by `: `.
struct Chain<'a>(&'a (dyn std::error::Error + 'static));

impl std::fmt::Display for Chain<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)?;
        let mut source = self.0.source();
        while let Some(cause) = source {
            write!(f, ": {cause}")?;
            source = cause.source();
        }
        Ok(())
    }
}

impl std::error::Error for TransportBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}

/// Wrap a reqwest transport error as [`Error::Instance`], retaining its source.
///
/// HTTP status failures are handled separately to preserve response headers
/// and bodies.
pub fn from_reqwest(err: reqwest::Error) -> Error {
    Error::instance(err)
}

/// The most of a non-success reply's body its error keeps.
const ERROR_BODY_LIMIT: usize = 64 * 1024;

/// Read at most 64 KiB of a failed response, retaining status and headers.
/// The client's timeout policy still governs body reads.
async fn non_success_status_error(response: reqwest::Response) -> Error {
    let status = response.status();
    let headers = response.headers().clone();
    non_success_body_error(status, headers, &mut Box::pin(response.bytes_stream())).await
}

/// The error a non-success reply becomes, read from its body's chunks: its
/// status and headers, and at most the first 64 KiB of the body decoded as
/// text by its charset, with a truncation marker when there was more. The
/// body is read until one byte past the limit or its first error; the bytes
/// read before an error are kept, and with none the error is the text. The
/// rest of `body` is left unread for the caller. Every transport that gives
/// providers rig-reqwest's errors builds them here, so the same reply gives
/// the same error whichever reads it.
pub async fn non_success_body_error<S, B, E>(
    status: http::StatusCode,
    headers: http::HeaderMap,
    body: &mut S,
) -> Error
where
    S: futures::Stream<Item = std::result::Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    use futures::StreamExt;

    let mut bytes = Vec::new();
    let mut failure = None;
    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(chunk) => {
                // One byte past the limit is enough to know the body was cut.
                let remaining = ERROR_BODY_LIMIT + 1 - bytes.len();
                bytes.extend(chunk.as_ref().iter().take(remaining));
                if bytes.len() > ERROR_BODY_LIMIT {
                    break;
                }
            }
            Err(error) => {
                failure = Some(error.to_string());
                break;
            }
        }
    }
    let kept = bytes.get(..ERROR_BODY_LIMIT).unwrap_or(&bytes);
    let mut text = if kept.is_empty() {
        failure
            .map(|failure| format!("failed to read error response body: {failure}"))
            .unwrap_or_default()
    } else {
        decode_error_body(kept.to_vec(), &headers).await
    };
    if bytes.len() > ERROR_BODY_LIMIT {
        text.push_str("\n[Response body truncated after 64 KiB]");
    }
    Error::non_success_with_details(status, headers, text)
}

/// Keep reqwest's charset decoding for the bounded, in-memory body.
#[cfg(not(target_family = "wasm"))]
async fn decode_error_body(bytes: Vec<u8>, headers: &http::HeaderMap) -> String {
    let mut response = http::Response::new(bytes);
    *response.headers_mut() = headers.clone();
    reqwest::Response::from(response)
        .text()
        .await
        .unwrap_or_else(|error| format!("failed to read error response body: {error}"))
}

#[cfg(target_family = "wasm")]
async fn decode_error_body(bytes: Vec<u8>, _headers: &http::HeaderMap) -> String {
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    String::from_utf8_lossy(bytes).into_owned()
}

/// Keep successful bodies lazy and owned by the response on every executor.
async fn into_response<U>(mut response: reqwest::Response) -> Result<Response<LazyBody<U>>>
where
    U: From<Bytes>,
    U: WasmCompatSend + 'static,
{
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }

    let mut res = Response::builder().status(response.status());
    if let Some(headers) = res.headers_mut() {
        *headers = std::mem::take(response.headers_mut());
    }

    let body = async {
        let bytes = response.bytes().await.map_err(Error::instance)?;
        Ok(U::from(bytes))
    };
    #[cfg(not(target_family = "wasm"))]
    let body = rig_http::runtime::bind(body)?;
    let body: LazyBody<U> = Box::pin(body);
    res.body(body).map_err(Error::Protocol)
}

fn streaming_head(response: &mut reqwest::Response) -> http::response::Builder {
    #[cfg(not(target_family = "wasm"))]
    let mut res = Response::builder()
        .status(response.status())
        .version(response.version());

    #[cfg(target_family = "wasm")]
    let mut res = Response::builder().status(response.status());

    if let Some(hs) = res.headers_mut() {
        *hs = std::mem::take(response.headers_mut());
    }
    res
}

/// Convert an already-sent streaming response into the transport-agnostic
/// [`StreamingResponse`], rejecting non-success statuses with the
/// headers-preserving error. The body retains the request's reactor context.
async fn into_streaming_response(mut response: reqwest::Response) -> Result<StreamingResponse> {
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }
    let res = streaming_head(&mut response);

    use futures::StreamExt;
    let stream = response
        .bytes_stream()
        .map(|chunk| chunk.map_err(Error::instance));
    #[cfg(not(target_family = "wasm"))]
    let stream = rig_http::runtime::bind_stream(stream)?;
    let stream: Pin<Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>> = Box::pin(stream);
    res.body(stream).map_err(Error::Protocol)
}

/// A part's content type was not a MIME type reqwest would accept.
#[derive(Debug, thiserror::Error)]
#[error("multipart part {part:?} has an unusable content type {content_type:?}: {source}")]
struct InvalidPartContentType {
    part: String,
    content_type: String,
    source: reqwest::Error,
}

/// Render a [`MultipartForm`] as a `reqwest::multipart::Form`.
///
/// Returns an error when a binary part has a content type reqwest rejects.
pub fn multipart_form(value: MultipartForm) -> Result<reqwest::multipart::Form> {
    let mut form = reqwest::multipart::Form::new();

    for part in value.into_parts() {
        let (name, content, filename, content_type) = part.into_pieces();
        match content {
            PartContent::Text(text) => {
                form = form.text(name, text);
            }
            PartContent::Binary(bytes) => {
                // A `Bytes` body is the same reusable part `Part::bytes` makes, uncopied.
                let mut req_part = reqwest::multipart::Part::stream(bytes);
                if let Some(content_type) = content_type.as_ref() {
                    req_part = req_part.mime_str(content_type.as_ref()).map_err(|source| {
                        Error::instance(InvalidPartContentType {
                            part: name.clone(),
                            content_type: content_type.as_ref().to_string(),
                            source,
                        })
                    })?;
                }

                if let Some(filename) = filename {
                    req_part = req_part.file_name(filename);
                }

                form = form.part(name, req_part);
            }
        }
    }

    Ok(form)
}

/// What a request carries: bytes, or a multipart form.
enum Payload {
    Bytes(Bytes),
    Form(reqwest::multipart::Form),
}

/// A reqwest client, plain or with middleware.
trait Execute: WasmCompatSend + WasmCompatSync {
    /// The send of `parts` with `payload`, or why it cannot be sent.
    fn execute(
        &self,
        parts: http::request::Parts,
        payload: Result<Payload>,
    ) -> Result<impl Future<Output = Result<reqwest::Response>> + WasmCompatSend + 'static>;
}

impl Execute for ReqwestClient {
    fn execute(
        &self,
        parts: http::request::Parts,
        payload: Result<Payload>,
    ) -> Result<impl Future<Output = Result<reqwest::Response>> + WasmCompatSend + 'static> {
        let client = match &self.0 {
            Built::Client(client) => client,
            Built::Failed(error) => return Err(unbuilt(error)),
        };
        let request = client
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers);
        let request = match payload? {
            Payload::Bytes(bytes) => request.body(bytes),
            Payload::Form(form) => request.multipart(form),
        };
        Ok(async move { request.send().await.map_err(Error::instance) })
    }
}

/// Sends `parts` with `payload` through `client` and reads the reply with
/// `convert`. The reactor is selected on first poll and kept through the
/// conversion.
fn send_with<C, T, F>(
    client: &C,
    parts: http::request::Parts,
    payload: Result<Payload>,
    convert: impl FnOnce(reqwest::Response) -> F + WasmCompatSend + 'static,
) -> impl Future<Output = Result<T>> + WasmCompatSend + 'static
where
    C: Execute,
    F: Future<Output = Result<T>> + WasmCompatSend,
{
    let sent = client.execute(parts, payload);
    async move {
        let sent = sent?;
        let operation = async move { convert(sent.await?).await };
        #[cfg(not(target_family = "wasm"))]
        let operation = rig_http::runtime::bind(operation)?;
        operation.await
    }
}

/// [`HttpClientExt`] for a client that implements [`Execute`].
macro_rules! http_client_ext {
    ($client:ty) => {
        impl HttpClientExt for $client {
            fn send<T, U>(
                &self,
                req: Request<T>,
            ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
            where
                T: Into<Bytes>,
                U: From<Bytes> + WasmCompatSend + 'static,
            {
                let (parts, body) = req.into_parts();
                let payload = Ok(Payload::Bytes(body.into()));
                send_with(self, parts, payload, into_response::<U>)
            }

            fn send_multipart<U>(
                &self,
                req: Request<MultipartForm>,
            ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
            where
                U: From<Bytes> + WasmCompatSend + 'static,
            {
                let (parts, form) = req.into_parts();
                // Reject invalid MIME types locally rather than sending incomplete metadata.
                let payload = multipart_form(form).map(Payload::Form);
                send_with(self, parts, payload, into_response::<U>)
            }

            fn send_streaming<T>(
                &self,
                req: Request<T>,
            ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
            where
                T: Into<Bytes> + WasmCompatSend,
            {
                let (parts, body) = req.into_parts();
                let payload = Ok(Payload::Bytes(body.into()));
                send_with(self, parts, payload, into_streaming_response)
            }
        }
    };
}
http_client_ext!(ReqwestClient);

#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
mod middleware;
#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
pub use middleware::ReqwestMiddlewareClient;

// Compile-time thread-safety contract: the transport handle is shared across
// threads by every host runtime.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<ReqwestClient>();
};

#[cfg(test)]
mod tests;
