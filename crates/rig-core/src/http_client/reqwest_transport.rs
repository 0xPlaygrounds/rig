//! The bundled `reqwest` transport, behind the non-default `reqwest` feature.
//!
//! rig-core is transport-agnostic: every provider client is generic over an
//! `H: HttpClientExt`. This module supplies the implementation for
//! [`reqwest::Client`] (and, behind `reqwest-middleware`, for
//! `reqwest_middleware::ClientWithMiddleware`), which is what
//! [`super::DefaultHttp`] resolves to when the feature is on.
//!
//! # Running without a tokio runtime
//!
//! Async reqwest needs a tokio reactor on native targets. Inside a tokio
//! runtime this transport awaits reqwest futures directly. Outside one —
//! Bevy task pools, smol, `futures::executor::block_on` — it drives them on a
//! lazily started single-worker fallback runtime and hands the caller plain
//! runtime-agnostic futures (a tokio `JoinHandle`, a `futures` channel receiver
//! for streamed bodies), so nothing is ever `block_on`'d and no thread parks.

pub use reqwest;

use super::{
    Error, HttpClientExt, LazyBody, MultipartForm, Request, Response, Result, StreamingResponse,
    multipart::PartContent,
};
use crate::wasm_compat::*;
use bytes::Bytes;
use std::pin::Pin;

/// Map a transport-level `reqwest::Error` onto the transport-agnostic
/// [`Error`].
///
/// A failure that carries a status (an `error_for_status` rejection) keeps
/// it as [`Error::InvalidStatusCode`] so provider retry and error-inspection
/// paths can still read the code; a response-less failure (connect, decode,
/// timeout) becomes [`Error::Instance`].
pub fn from_reqwest(err: reqwest::Error) -> Error {
    err.status()
        .map_or_else(|| Error::instance(err), Error::InvalidStatusCode)
}

/// Read the status, headers and body off a failed `reqwest::Response` and
/// build the headers-preserving non-success error (rig#2314).
async fn non_success_status_error(response: reqwest::Response) -> Error {
    let status = response.status();
    let headers = response.headers().clone();
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("failed to read error response body: {error}"));
    Error::non_success_with_details(status, headers, body)
}

/// Build the transport-agnostic response whose body is read lazily (the
/// caller awaits it later). Only valid when the caller can poll reqwest
/// futures — i.e. inside a tokio runtime (or on wasm).
async fn into_lazy_response<U>(response: reqwest::Response) -> Result<Response<LazyBody<U>>>
where
    U: From<Bytes>,
    U: WasmCompatSend + 'static,
{
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }

    let mut res = Response::builder().status(response.status());
    if let Some(headers) = res.headers_mut() {
        *headers = response.headers().clone();
    }

    let body: LazyBody<U> = Box::pin(async {
        let bytes = response.bytes().await.map_err(Error::instance)?;
        Ok(U::from(bytes))
    });

    res.body(body).map_err(Error::Protocol)
}

/// Like [`into_lazy_response`], but reads the body eagerly — for the
/// off-runtime path, where the returned future may be polled by an executor
/// that cannot drive reqwest.
#[cfg(not(target_family = "wasm"))]
async fn into_eager_response<U>(response: reqwest::Response) -> Result<Response<LazyBody<U>>>
where
    U: From<Bytes>,
    U: WasmCompatSend + 'static,
{
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }

    let mut res = Response::builder().status(response.status());
    if let Some(headers) = res.headers_mut() {
        *headers = response.headers().clone();
    }

    let bytes = response.bytes().await.map_err(Error::instance)?;
    let body: LazyBody<U> = Box::pin(std::future::ready(Ok(U::from(bytes))));
    res.body(body).map_err(Error::Protocol)
}

fn streaming_head(response: &reqwest::Response) -> http::response::Builder {
    #[cfg(not(target_family = "wasm"))]
    let mut res = Response::builder()
        .status(response.status())
        .version(response.version());

    #[cfg(target_family = "wasm")]
    let mut res = Response::builder().status(response.status());

    if let Some(hs) = res.headers_mut() {
        *hs = response.headers().clone();
    }
    res
}

/// Convert an already-sent streaming response into the transport-agnostic
/// [`StreamingResponse`], rejecting non-success statuses with the
/// headers-preserving error. The byte stream is reqwest's own, so this is
/// only valid where the caller can poll reqwest futures.
async fn into_streaming_response(response: reqwest::Response) -> Result<StreamingResponse> {
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }
    let res = streaming_head(&response);

    use futures::StreamExt;
    let mapped_stream: Pin<Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>> = Box::pin(
        response
            .bytes_stream()
            .map(|chunk| chunk.map_err(Error::instance)),
    );

    res.body(mapped_stream).map_err(Error::Protocol)
}

/// Off-runtime streaming: the body is *driven* on the fallback runtime and
/// forwarded through a bounded channel; the caller's executor polls only the
/// receiver, which is a plain `futures` stream.
#[cfg(not(target_family = "wasm"))]
async fn into_forwarded_streaming_response(
    response: reqwest::Response,
) -> Result<StreamingResponse> {
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }
    let res = streaming_head(&response);

    use futures::{SinkExt, StreamExt};
    let (mut tx, rx) = futures::channel::mpsc::channel::<Result<Bytes>>(16);
    super::runtime::spawn_off_runtime(async move {
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            if tx.send(chunk.map_err(Error::instance)).await.is_err() {
                // Receiver dropped: the consumer stopped reading.
                break;
            }
        }
    })?;
    let stream: Pin<Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>> = Box::pin(rx);
    res.body(stream).map_err(Error::Protocol)
}

/// Render a [`MultipartForm`] as a `reqwest::multipart::Form`.
pub fn multipart_form(value: MultipartForm) -> reqwest::multipart::Form {
    let mut form = reqwest::multipart::Form::new();

    for part in value.into_parts() {
        let (name, content, filename, content_type) = part.into_pieces();
        match content {
            PartContent::Text(text) => {
                form = form.text(name, text);
            }
            PartContent::Binary(bytes) => {
                let mut req_part = if let Some(content_type) = content_type.as_ref() {
                    reqwest::multipart::Part::bytes(bytes.to_vec())
                        .mime_str(content_type.as_ref())
                        .unwrap_or_else(|_| reqwest::multipart::Part::bytes(bytes.to_vec()))
                } else {
                    reqwest::multipart::Part::bytes(bytes.to_vec())
                };

                if let Some(filename) = filename {
                    req_part = req_part.file_name(filename);
                }

                form = form.part(name, req_part);
            }
        }
    }

    form
}

/// The one request-driving routine both reqwest-flavoured clients share:
/// `reqwest::Client` and `ClientWithMiddleware` expose the same
/// `request(..) -> RequestBuilder` / `send()` surface but are unrelated types,
/// so the shared code is written once against a tiny private trait.
trait ReqwestLike: Clone + WasmCompatSend + WasmCompatSync + 'static {
    type Builder: RequestBuilderLike;
    fn request_builder(&self, method: http::Method, url: String) -> Self::Builder;
}

trait RequestBuilderLike: Sized + WasmCompatSend + 'static {
    fn with_headers(self, headers: http::HeaderMap) -> Self;
    fn with_body(self, body: reqwest::Body) -> Self;
    fn with_multipart(self, form: reqwest::multipart::Form) -> Self;
    fn send_request(self) -> impl Future<Output = Result<reqwest::Response>> + WasmCompatSend;
}

impl ReqwestLike for reqwest::Client {
    type Builder = reqwest::RequestBuilder;
    fn request_builder(&self, method: http::Method, url: String) -> Self::Builder {
        self.request(method, url)
    }
}

impl RequestBuilderLike for reqwest::RequestBuilder {
    fn with_headers(self, headers: http::HeaderMap) -> Self {
        self.headers(headers)
    }
    fn with_body(self, body: reqwest::Body) -> Self {
        self.body(body)
    }
    fn with_multipart(self, form: reqwest::multipart::Form) -> Self {
        self.multipart(form)
    }
    async fn send_request(self) -> Result<reqwest::Response> {
        self.send().await.map_err(Error::instance)
    }
}

#[cfg(feature = "reqwest-middleware")]
impl ReqwestLike for reqwest_middleware::ClientWithMiddleware {
    type Builder = reqwest_middleware::RequestBuilder;
    fn request_builder(&self, method: http::Method, url: String) -> Self::Builder {
        self.request(method, url)
    }
}

#[cfg(feature = "reqwest-middleware")]
impl RequestBuilderLike for reqwest_middleware::RequestBuilder {
    fn with_headers(self, headers: http::HeaderMap) -> Self {
        self.headers(headers)
    }
    fn with_body(self, body: reqwest::Body) -> Self {
        self.body(body)
    }
    fn with_multipart(self, form: reqwest::multipart::Form) -> Self {
        self.multipart(form)
    }
    async fn send_request(self) -> Result<reqwest::Response> {
        self.send().await.map_err(Error::instance)
    }
}

/// Drive `request` and convert its response on whichever side can poll
/// reqwest: directly when inside tokio (or on wasm), on the fallback runtime
/// otherwise — in which case `off_runtime` must produce a response whose
/// body no longer needs reqwest to be polled.
async fn drive<B, T, OnRt, OffRt, FutOn, FutOff>(
    request: B,
    on_runtime: OnRt,
    off_runtime: OffRt,
) -> Result<T>
where
    B: RequestBuilderLike,
    T: WasmCompatSend + 'static,
    OnRt: FnOnce(reqwest::Response) -> FutOn + WasmCompatSend + 'static,
    FutOn: Future<Output = Result<T>> + WasmCompatSend,
    OffRt: FnOnce(reqwest::Response) -> FutOff + WasmCompatSend + 'static,
    FutOff: Future<Output = Result<T>> + WasmCompatSend,
{
    #[cfg(not(target_family = "wasm"))]
    if !super::runtime::in_tokio() {
        return super::runtime::run_off_runtime(async move {
            let response = request.send_request().await?;
            off_runtime(response).await
        })
        .await?;
    }
    #[cfg(target_family = "wasm")]
    let _ = &off_runtime;
    let response = request.send_request().await?;
    on_runtime(response).await
}

fn send_via<C, T, U>(
    client: &C,
    req: Request<T>,
) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
where
    C: ReqwestLike,
    T: Into<Bytes>,
    U: From<Bytes> + WasmCompatSend + 'static,
{
    let (parts, body) = req.into_parts();
    let req = client
        .request_builder(parts.method, parts.uri.to_string())
        .with_headers(parts.headers)
        .with_body(body.into().into());

    #[cfg(not(target_family = "wasm"))]
    let off = into_eager_response::<U>;
    #[cfg(target_family = "wasm")]
    let off = into_lazy_response::<U>;
    drive(req, into_lazy_response::<U>, off)
}

fn send_multipart_via<C, U>(
    client: &C,
    req: Request<MultipartForm>,
) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
where
    C: ReqwestLike,
    U: From<Bytes> + WasmCompatSend + 'static,
{
    let (parts, body) = req.into_parts();
    let req = client
        .request_builder(parts.method, parts.uri.to_string())
        .with_headers(parts.headers)
        .with_multipart(multipart_form(body));

    #[cfg(not(target_family = "wasm"))]
    let off = into_eager_response::<U>;
    #[cfg(target_family = "wasm")]
    let off = into_lazy_response::<U>;
    drive(req, into_lazy_response::<U>, off)
}

fn send_streaming_via<C, T>(
    client: &C,
    req: Request<T>,
) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
where
    C: ReqwestLike,
    T: Into<Bytes> + WasmCompatSend,
{
    let (parts, body) = req.into_parts();
    let req = client
        .request_builder(parts.method, parts.uri.to_string())
        .with_headers(parts.headers)
        .with_body(body.into().into());

    #[cfg(not(target_family = "wasm"))]
    let off = into_forwarded_streaming_response;
    #[cfg(target_family = "wasm")]
    let off = into_streaming_response;
    drive(req, into_streaming_response, off)
}

macro_rules! impl_http_client_ext_via {
    ($(#[$attribute:meta])* $client:ty) => {
        $(#[$attribute])*
        impl HttpClientExt for $client {
            fn send<T, U>(
                &self,
                req: Request<T>,
            ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
            where
                T: Into<Bytes>,
                U: From<Bytes> + WasmCompatSend + 'static,
            {
                send_via(self, req)
            }

            fn send_multipart<U>(
                &self,
                req: Request<MultipartForm>,
            ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
            where
                U: From<Bytes> + WasmCompatSend + 'static,
            {
                send_multipart_via(self, req)
            }

            fn send_streaming<T>(
                &self,
                req: Request<T>,
            ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
            where
                T: Into<Bytes> + WasmCompatSend,
            {
                send_streaming_via(self, req)
            }
        }
    };
}

impl_http_client_ext_via!(reqwest::Client);

impl_http_client_ext_via!(
    #[cfg(feature = "reqwest-middleware")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest-middleware")))]
    reqwest_middleware::ClientWithMiddleware
);

// Compile-time thread-safety contract: the transport handle is shared across
// threads by every host runtime.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<reqwest::Client>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use http::StatusCode;

    /// rig#2210: the bundled transport's own error constructor is where the
    /// headers are captured, so drive it with a real `reqwest::Response`.
    #[tokio::test]
    async fn non_success_status_error_preserves_response_headers() {
        let response = http::Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .header("retry-after", "20")
            .header("x-ratelimit-remaining", "0")
            .body(r#"{"error":{"message":"rate limited"}}"#)
            .expect("valid response");

        let error = non_success_status_error(reqwest::Response::from(response)).await;

        assert!(matches!(
            &error,
            Error::InvalidStatusCodeWithDetails { status, .. } if *status == StatusCode::TOO_MANY_REQUESTS
        ));
        let headers = error
            .non_success_headers()
            .expect("headers captured at error construction");
        assert_eq!(
            headers.get("retry-after").and_then(|v| v.to_str().ok()),
            Some("20")
        );
        assert_eq!(
            headers
                .get("x-ratelimit-remaining")
                .and_then(|v| v.to_str().ok()),
            Some("0")
        );
    }
}
