//! The bundled `reqwest` transport: [`HttpClientExt`] for [`reqwest::Client`]
//! (and, behind the `reqwest-middleware` feature, for
//! [`reqwest_middleware::ClientWithMiddleware`]), plus the glue that maps
//! reqwest's types onto the transport-agnostic ones in the parent module.
//!
//! This module is the only place in rig-core that names a reqwest type in
//! non-test code; it is the interim home of everything that moves to a
//! dedicated reqwest transport crate in the transport-crate split.

use super::{
    Error, HttpClientExt, LazyBody, MultipartForm, NoBody, Request, Response, Result,
    StreamingResponse, instance_error,
    multipart::{Part, PartContent},
};
use crate::wasm_compat::*;
use bytes::Bytes;
use std::pin::Pin;

pub use reqwest::Client as ReqwestClient;

impl From<NoBody> for reqwest::Body {
    fn from(_: NoBody) -> Self {
        reqwest::Body::default()
    }
}

/// Map a transport-level `reqwest::Error` onto the transport-agnostic
/// [`Error`].
///
/// A failure that carries a status (an `error_for_status` rejection) keeps
/// it as [`Error::InvalidStatusCode`] so provider retry and error-inspection
/// paths can still read the code; a response-less failure (connect, decode,
/// timeout) becomes [`Error::Instance`].
pub fn from_reqwest(err: reqwest::Error) -> Error {
    match err.status() {
        Some(status) => Error::InvalidStatusCode(status),
        None => Error::Instance(Box::new(err)),
    }
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
        let bytes = response.bytes().await.map_err(instance_error)?;
        Ok(U::from(bytes))
    });

    res.body(body).map_err(Error::Protocol)
}

/// Convert an already-sent streaming response into the transport-agnostic
/// [`StreamingResponse`], rejecting non-success statuses with the
/// headers-preserving error.
async fn into_streaming_response(response: reqwest::Response) -> Result<StreamingResponse> {
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }

    #[cfg(not(target_family = "wasm"))]
    let mut res = Response::builder()
        .status(response.status())
        .version(response.version());

    #[cfg(target_family = "wasm")]
    let mut res = Response::builder().status(response.status());

    if let Some(hs) = res.headers_mut() {
        *hs = response.headers().clone();
    }

    use futures::StreamExt;

    let mapped_stream: Pin<Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>> = Box::pin(
        response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| Error::Instance(Box::new(e)))),
    );

    res.body(mapped_stream).map_err(Error::Protocol)
}

impl From<MultipartForm> for reqwest::multipart::Form {
    fn from(value: MultipartForm) -> Self {
        let mut form = reqwest::multipart::Form::new();

        for Part {
            name,
            content,
            filename,
            content_type,
        } in value.parts
        {
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
}

/// The one request-driving routine both reqwest-flavoured clients share:
/// `reqwest::Client` and `ClientWithMiddleware` expose the same
/// `request(..) -> RequestBuilder` / `execute(..)` surface but are unrelated
/// types, so the shared code is written once against a tiny private trait.
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
        self.send().await.map_err(instance_error)
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
        self.send().await.map_err(instance_error)
    }
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

    async move { into_lazy_response(req.send_request().await?).await }
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
        .with_multipart(reqwest::multipart::Form::from(body));

    async move { into_lazy_response(req.send_request().await?).await }
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

    async move { into_streaming_response(req.send_request().await?).await }
}

impl HttpClientExt for reqwest::Client {
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

#[cfg(feature = "reqwest-middleware")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest-middleware")))]
impl HttpClientExt for reqwest_middleware::ClientWithMiddleware {
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

        assert_eq!(
            error.non_success_status(),
            Some(StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(
            error.non_success_body(),
            Some(r#"{"error":{"message":"rate limited"}}"#)
        );
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
