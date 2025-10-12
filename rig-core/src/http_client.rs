use crate::if_wasm;
use bytes::Bytes;
#[cfg(not(target_family = "wasm"))]
use futures::stream::BoxStream;
#[cfg(target_family = "wasm")]
use futures::stream::Stream;
pub use http::{HeaderMap, HeaderValue, Method, Request, Response, Uri, request::Builder};
use reqwest::Body;
use std::future::Future;

if_wasm! {
    use std::pin::Pin;
}

use crate::wasm_compat::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Http error: {0}")]
    Protocol(#[from] http::Error),
    #[cfg(not(target_family = "wasm"))]
    #[error("Http client error: {0}")]
    Instance(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    #[error("Http client error: {0}")]
    Instance(#[from] Box<dyn std::error::Error + 'static>),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(not(target_family = "wasm"))]
fn instance_error<E: std::error::Error + Send + Sync + 'static>(error: E) -> Error {
    Error::Instance(error.into())
}

#[cfg(target_family = "wasm")]
fn instance_error<E: std::error::Error + 'static>(error: E) -> Error {
    Error::Instance(error.into())
}

pub type LazyBytes = WasmBoxedFuture<'static, Result<Bytes>>;
pub type LazyBody<T> = WasmBoxedFuture<'static, Result<T>>;

#[cfg(not(target_family = "wasm"))]
pub type ByteStream = BoxStream<'static, Result<Bytes>>;

#[cfg(target_family = "wasm")]
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + 'static>>;

pub type StreamingResponse = Response<ByteStream>;

pub struct NoBody;

impl From<NoBody> for Bytes {
    fn from(_: NoBody) -> Self {
        Bytes::new()
    }
}

impl From<NoBody> for Body {
    fn from(_: NoBody) -> Self {
        reqwest::Body::default()
    }
}

pub async fn text(response: Response<LazyBody<Vec<u8>>>) -> Result<String> {
    let text = response.into_body().await?;
    Ok(String::from(String::from_utf8_lossy(&text)))
}

pub fn with_bearer_auth(req: Builder, auth: &str) -> Result<Builder> {
    let auth_header =
        HeaderValue::from_str(&format!("Bearer {}", auth)).map_err(http::Error::from)?;

    Ok(req.header("Authorization", auth_header))
}

pub trait HttpClientExt: WasmCompatSend + WasmCompatSync {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>;
}

impl HttpClientExt for reqwest::Client {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        U: From<Bytes> + WasmCompatSend,
    {
        let (parts, body) = req.into_parts();
        let req = self
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
            .body(body.into());

        async move {
            let response = req.send().await.map_err(instance_error)?;

            let mut res = Response::builder().status(response.status());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            let body: LazyBody<U> = Box::pin(async {
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| Error::Instance(e.into()))?;

                let body = U::from(bytes);
                Ok(body)
            });

            res.body(body).map_err(Error::Protocol)
        }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
    {
        let (parts, body) = req.into_parts();
        let req = self
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
            .body(body.into());

        async move {
            let response: reqwest::Response = req.send().await.map_err(instance_error)?;

            #[cfg(not(target_family = "wasm"))]
            let mut res = Response::builder()
                .status(response.status())
                .version(response.version());

            #[cfg(target_family = "wasm")]
            let mut res = Response::builder().status(response.status());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            let stream: ByteStream = {
                use futures::TryStreamExt;
                Box::pin(response.bytes_stream().map_err(instance_error))
            };

            Ok(res.body(stream)?)
        }
    }
}

#[cfg(feature = "rama")]
#[cfg_attr(docsrs, doc(cfg(feature = "rama")))]
impl<ModifiedBody, ConnResponse> HttpClientExt
    for rama::http::client::EasyHttpWebClient<
        (),
        rama::http::Body,
        rama::net::client::EstablishedClientConnection<ConnResponse, (), Request<ModifiedBody>>,
    >
where
    ModifiedBody: rama::http::dep::http_body::Body<Data: Send + 'static, Error: Into<rama::error::BoxError>>
        + Unpin
        + Send
        + 'static,
    ConnResponse: rama::Service<
            (),
            Request<ModifiedBody>,
            Response = rama::http::Response,
            Error = rama::error::BoxError,
        >,
{
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        U: From<Bytes> + WasmCompatSend,
    {
        let client = self.clone();
        use rama::Service;
        let (parts, body) = req.into_parts();

        let mut request = rama::http::Request::builder()
            .uri(parts.uri)
            .method(parts.method);

        for (key, val) in parts.headers {
            request = request.header(key.unwrap(), val);
        }

        let bytes: Bytes = body.into();
        let body = rama::http::Body::from(bytes);

        let req = request.body(body).unwrap();

        async move {
            let response = client
                .serve(rama::Context::new((), rama::rt::Executor::new()), req)
                .await
                .map_err(instance_error)?;

            let mut res = Response::builder()
                .status(response.status())
                .version(response.version());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            let body: LazyBody<U> = Box::pin(async move {
                let bytes = response.into_body();
                let bytes = bytes.collect().await.unwrap().to_bytes();

                let body = U::from(bytes);
                Ok(body)
            });

            res.body(body).map_err(Error::Protocol)
        }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
    {
        let client = self.clone();
        use futures::StreamExt;
        use rama::Service;
        let (parts, body) = req.into_parts();

        let mut request = rama::http::Request::builder()
            .uri(parts.uri)
            .method(parts.method);

        for (key, val) in parts.headers {
            request = request.header(key.unwrap(), val);
        }

        let bytes: Bytes = body.into();
        let body = rama::http::Body::from(bytes);

        let req = request.body(body).unwrap();

        async move {
            let response = client
                .serve(rama::Context::new((), rama::rt::Executor::new()), req)
                .await
                .unwrap();

            #[cfg(not(target_family = "wasm"))]
            let mut res = Response::builder()
                .status(response.status())
                .version(response.version());

            #[cfg(target_family = "wasm")]
            let mut res = Response::builder().status(response.status());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            let stream = response.into_body().into_data_stream();

            let boxed_stream: std::pin::Pin<
                Box<dyn futures::Stream<Item = core::result::Result<Bytes, Error>> + Send>,
            > = Box::pin(stream.map(|result| result.map_err(Error::Instance)));
            res.body(boxed_stream)
                .map_err(|_| Error::Instance("failed to build response".into()))
        }
    }
}
