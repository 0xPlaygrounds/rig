use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
pub use http::{HeaderMap, HeaderValue, Method, Request, Response, Uri, request::Builder};
use reqwest::Body;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Http error: {0}")]
    Protocol(#[from] http::Error),
    #[error("Http client error: {0}")]
    Instance(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
}

pub type Result<T> = std::result::Result<T, Error>;

fn instance_error<E: std::error::Error + Send + Sync + 'static>(error: E) -> Error {
    Error::Instance(error.into())
}

pub type LazyBytes = Pin<Box<dyn Future<Output = Result<Bytes>> + Send + 'static>>;
pub type LazyBody<T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'static>>;

pub type ByteStream = BoxStream<'static, Result<Bytes>>;
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

pub trait HttpClientExt: Send + Sync {
    fn request<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + Send
    where
        T: Into<Bytes> + Send,
        U: From<Bytes> + Send;

    fn request_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + Send
    where
        T: Into<Bytes>;

    fn get<T>(&self, uri: Uri) -> impl Future<Output = Result<Response<LazyBody<T>>>> + Send
    where
        T: From<Bytes> + Send,
    {
        async {
            let req = Request::builder()
                .method(Method::GET)
                .uri(uri)
                .body(NoBody)?;

            self.request(req).await
        }
    }

    fn post<T, R>(
        &self,
        uri: Uri,
        body: T,
    ) -> impl Future<Output = Result<Response<LazyBody<R>>>> + Send
    where
        T: Into<Bytes> + Send,
        R: From<Bytes> + Send,
    {
        async {
            let req = Request::builder()
                .method(Method::POST)
                .uri(uri)
                .body(body)?;
            self.request(req).await
        }
    }
}

impl HttpClientExt for reqwest::Client {
    fn request<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + Send
    where
        T: Into<Bytes>,
        U: From<Bytes> + Send,
    {
        let (parts, body) = req.into_parts();
        let req = self
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers)
            .body(body.into());

        async move {
            let response = req.send().await.map_err(instance_error)?;

            let mut res = Response::builder()
                .status(response.status())
                .version(response.version());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            let body: LazyBody<U> = Box::pin(async move {
                let bytes = response.bytes().await.map_err(instance_error)?;
                let body = U::from(bytes);
                Ok(body)
            });

            res.body(body).map_err(Error::Protocol)
        }
    }

    fn request_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + Send
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

            let mut res = Response::builder()
                .status(response.status())
                .version(response.version());

            if let Some(hs) = res.headers_mut() {
                *hs = response.headers().clone();
            }

            let stream: ByteStream =
                Box::pin(response.bytes_stream().map(|r| r.map_err(instance_error)));

            Ok(res.body(stream)?)
        }
    }
}
