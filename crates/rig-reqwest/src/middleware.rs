//! The middleware client: a `reqwest_middleware` client the transport sends
//! through.

use rig_http::http_client::{
    HttpClientExt, LazyBody, MultipartForm, Request, Response, Result, StreamingResponse,
};
use rig_http::wasm_compat::WasmCompatSend;

use super::{
    Bytes, Error, Execute, Payload, from_reqwest, into_response, into_streaming_response,
    multipart_form, send_with,
};

/// A configured middleware client implementing [`HttpClientExt`].
///
/// Build the inner client with `reqwest_middleware::ClientBuilder`, then wrap
/// it with `From<ClientWithMiddleware>`.
#[cfg_attr(
    docsrs,
    doc(cfg(any(
        feature = "reqwest-middleware-rustls",
        feature = "reqwest-middleware-native-tls"
    )))
)]
#[derive(Clone, Debug)]
pub struct ReqwestMiddlewareClient(reqwest_middleware::ClientWithMiddleware);

impl ReqwestMiddlewareClient {
    /// Take the inner client back.
    #[must_use]
    pub fn into_inner(self) -> reqwest_middleware::ClientWithMiddleware {
        self.0
    }
}

impl From<reqwest_middleware::ClientWithMiddleware> for ReqwestMiddlewareClient {
    fn from(client: reqwest_middleware::ClientWithMiddleware) -> Self {
        Self(client)
    }
}

impl AsRef<reqwest_middleware::ClientWithMiddleware> for ReqwestMiddlewareClient {
    fn as_ref(&self) -> &reqwest_middleware::ClientWithMiddleware {
        &self.0
    }
}

impl Execute for ReqwestMiddlewareClient {
    fn execute(
        &self,
        parts: http::request::Parts,
        payload: Result<Payload>,
    ) -> Result<impl Future<Output = Result<reqwest::Response>> + WasmCompatSend + 'static> {
        let request = self
            .0
            .request(parts.method, parts.uri.to_string())
            .headers(parts.headers);
        let request = match payload? {
            Payload::Bytes(bytes) => request.body(bytes),
            Payload::Form(form) => request.multipart(form),
        };
        Ok(async move { request.send().await.map_err(from_middleware) })
    }
}

/// Classify a middleware client's error: reqwest's own as [`from_reqwest`]
/// does, an [`Error`] a middleware returned as it is, and any other
/// middleware failure as [`Error::Permanent`], since nothing says that
/// sending again would help.
fn from_middleware(error: reqwest_middleware::Error) -> Error {
    match error {
        reqwest_middleware::Error::Reqwest(error) => from_reqwest(error),
        reqwest_middleware::Error::Middleware(error) => match error.downcast::<reqwest::Error>() {
            Ok(error) => from_reqwest(error),
            Err(error) => match error.downcast::<Error>() {
                Ok(error) => error,
                Err(error) => Error::permanent(error),
            },
        },
    }
}

http_client_ext!(ReqwestMiddlewareClient);
