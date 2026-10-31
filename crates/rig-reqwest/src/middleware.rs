//! The middleware client: a `reqwest_middleware` client the transport sends
//! through.

use rig_http::http_client::{
    HttpClientExt, LazyBody, MultipartForm, Request, Response, Result, StreamingResponse,
};
use rig_http::wasm_compat::WasmCompatSend;

use super::{
    Bytes, Error, Execute, Payload, from_reqwest, into_response, into_streaming_response,
    is_transient, multipart_form, send_with,
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
/// does. A middleware's error, with any context it added, is classified as
/// the first reqwest or rig error in its source chain, looking through
/// middleware errors it wraps; a middleware that knows its failure is final
/// says so with [`Error::permanent`]. An error the chain does not classify
/// is [`Error::Transient`], since it may hide a dropped connection.
pub(crate) fn from_middleware(error: reqwest_middleware::Error) -> Error {
    let error = match error {
        reqwest_middleware::Error::Reqwest(error) => return from_reqwest(error),
        reqwest_middleware::Error::Middleware(error) => error,
    };
    if wrapped_transient(&*error) {
        Error::transient(error)
    } else {
        Error::permanent(error)
    }
}

/// Whether the first reqwest or rig error in `error`'s source chain may
/// pass on a second send; true when there is none.
fn wrapped_transient(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut link = Some(error);
    while let Some(error) = link {
        if let Some(error) = error.downcast_ref::<reqwest::Error>() {
            return is_transient(error);
        }
        match error.downcast_ref::<reqwest_middleware::Error>() {
            Some(reqwest_middleware::Error::Reqwest(error)) => return is_transient(error),
            Some(reqwest_middleware::Error::Middleware(error)) => {
                return wrapped_transient(&**error);
            }
            None => {}
        }
        if let Some(error) = error.downcast_ref::<Error>() {
            return error.is_transient();
        }
        link = error.source();
    }
    true
}

http_client_ext!(ReqwestMiddlewareClient);
