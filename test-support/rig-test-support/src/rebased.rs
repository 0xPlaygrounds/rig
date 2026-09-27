//! An HTTP client that sends a provider's requests to a cassette server: a
//! request under `from` goes to the same path under `to`. A model built from
//! data (a registered `ProviderRef`) addresses the provider's real host, and
//! this is how its replay reaches the recording.

use bytes::Bytes;
use rig_core::http_client::{
    DynHttpClient, HttpClientExt, LazyBody, MultipartForm, Request, Response, Result,
    StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;

/// Requests under `from` are sent under `to` through `inner`; any other
/// request is sent unchanged.
#[derive(Clone, Debug)]
pub struct Rebased {
    from: String,
    to: String,
    inner: DynHttpClient,
}

impl Rebased {
    /// Send requests under `from` to `to`, through `inner`.
    pub fn new(from: impl Into<String>, to: impl Into<String>, inner: DynHttpClient) -> Self {
        Self {
            from: from.into().trim_end_matches('/').to_owned(),
            to: to.into().trim_end_matches('/').to_owned(),
            inner,
        }
    }

    fn rebase<T>(&self, mut request: Request<T>) -> Request<T> {
        let uri = request.uri().to_string();
        if let Some(rest) = uri.strip_prefix(&self.from)
            && let Ok(rebased) = format!("{}{rest}", self.to).parse()
        {
            *request.uri_mut() = rebased;
        }
        request
    }
}

impl HttpClientExt for Rebased {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        self.inner.send(self.rebase(req))
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        self.inner.send_multipart(self.rebase(req))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        self.inner.send_streaming(self.rebase(req))
    }
}
