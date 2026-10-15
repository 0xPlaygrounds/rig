use bytes::Bytes;
use futures::StreamExt;
use http::header::{CONTENT_TYPE, HeaderValue};
use rig_http::http_client::{
    BoxedStream, Error, HttpClientExt, LazyBody, MultipartForm, Request, Response, Result,
    StreamingResponse,
};
use rig_http::wasm_compat::WasmCompatSend;
use wstd::http::{Body, BodyExt, Client};

/// A `wasi:http` client implementing [`HttpClientExt`].
///
/// [`Default`] uses the host's timeouts. `From<wstd::http::Client>` wraps a
/// client configured with its own connect, first-byte and between-bytes
/// timeouts.
#[derive(Clone, Debug, Default)]
pub struct WstdClient(Client);

impl WstdClient {
    /// The wstd client this transport sends through.
    pub fn inner(&self) -> &Client {
        &self.0
    }
}

impl From<Client> for WstdClient {
    fn from(client: Client) -> Self {
        Self(client)
    }
}

/// A wstd failure without a response, as [`Error::Instance`].
fn instance(error: wstd::http::Error) -> Error {
    Error::Instance(error.into())
}

async fn send(client: Client, req: Request<Bytes>) -> Result<Response<Body>> {
    client.send(req).await.map_err(instance)
}

async fn collect(body: Body) -> Result<Bytes> {
    let collected = body.into_boxed_body().collect().await.map_err(instance)?;
    Ok(collected.to_bytes())
}

/// The headers-preserving error for a non-success response, with its body
/// read as text.
async fn non_success(response: Response<Body>) -> Error {
    let (parts, body) = response.into_parts();
    let body = match collect(body).await {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(error) => format!("failed to read error response body: {error}"),
    };
    Error::non_success_with_details(parts.status, parts.headers, body)
}

/// A successful response with its body read when awaited.
async fn into_response<U>(response: Response<Body>) -> Result<Response<LazyBody<U>>>
where
    U: From<Bytes> + WasmCompatSend + 'static,
{
    if !response.status().is_success() {
        return Err(non_success(response).await);
    }
    let (parts, body) = response.into_parts();
    let body: LazyBody<U> = Box::pin(async move { collect(body).await.map(U::from) });
    Ok(Response::from_parts(parts, body))
}

/// A successful response with its body streamed chunk by chunk.
async fn into_streaming_response(response: Response<Body>) -> Result<StreamingResponse> {
    if !response.status().is_success() {
        return Err(non_success(response).await);
    }
    let (parts, body) = response.into_parts();
    let stream = body
        .into_boxed_body()
        .into_data_stream()
        .map(|chunk| chunk.map_err(instance));
    let stream: BoxedStream = Box::pin(stream);
    Ok(Response::from_parts(parts, stream))
}

/// `req` as a multipart request: the encoded form as its body and the
/// matching `Content-Type`.
fn multipart(req: Request<MultipartForm>) -> Result<Request<Bytes>> {
    let (mut parts, form) = req.into_parts();
    let (boundary, body) = form.encode();
    let content_type = HeaderValue::from_str(&format!("multipart/form-data; boundary={boundary}"))?;
    parts.headers.insert(CONTENT_TYPE, content_type);
    Ok(Request::from_parts(parts, body))
}

impl HttpClientExt for WstdClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let client = self.0.clone();
        let req = req.map(Into::into);
        async move { into_response(send(client, req).await?).await }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let client = self.0.clone();
        let req = multipart(req);
        async move { into_response(send(client, req?).await?).await }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let client = self.0.clone();
        let req = req.map(Into::into);
        async move { into_streaming_response(send(client, req).await?).await }
    }
}

#[cfg(test)]
mod tests;
