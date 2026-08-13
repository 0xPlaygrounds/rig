use futures::FutureExt;
use rig::providers::venice;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

// The direct-recording client below exists for the speech-synthesis scenario
// alone, so it is gated on the same feature that compiles it — the PR gate
// builds this target with the default feature set, where an ungated helper
// would be dead code under `-D warnings`.
#[cfg(feature = "audio")]
use {
    crate::cassettes::{DirectHttpRequest, DirectHttpResponse, DirectRecorder},
    bytes::Bytes,
    rig::http_client::{self, HttpClientExt, LazyBody, MultipartForm, Request, Response},
};

const VENICE_BASE_URL: &str = venice::VENICE_API_BASE_URL;

async fn venice_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, venice::Client) {
    let cassette = ProviderCassette::start("venice", spec, VENICE_BASE_URL).await;
    let client = venice::Client::builder()
        .api_key(cassette.api_key("VENICE_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn with_venice_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(venice::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = venice_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Client used by the scenarios whose *response* body is binary.
///
/// The shared proxy recorder exports cassettes through httpmock, which stores
/// bodies as strings and therefore drops a non-UTF-8 payload entirely — a
/// recorded speech response came back as `body: null` and replayed as zero
/// bytes. Venice's TTS endpoint answers with raw audio, so those scenarios
/// take the direct-recording path (the same one Bedrock's event-stream
/// cassettes use), which stores non-UTF-8 bodies as base64.
///
/// In replay mode this is a plain reqwest client pointed at the replay
/// server; only record mode carries a recorder.
#[cfg(feature = "audio")]
#[derive(Clone, Debug, Default)]
pub(super) struct DirectRecordingHttpClient {
    inner: reqwest::Client,
    recorder: Option<DirectRecorder>,
}

#[cfg(feature = "audio")]
impl HttpClientExt for DirectRecordingHttpClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + Send + 'static
    where
        T: Into<Bytes> + Send,
        U: From<Bytes> + Send + 'static,
    {
        let inner = self.inner.clone();
        let recorder = self.recorder.clone();
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        let method = parts.method.to_string();
        let uri = parts.uri.to_string();
        let request_headers = owned_headers(&parts.headers);
        let request = Request::from_parts(parts, body.clone());

        async move {
            let response = HttpClientExt::send::<Bytes, Bytes>(&inner, request).await?;
            let (parts, lazy_body) = response.into_parts();
            // Buffered, not streamed: the recorder needs the whole payload,
            // and the caller gets the same bytes back below.
            let bytes = lazy_body.await?;

            if let Some(recorder) = recorder {
                let response_headers = owned_headers(&parts.headers);
                recorder
                    .record_http_interaction(
                        DirectHttpRequest {
                            method: &method,
                            uri: &uri,
                            headers: request_headers.iter().map(|(name, value)| (name, value)),
                            body: &body,
                        },
                        DirectHttpResponse {
                            status: parts.status.as_u16(),
                            headers: response_headers.iter().map(|(name, value)| (name, value)),
                            body: &bytes,
                        },
                    )
                    .await;
            }

            let body: LazyBody<U> = Box::pin(async move { Ok(U::from(bytes)) });
            Ok(Response::from_parts(parts, body))
        }
    }

    // Only the unary path is recorded: the scenarios on this client are
    // JSON-in/audio-out. Multipart and streaming pass through so the type
    // still satisfies the trait.
    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + Send + 'static
    where
        U: From<Bytes> + Send + 'static,
    {
        self.inner.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>> + Send
    where
        T: Into<Bytes> + Send,
    {
        self.inner.send_streaming(req)
    }
}

#[cfg(feature = "audio")]
fn owned_headers(headers: &http_client::HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect()
}

/// Cassette wrapper for scenarios whose response body is binary; see
/// [`DirectRecordingHttpClient`].
#[cfg(feature = "audio")]
pub(super) async fn with_venice_direct_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(venice::Client<DirectRecordingHttpClient>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start_direct_recording("venice", spec, VENICE_BASE_URL).await;
    let http_client = DirectRecordingHttpClient {
        inner: reqwest::Client::new(),
        recorder: cassette.direct_recorder(),
    };
    let client = venice::Client::builder()
        .api_key(cassette.api_key("VENICE_API_KEY"))
        .base_url(cassette.base_url())
        .http_client(http_client)
        .build()
        .expect("client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_venice_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(venice::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = venice_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
