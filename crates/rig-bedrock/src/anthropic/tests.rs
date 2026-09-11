//! Tests for the AWS-fronted Anthropic provider.
//!
//! The one that earns its keep is `a_sigv4_client_signs_the_request_rig_core_builds`. The signing
//! implementation lives here and the request builder that has to invoke it lives in rig-core, so
//! every piece can be individually correct while the two are not actually connected — a hook that
//! is never called returns nothing, and an unsigned request fails with a 403 that reads like a
//! credential problem rather than a wiring one. That test drives a real completion through rig-core
//! and asserts the signature arrived on the captured request.

use super::*;
use futures::StreamExt;
use rig_core::client::CompletionClient;
use rig_core::completion::{CompletionModel as _, CompletionRequest};
use rig_core::http_client::{Request, Response, StreamingResponse};
use rig_core::message;
use rig_core::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig_core::test_utils::RecordingHttpClient;
use rig_core::wasm_compat::WasmCompatSend;
use std::sync::{Arc, Mutex};

const ENDPOINT: &str = "https://bedrock-mantle.us-east-1.api.aws/anthropic";

fn completion_request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![message::Message::from("Hello")],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Drive a completion and return the request the transport saw.
///
/// The mock response body is not a Messages response, so the call fails while decoding. That is
/// deliberate: the request has already been handed to the transport by then, and the request is what
/// these tests are about.
async fn captured_request(key: AnthropicKey) -> rig_core::test_utils::CapturedHttpRequest {
    let http = RecordingHttpClient::new("{}");
    let client = Client::builder()
        .api_key(key)
        .base_url(ENDPOINT)
        .http_client(http.clone())
        .build()
        .expect("a client with an endpoint and a key should build");

    let _ = client
        .completion_model(CLAUDE_SONNET_4_6)
        .completion(completion_request())
        .await;

    let mut requests = http.requests();
    // Anti-vacuity. With no captured request, every "header is absent" assertion below would pass
    // trivially, and so would the whole point of the test.
    assert_eq!(
        requests.len(),
        1,
        "expected exactly one request to reach the transport"
    );
    requests.remove(0)
}

#[test]
fn a_sigv4_key_sends_no_auth_header() {
    assert!(
        AnthropicKey::sigv4("us-east-1").into_header().is_none(),
        "a signed request must not also carry a static auth header"
    );
}

#[test]
fn an_api_key_still_sends_x_api_key() {
    let (name, value) = AnthropicKey::from("secret")
        .into_header()
        .expect("the api-key variant produces a header")
        .expect("the header value is valid");

    assert_eq!(name, "x-api-key");
    assert_eq!(value, "secret");
}

#[test]
fn building_without_an_endpoint_is_rejected() {
    let error = Client::builder()
        .api_key(AnthropicKey::sigv4("us-east-1"))
        .http_client(RecordingHttpClient::new(""))
        .build()
        .expect_err("there is no default endpoint, so this must not build");

    let message = error.to_string();
    assert!(
        message.contains("ANTHROPIC_BASE_URL"),
        "the error should name how to supply an endpoint: {message}"
    );
}

/// The request rig-core builds must carry the signature this crate computes.
#[tokio::test]
async fn a_sigv4_client_signs_the_request_rig_core_builds() {
    install_static_test_credentials();

    let request = captured_request(AnthropicKey::sigv4("us-east-1")).await;
    let headers = &request.headers;

    let authorization = headers
        .get(http::header::AUTHORIZATION)
        .expect("rig-core did not apply the headers this provider returned")
        .to_str()
        .expect("an authorization header is ASCII");

    assert!(
        authorization.starts_with("AWS4-HMAC-SHA256"),
        "not a SigV4 authorization header: {authorization}"
    );
    // Pins the credential scope, which is the part that silently 403s when it is wrong: the region
    // comes from the key and the service name is `bedrock-mantle`, not `bedrock`.
    assert!(
        authorization.contains("/us-east-1/bedrock-mantle/aws4_request"),
        "wrong credential scope: {authorization}"
    );
    assert!(
        headers.contains_key("x-amz-date"),
        "a SigV4 request must carry x-amz-date"
    );
    assert!(
        !headers.contains_key("x-api-key"),
        "a signed request must not also carry an API key"
    );
    // The client sets its own `Host`; see `sigv4::tests::signed_headers_never_include_host`.
    assert!(
        !headers.contains_key(http::header::HOST),
        "the signer must not add a second host header"
    );
}

/// Records the headers of streaming requests, then fails.
///
/// rig-core's bundled mocks record unary requests only, and the streaming path signs at its own
/// call site — a duplicate of the unary one, because the two paths build their requests
/// independently. Failing after recording is enough: signing happens before the request reaches the
/// transport, so the headers are already final by the time this is called.
#[derive(Clone, Debug, Default)]
struct StreamingHeaderRecorder {
    headers: Arc<Mutex<Vec<http::HeaderMap>>>,
}

impl StreamingHeaderRecorder {
    fn recorded(&self) -> Vec<http::HeaderMap> {
        self.headers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl rig_core::http_client::HttpClientExt for StreamingHeaderRecorder {
    fn send<T, U>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<rig_core::http_client::LazyBody<U>>>>
    + WasmCompatSend
    + 'static
    where
        T: Into<bytes::Bytes> + WasmCompatSend,
        U: From<bytes::Bytes> + WasmCompatSend + 'static,
    {
        futures::future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_multipart<U>(
        &self,
        _req: Request<rig_core::http_client::MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<rig_core::http_client::LazyBody<U>>>>
    + WasmCompatSend
    + 'static
    where
        U: From<bytes::Bytes> + WasmCompatSend + 'static,
    {
        futures::future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<bytes::Bytes> + WasmCompatSend,
    {
        self.headers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(req.headers().clone());

        futures::future::ready(Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::NOT_IMPLEMENTED,
        )))
    }
}

/// The streaming path signs too. It is a second call site, so it can regress on its own.
#[tokio::test]
async fn the_streaming_path_signs_as_well() {
    install_static_test_credentials();

    let http = StreamingHeaderRecorder::default();
    let client = Client::builder()
        .api_key(AnthropicKey::sigv4("us-east-1"))
        .base_url(ENDPOINT)
        .http_client(http.clone())
        .build()
        .expect("a sigv4 client with an endpoint should build");

    // The transport is only reached when the stream is polled, so open it and drain it. It fails on
    // the first poll, which is after the request — and its headers — were handed over.
    if let Ok(mut stream) = client
        .completion_model(CLAUDE_SONNET_4_6)
        .stream(completion_request())
        .await
    {
        while stream.next().await.is_some() {}
    }

    let recorded = http.recorded();
    // Anti-vacuity: with nothing recorded the assertions below prove nothing.
    assert_eq!(
        recorded.len(),
        1,
        "expected exactly one streaming request to reach the transport"
    );
    let authorization = recorded[0]
        .get(http::header::AUTHORIZATION)
        .expect("the streaming path did not apply the signing headers")
        .to_str()
        .expect("an authorization header is ASCII");

    assert!(
        authorization.starts_with("AWS4-HMAC-SHA256"),
        "not a SigV4 authorization header: {authorization}"
    );
    assert!(
        authorization.contains("/us-east-1/bedrock-mantle/aws4_request"),
        "wrong credential scope: {authorization}"
    );
}

/// The same provider must not sign when it was given an API key, and must send the key instead.
#[tokio::test]
async fn an_api_key_client_signs_nothing() {
    let request = captured_request(AnthropicKey::from("secret")).await;
    let headers = &request.headers;

    assert_eq!(
        headers
            .get("x-api-key")
            .expect("the api-key variant sends x-api-key"),
        "secret"
    );
    assert!(
        !headers.contains_key(http::header::AUTHORIZATION),
        "nothing should have been signed: {headers:?}"
    );
}
