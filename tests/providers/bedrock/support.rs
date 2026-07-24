use std::future::Future;
use std::panic::AssertUnwindSafe;

use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::config::Credentials;
use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpConnector,
};
use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_runtime_api::http::StatusCode;
use aws_smithy_types::body::SdkBody;
use futures::FutureExt;
use rig::bedrock::client::Client;
use rig::bedrock::mantle::{self, CompletionsClient, ResponsesClient};

use crate::cassettes::{
    CassetteMode, CassetteSpec, DirectHttpRequest, DirectHttpResponse, DirectRecorder,
    ProviderCassette,
};

const BEDROCK_REAL_BASE_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com";
const BEDROCK_REGION: &str = "us-east-1";
/// Mantle OpenAI-compatible default base (GPT-OSS / Completions + Responses).
/// Uses HTTP ProviderCassette (bearer auth), not Converse SigV4 direct recording.
const MANTLE_REAL_BASE_URL: &str = "https://bedrock-mantle.us-east-1.api.aws/v1";
/// Mantle GPT-5.x Responses base (Luna / Sol / Terra / gpt-5.4 / gpt-5.5).
const MANTLE_GPT5_REAL_BASE_URL: &str = "https://bedrock-mantle.us-east-1.api.aws/openai/v1";

async fn bedrock_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, Client) {
    match CassetteMode::current() {
        CassetteMode::Replay => replay_bedrock_cassette(spec).await,
        CassetteMode::Record => record_bedrock_cassette(spec).await,
    }
}

async fn replay_bedrock_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, Client) {
    let cassette =
        ProviderCassette::start_direct_recording("bedrock", spec, BEDROCK_REAL_BASE_URL).await;
    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(BEDROCK_REGION))
        .credentials_provider(Credentials::new(
            "test-access-key",
            "test-secret-key",
            None,
            None,
            "rig-bedrock-cassette",
        ))
        .endpoint_url(cassette.base_url())
        .load()
        .await;
    let aws_client = aws_sdk_bedrockruntime::Client::new(&sdk_config);
    let client = Client::from(aws_client);

    (cassette, client)
}

async fn record_bedrock_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, Client) {
    let cassette =
        ProviderCassette::start_direct_recording("bedrock", spec, BEDROCK_REAL_BASE_URL).await;
    let recorder = cassette
        .direct_recorder()
        .expect("Bedrock record mode should use a direct recorder");
    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(BEDROCK_REGION))
        .load()
        .await;
    let bedrock_config = aws_sdk_bedrockruntime::config::Builder::from(&sdk_config)
        .region(Region::new(BEDROCK_REGION))
        .endpoint_url(BEDROCK_REAL_BASE_URL)
        .http_client(RecordingBedrockHttpClient::new(recorder))
        .build();
    let aws_client = aws_sdk_bedrockruntime::Client::from_conf(bedrock_config);
    let client = Client::from(aws_client);

    (cassette, client)
}

#[derive(Clone, Debug)]
struct RecordingBedrockHttpClient {
    connector: SharedHttpConnector,
}

impl RecordingBedrockHttpClient {
    fn new(recorder: DirectRecorder) -> Self {
        Self {
            connector: SharedHttpConnector::new(RecordingBedrockConnector {
                client: reqwest::Client::new(),
                recorder,
            }),
        }
    }
}

impl HttpClient for RecordingBedrockHttpClient {
    fn http_connector(
        &self,
        _settings: &HttpConnectorSettings,
        _components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        self.connector.clone()
    }
}

// The direct Bedrock recorder buffers in-memory request bodies and full response
// bodies before handing the response back to the AWS SDK. This lets Bedrock
// event-stream responses replay from binary cassette bodies without proxying or
// rewriting SigV4-signed requests.
#[derive(Clone, Debug)]
struct RecordingBedrockConnector {
    client: reqwest::Client,
    recorder: DirectRecorder,
}

impl HttpConnector for RecordingBedrockConnector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        let client = self.client.clone();
        let recorder = self.recorder.clone();
        HttpConnectorFuture::new(async move {
            let method = request.method().to_string();
            let uri = request.uri().to_string();
            let request_headers = request
                .headers()
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect::<Vec<_>>();
            let request_body = request
                .body()
                .bytes()
                .ok_or_else(|| {
                    ConnectorError::user(
                        "Bedrock cassette record mode only supports in-memory request bodies"
                            .into(),
                    )
                })?
                .to_vec();

            let mut builder = client.request(
                method
                    .parse()
                    .map_err(|error| ConnectorError::user(Box::new(error)))?,
                &uri,
            );
            for (name, value) in &request_headers {
                builder = builder.header(name.as_str(), value.as_str());
            }
            let response = builder
                .body(request_body.clone())
                .send()
                .await
                .map_err(|error| ConnectorError::io(Box::new(error)))?;
            let status = response.status().as_u16();
            let response_headers = response
                .headers()
                .iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|value| (name.as_str().to_string(), value.to_string()))
                })
                .collect::<Vec<_>>();
            let response_body = response
                .bytes()
                .await
                .map_err(|error| ConnectorError::io(Box::new(error)))?;

            recorder
                .record_http_interaction(
                    DirectHttpRequest {
                        method: &method,
                        uri: &uri,
                        headers: request_headers.iter().map(|(name, value)| (name, value)),
                        body: &request_body,
                    },
                    DirectHttpResponse {
                        status,
                        headers: response_headers.iter().map(|(name, value)| (name, value)),
                        body: &response_body,
                    },
                )
                .await;

            let mut response = HttpResponse::new(
                StatusCode::try_from(status)
                    .map_err(|error| ConnectorError::user(Box::new(error)))?,
                SdkBody::from(response_body.to_vec()),
            );
            for (name, value) in response_headers {
                response
                    .headers_mut()
                    .try_append(name, value)
                    .map_err(|error| ConnectorError::user(Box::new(error)))?;
            }

            Ok(response)
        })
    }
}

pub(super) async fn with_bedrock_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = bedrock_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Resolve a Mantle bearer token for cassette **record** mode.
///
/// Prefer `AWS_BEARER_TOKEN_BEDROCK` when set; otherwise mint a short-term IAM token.
/// Replay mode never calls this — it uses a dummy key (auth is not matched on wire).
async fn mantle_record_api_key() -> String {
    match std::env::var(mantle::AWS_BEARER_TOKEN_BEDROCK_ENV) {
        Ok(token) if !token.is_empty() => token,
        _ => mantle::generate_short_term_token(BEDROCK_REGION)
            .await
            .expect("mint Mantle short-term token for cassette record mode"),
    }
}

async fn bedrock_mantle_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, ResponsesClient) {
    let cassette = ProviderCassette::start("bedrock", spec, MANTLE_REAL_BASE_URL).await;
    let api_key = match CassetteMode::current() {
        CassetteMode::Record => mantle_record_api_key().await,
        CassetteMode::Replay => "bedrock-api-key-cassette-replay".to_string(),
    };
    let client = mantle::ClientBuilder::new()
        .api_key(api_key)
        .base_url(cassette.base_url())
        .build()
        .await
        .expect("mantle responses client should build");
    (cassette, client)
}

async fn bedrock_mantle_completions_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, CompletionsClient) {
    let cassette = ProviderCassette::start("bedrock", spec, MANTLE_REAL_BASE_URL).await;
    let api_key = match CassetteMode::current() {
        CassetteMode::Record => mantle_record_api_key().await,
        CassetteMode::Replay => "bedrock-api-key-cassette-replay".to_string(),
    };
    let client = mantle::ClientBuilder::new()
        .api_key(api_key)
        .base_url(cassette.base_url())
        .build_completions()
        .await
        .expect("mantle completions client should build");
    (cassette, client)
}

/// Mantle Responses HTTP cassette (OpenAI-compatible bearer auth, not Converse SigV4).
pub(super) async fn with_bedrock_mantle_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(ResponsesClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = bedrock_mantle_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Mantle Completions HTTP cassette (OpenAI-compatible bearer auth).
pub(super) async fn with_bedrock_mantle_completions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(CompletionsClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = bedrock_mantle_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

async fn bedrock_mantle_gpt5_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, ResponsesClient) {
    let cassette = ProviderCassette::start("bedrock", spec, MANTLE_GPT5_REAL_BASE_URL).await;
    let api_key = match CassetteMode::current() {
        CassetteMode::Record => mantle_record_api_key().await,
        CassetteMode::Replay => "bedrock-api-key-cassette-replay".to_string(),
    };
    let client = mantle::ClientBuilder::new()
        .api_key(api_key)
        .base_url(cassette.base_url())
        .build()
        .await
        .expect("mantle GPT-5 Responses client should build");
    (cassette, client)
}

/// Mantle GPT-5.x Responses HTTP cassette (`/openai/v1` base: Luna / Sol / Terra / …).
pub(super) async fn with_bedrock_mantle_gpt5_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(ResponsesClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = bedrock_mantle_gpt5_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
