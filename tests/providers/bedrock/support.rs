use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

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
use rig::agent::AgentBuilder;
use rig::bedrock::functions::Config;
use rig::provider::{ProviderConfig, Runtime};

use crate::cassettes::{
    CassetteMode, CassetteSpec, DirectHttpRequest, DirectHttpResponse, DirectRecorder,
    ProviderCassette,
};

const BEDROCK_REAL_BASE_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com";
const BEDROCK_REGION: &str = "us-east-1";

/// The per-test Bedrock handle: the cassette-wired AWS SDK client (which the
/// `rig::bedrock::functions::*` free functions take directly) plus a provider
/// [`Runtime`] whose Bedrock cache is seeded with that same client, so
/// `ProviderConfig::Bedrock` agents replay through the recording/replaying
/// transport.
pub(super) struct BedrockHarness {
    aws_client: aws_sdk_bedrockruntime::Client,
    runtime: Arc<Runtime>,
}

impl BedrockHarness {
    fn new(aws_client: aws_sdk_bedrockruntime::Client) -> Self {
        Self {
            aws_client,
            runtime: Arc::new(Runtime::new()),
        }
    }

    /// The cassette-wired AWS client, for the model-level
    /// `rig::bedrock::functions::{complete, open_stream, embed}` entry points.
    pub fn aws_client(&self) -> &aws_sdk_bedrockruntime::Client {
        &self.aws_client
    }

    /// An [`AgentBuilder`] over `cfg`, with this harness's AWS client seeded
    /// into the runtime cache for exactly that config.
    pub async fn agent_from_config(&self, cfg: Config) -> AgentBuilder {
        self.runtime
            .seed_bedrock_client(cfg.clone(), self.aws_client.clone())
            .await;
        AgentBuilder::new(ProviderConfig::Bedrock(cfg)).runtime(self.runtime.clone())
    }

    /// An [`AgentBuilder`] for `model` with default Bedrock configuration.
    pub async fn agent(&self, model: &str) -> AgentBuilder {
        self.agent_from_config(Config::new(model)).await
    }
}

async fn bedrock_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, BedrockHarness) {
    let (cassette, aws_client) = match CassetteMode::current() {
        CassetteMode::Replay => replay_bedrock_cassette(spec).await,
        CassetteMode::Record => record_bedrock_cassette(spec).await,
    };
    (cassette, BedrockHarness::new(aws_client))
}

async fn replay_bedrock_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, aws_sdk_bedrockruntime::Client) {
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

    (cassette, aws_client)
}

async fn record_bedrock_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, aws_sdk_bedrockruntime::Client) {
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

    (cassette, aws_client)
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
    F: FnOnce(BedrockHarness) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, harness) = bedrock_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(harness)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
