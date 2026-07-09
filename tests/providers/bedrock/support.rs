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

use crate::cassettes::{CassetteMode, CassetteSpec, DirectRecorder, ProviderCassette};

const BEDROCK_REAL_BASE_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com";
const BEDROCK_REGION: &str = "us-east-1";

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
                    &method,
                    &uri,
                    request_headers.iter().map(|(name, value)| (name, value)),
                    &request_body,
                    status,
                    response_headers.iter().map(|(name, value)| (name, value)),
                    &response_body,
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
