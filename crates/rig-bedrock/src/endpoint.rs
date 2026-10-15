//! Bedrock's HTTP inference APIs: Anthropic's Messages API for Claude, and
//! the OpenAI-compatible Chat Completions and Responses APIs, on either the
//! `bedrock-runtime` or the `bedrock-mantle` endpoint. Requests go through
//! Rig's own Anthropic and OpenAI wires, authenticated with SigV4 from AWS
//! credentials or with a Bedrock API key.
//!
//! On `bedrock-mantle` most OpenAI-compatible models are served under `/v1`
//! and some, such as OpenAI's own GPT models, under `/openai/v1`; each
//! model's card in the Bedrock documentation names its path. Bedrock stores
//! Responses for 30 days unless a request sets `store: false`.
//!
//! ```no_run
//! use rig_bedrock::endpoint::{BedrockEndpoint, Host, MANTLE_CLAUDE_SONNET_5, MANTLE_GPT_OSS_120B};
//! use rig_core::completion::CompletionRequest;
//!
//! # async fn run(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
//! let bedrock = BedrockEndpoint::from_env(Host::Mantle, http).await?;
//! let claude = bedrock.messages(MANTLE_CLAUDE_SONNET_5);
//! let reply = claude.call(CompletionRequest::new("Capital of France?")).await?;
//! let gpt = bedrock.openai().chat(MANTLE_GPT_OSS_120B);
//! # let _ = (reply, gpt);
//! # Ok(())
//! # }
//! ```

use aws_config::{BehaviorVersion, SdkConfig};
use aws_credential_types::provider::SharedCredentialsProvider;
use rig_core::Model;
use rig_core::catalog::ModelFacts;
use rig_core::client::env;
use rig_core::http_client::{DynHttpClient, HttpClientExt};
use rig_core::providers::anthropic::{self, AnthropicConfig, Messages, ToolInputStreaming};
use rig_core::providers::openai::{self, OpenAI, OpenAIConfig};
use rig_core::providers::registry::ProviderId;
use rig_core::wire::Secret;

use crate::completion::PROVIDER_NAME;
use crate::sigv4::{AwsError, SigV4Http};

/// The environment variable AWS tools read a Bedrock API key from.
pub const API_KEY_ENV: &str = "AWS_BEARER_TOKEN_BEDROCK";

/// Claude Sonnet 5 on `bedrock-mantle`.
pub const MANTLE_CLAUDE_SONNET_5: &str = "anthropic.claude-sonnet-5";
/// gpt-oss-120b on `bedrock-mantle`, served under `/v1`.
pub const MANTLE_GPT_OSS_120B: &str = "openai.gpt-oss-120b";
/// gpt-oss-120b on `bedrock-runtime`, Chat Completions only.
pub const RUNTIME_GPT_OSS_120B: &str = "openai.gpt-oss-120b-1:0";
/// gpt-oss-20b on `bedrock-runtime`, Chat Completions only.
pub const RUNTIME_GPT_OSS_20B: &str = "openai.gpt-oss-20b-1:0";

/// Bedrock's Messages-format endpoint. Tool input is not requested as it
/// streams, since Bedrock does not document either of Anthropic's ways to
/// ask for it.
pub const MESSAGES_DIALECT: anthropic::Dialect = anthropic::Dialect {
    request_id_header: None,
    quirks: messages_quirks(),
    ..anthropic::compatible(PROVIDER_NAME, "", API_KEY_ENV, None)
};

/// Bedrock's OpenAI-compatible endpoint. Chat Completions is the default
/// route because every OpenAI-compatible model on both endpoints serves
/// it. Neither endpoint offers a credential check that spends no tokens.
pub const OPENAI_DIALECT: openai::wire::Dialect = openai::wire::Dialect {
    quirks: openai_quirks(),
    ..openai::wire::Dialect::gateway(PROVIDER_NAME, "", API_KEY_ENV)
};

const fn messages_quirks() -> anthropic::Quirks {
    let mut quirks = anthropic::Quirks::anthropic();
    quirks.tool_input_streaming = ToolInputStreaming::Off;
    quirks
}

const fn openai_quirks() -> openai::wire::Quirks {
    let mut quirks = openai::wire::Quirks::openai();
    quirks.verify_path = "";
    quirks
}

/// Which Bedrock inference endpoint requests go to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Host {
    /// `bedrock-runtime.{region}.amazonaws.com`, signed as `bedrock`.
    Runtime,
    /// `bedrock-mantle.{region}.api.aws`, signed as `bedrock-mantle`.
    Mantle,
}

impl Host {
    /// The endpoint's root URL in `region`.
    pub fn url(self, region: &str) -> String {
        match self {
            Self::Runtime => format!("https://bedrock-runtime.{region}.amazonaws.com"),
            Self::Mantle => format!("https://bedrock-mantle.{region}.api.aws"),
        }
    }

    /// The service name SigV4 signatures for this endpoint are scoped to.
    pub fn signing_name(self) -> &'static str {
        match self {
            Self::Runtime => "bedrock",
            Self::Mantle => "bedrock-mantle",
        }
    }

    /// The path the OpenAI-compatible APIs are served under by default.
    pub fn openai_path(self) -> &'static str {
        match self {
            Self::Runtime => "/openai/v1",
            Self::Mantle => "/v1",
        }
    }
}

/// How requests to Bedrock are authenticated.
#[derive(Clone, Debug)]
pub enum Credential {
    /// SigV4 signatures from AWS credentials, refreshed as the provider
    /// refreshes them.
    SigV4(SharedCredentialsProvider),
    /// A Bedrock API key.
    ApiKey(Secret),
}

/// A Bedrock HTTP endpoint in one region with its credential, and the
/// models it serves over Rig's Anthropic and OpenAI wires. Clones share the
/// transport.
#[derive(Clone, Debug)]
pub struct BedrockEndpoint {
    host: Host,
    region: String,
    api_key: Secret,
    http: DynHttpClient,
}

impl BedrockEndpoint {
    /// `host` in `region`, authenticated with `credential`, sending through
    /// `http`.
    pub fn new(
        host: Host,
        region: impl Into<String>,
        credential: Credential,
        http: impl HttpClientExt + 'static,
    ) -> Self {
        let region = region.into();
        let (api_key, http) = match credential {
            Credential::SigV4(credentials) => (
                Secret::from(""),
                DynHttpClient::new(SigV4Http::new(
                    http,
                    credentials,
                    region.clone(),
                    host.signing_name(),
                )),
            ),
            Credential::ApiKey(key) => (key, DynHttpClient::new(http)),
        };
        Self {
            host,
            region,
            api_key,
            http,
        }
    }

    /// `host` in the region of `config`, signed with its credentials.
    pub fn from_sdk_config(
        host: Host,
        config: &SdkConfig,
        http: impl HttpClientExt + 'static,
    ) -> Result<Self, AwsError> {
        let region = config.region().ok_or(AwsError::NoRegion)?;
        let credentials = config
            .credentials_provider()
            .ok_or(AwsError::NoCredentialsProvider)?;
        Ok(Self::new(
            host,
            region.as_ref(),
            Credential::SigV4(credentials),
            http,
        ))
    }

    /// `host` in the region the AWS SDK's default chain resolves. A
    /// non-empty [`API_KEY_ENV`] is the credential when set, as the AWS
    /// SDKs treat it; otherwise requests are signed with the default
    /// credential chain.
    pub async fn from_env(
        host: Host,
        http: impl HttpClientExt + 'static,
    ) -> Result<Self, AwsError> {
        let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
        match env::optional(API_KEY_ENV)?.filter(|key| !key.is_empty()) {
            Some(key) => {
                let region = config.region().ok_or(AwsError::NoRegion)?;
                Ok(Self::new(
                    host,
                    region.as_ref(),
                    Credential::ApiKey(key.into()),
                    http,
                ))
            }
            None => Self::from_sdk_config(host, &config, http),
        }
    }

    /// The endpoint requests go to.
    pub fn host(&self) -> Host {
        self.host
    }

    /// The region requests go to.
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Claude `model` over Anthropic's Messages API (`/anthropic/v1/messages`).
    /// A model the catalog lists under Bedrock gets that row's limits, so a
    /// request without `max_tokens` takes the model's output limit.
    pub fn messages(&self, model: impl Into<String>) -> Model<Messages> {
        let model = model.into();
        let facts = claude_facts(&model);
        let mut messages = AnthropicConfig::with_key(&MESSAGES_DIALECT, self.api_key.clone())
            .with_base_url(format!("{}/anthropic", self.host.url(&self.region)))
            .connect(self.http.clone())
            .completion(model);
        if let Some(facts) = facts {
            messages.wire = messages.wire.with_facts(facts);
        }
        messages
    }

    /// The OpenAI-compatible APIs at the endpoint's default path
    /// ([`Host::openai_path`]).
    pub fn openai(&self) -> OpenAI {
        self.openai_at(self.host.openai_path())
    }

    /// The OpenAI-compatible APIs under `path`, for a `bedrock-mantle` model
    /// served under `/openai/v1`.
    pub fn openai_at(&self, path: &str) -> OpenAI {
        OpenAIConfig::with_key(&OPENAI_DIALECT, self.api_key.clone())
            .with_base_url(format!("{}{path}", self.host.url(&self.region)))
            .connect(self.http.clone())
    }
}

/// The facts of the Bedrock Claude `model`, under Anthropic's name so the
/// Messages wire reads them. `None` for a model the catalog does not list.
fn claude_facts(model: &str) -> Option<ModelFacts> {
    let anthropic = ProviderId::catalog(anthropic::PROVIDER_NAME)?;
    let mut spec = crate::completion::spec_in(&ModelFacts::default(), model)?.clone();
    spec.id = model.to_owned();
    spec.provider = anthropic;
    Some(ModelFacts::new(spec))
}

#[cfg(test)]
mod tests;
