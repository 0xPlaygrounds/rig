//! AWS Bedrock as config + free functions (data-oriented face).
//!
//! Bedrock is a non-HTTP transport: requests go through the AWS SDK's
//! Converse API rather than a rig-owned HTTP runtime. The data-oriented
//! face is therefore a serde [`Config`] describing how to *build* an
//! `aws_sdk_bedrockruntime::Client` (never holding one), an async
//! [`client_from_config`] mirroring the existing [`crate::client::ClientBuilder`]
//! and `from_env` paths, and free [`complete`]/[`open_stream`] functions the
//! existing [`crate::completion::CompletionModel`] trait impl is rewired
//! through.

use aws_config::{BehaviorVersion, Region};
use rig_core::completion::{self, CompletionError, CompletionRequest};
use rig_core::providers::descriptor::ProviderDescriptor;
use rig_core::streaming::StreamingCompletionResponse;
use rig_core::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use serde::{Deserialize, Serialize};
use tracing::Instrument;

use crate::completion::resolve_request_model;
use crate::types::{
    assistant_content::AwsConverseOutput, completion_request::AwsCompletionRequest,
    converse_output::InternalConverseOutput, errors::AwsSdkConverseError,
};

/// Bedrock's capability sheet.
///
/// `supports_response_format` and `composes_native_output_with_tools` are
/// true because the Converse request builder sets `output_config` and
/// `tool_config` on the same request without gating.
/// `stream_include_usage` and `emits_complete_single_chunk_tool_calls` are
/// OpenAI-wire concepts that do not apply to the AWS SDK transport.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor::named("aws_bedrock")
    .with_tools(true)
    .with_response_format(true)
    .with_composes_native_output_with_tools(true)
    .with_max_embedding_documents(1024);

/// Plain-data description of how to build a Bedrock runtime client.
///
/// This describes client *construction* (region / profile / endpoint) as
/// data; it never holds an AWS client or credentials. Credential material is
/// resolved by the AWS SDK's default credential chain at
/// [`client_from_config`] time, exactly as the existing
/// [`crate::client::ClientBuilder`] and `Client::from_env` paths do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// AWS region (`None` defers to the SDK's default region resolution,
    /// matching `Client::from_env`).
    pub region: Option<String>,
    /// Named AWS profile to load credentials from (mirrors
    /// [`crate::client::Client::with_profile_name`]).
    pub profile: Option<String>,
    /// Custom endpoint URL override (local stacks, VPC endpoints).
    pub endpoint_url: Option<String>,
    /// Model identifier requests are built for.
    pub model: String,
}

impl Config {
    /// Config for `model` using the SDK's default credential chain and
    /// region resolution.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            region: None,
            profile: None,
            endpoint_url: None,
            model: model.into(),
        }
    }

    /// Pin the AWS region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Load credentials from a named AWS profile.
    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    /// Override the endpoint URL.
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }
}

/// Build an `aws_sdk_bedrockruntime::Client` from `cfg`.
///
/// Loads the AWS credential chain asynchronously, mirroring the existing
/// paths: no fields set behaves like `aws_config::load_from_env()`
/// (`Client::from_env`), `profile` mirrors `Client::with_profile_name`, and
/// `region` mirrors `ClientBuilder::region`.
pub async fn client_from_config(cfg: &Config) -> aws_sdk_bedrockruntime::Client {
    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(profile) = &cfg.profile {
        loader = loader.profile_name(profile.as_str());
    }
    if let Some(region) = &cfg.region {
        loader = loader.region(Region::new(region.clone()));
    }
    if let Some(endpoint_url) = &cfg.endpoint_url {
        loader = loader.endpoint_url(endpoint_url.as_str());
    }
    let sdk_config = loader.load().await;
    aws_sdk_bedrockruntime::Client::new(&sdk_config)
}

/// Send `request` through the Converse API and return the normalized
/// response. `model` is the default model, overridable per-request via
/// `CompletionRequest::model`.
///
/// Extracted from the `CompletionModel::completion` trait impl, which is
/// rewired through [`complete_with_options`]; behavior is unchanged.
pub async fn complete(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    complete_with_options(client, model, false, request).await
}

/// [`complete`] with Bedrock prompt caching controllable (the trait impl's
/// `with_prompt_caching` knob).
pub async fn complete_with_options(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    prompt_caching: bool,
    completion_request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    let request_model = resolve_request_model(model, &completion_request);

    let span = CompletionSpanBuilder::new("aws_bedrock", &request_model, CompletionOperation::Chat)
        .system_instructions(
            completion_request.preamble.as_deref(),
            completion_request.record_telemetry_content,
        )
        .build();

    let request = AwsCompletionRequest {
        inner: completion_request,
        prompt_caching,
    };

    let mut converse_builder = client.converse().model_id(request_model.clone());

    let tool_config = request.tools_config()?;
    let messages = request.messages()?;
    let output_config = request.output_config()?;
    converse_builder = converse_builder
        .set_additional_model_request_fields(request.additional_params())
        .set_inference_config(request.inference_config())
        .set_tool_config(tool_config)
        .set_system(request.system_prompt()?)
        .set_messages(Some(messages))
        .set_output_config(output_config);

    async move {
        let response = converse_builder
            .send()
            .await
            .map_err(|sdk_error| Into::<CompletionError>::into(AwsSdkConverseError(sdk_error)))?;

        let response: InternalConverseOutput = response
            .try_into()
            .map_err(|x| CompletionError::ProviderError(format!("Type conversion error: {x}")))?;

        let aws_output = AwsConverseOutput(response);

        let span = tracing::Span::current();
        span.record_response_metadata(&aws_output);
        let response: completion::CompletionResponse = aws_output.try_into()?;
        span.record_token_usage(&response.usage);

        Ok(response)
    }
    .instrument(span)
    .await
}

/// Open a streaming Converse completion for `request`.
///
/// Extracted from the `CompletionModel::stream` trait impl, which is rewired
/// through [`open_stream_with_options`]; behavior is unchanged.
pub async fn open_stream(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    request: CompletionRequest,
) -> Result<StreamingCompletionResponse, CompletionError> {
    open_stream_with_options(client, model, false, request).await
}

/// [`open_stream`] with Bedrock prompt caching controllable.
pub async fn open_stream_with_options(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    prompt_caching: bool,
    request: CompletionRequest,
) -> Result<StreamingCompletionResponse, CompletionError> {
    crate::streaming::stream_converse(client, model, prompt_caching, request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_defer_to_sdk_resolution() {
        let cfg = Config::new("amazon.nova-lite-v1:0");
        assert_eq!(cfg.region, None);
        assert_eq!(cfg.profile, None);
        assert_eq!(cfg.endpoint_url, None);
        assert_eq!(cfg.model, "amazon.nova-lite-v1:0");
    }

    #[test]
    fn config_serde_round_trip() {
        let cfg = Config::new("amazon.nova-lite-v1:0")
            .with_region("eu-west-1")
            .with_profile("dev")
            .with_endpoint_url("http://localhost:4566");
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cfg);
    }

    #[test]
    // Pinning const capability values is the point of this test.
    #[allow(clippy::assertions_on_constants)]
    fn descriptor_is_honest() {
        assert_eq!(DESCRIPTOR.name, "aws_bedrock");
        assert!(DESCRIPTOR.supports_tools);
        assert!(DESCRIPTOR.supports_response_format);
        // OpenAI-wire streaming concepts do not apply to the AWS SDK transport.
        assert!(!DESCRIPTOR.stream_include_usage);
        assert!(!DESCRIPTOR.emits_complete_single_chunk_tool_calls);
        assert_eq!(DESCRIPTOR.max_embedding_documents, Some(1024));
    }
}
