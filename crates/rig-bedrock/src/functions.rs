//! AWS Bedrock as config + free functions (data-oriented face).
//!
//! Bedrock is a non-HTTP transport: requests go through the AWS SDK's
//! Converse API rather than a rig-owned HTTP runtime. The data-oriented
//! face is therefore a serde [`Config`] describing how to *build* an
//! `aws_sdk_bedrockruntime::Client` (never holding one), an async
//! [`client_from_config`] that resolves the AWS credential chain, and free
//! [`complete`] / [`open_stream`] / [`embed`] / [`generate_image`] functions
//! taking that client explicitly.
//!
//! This module is the crate's only face: there is no Bedrock client type and
//! no model traits. Sibling modules hold the model-id constants
//! ([`crate::completion`], [`crate::embedding`], [`crate::image`]) and the
//! wire-type conversions these functions call.

use aws_config::{BehaviorVersion, Region};
use aws_smithy_types::Blob;
use rig_core::completion::{self, CompletionError, CompletionRequest};
use rig_core::providers::descriptor::ProviderDescriptor;
use rig_core::streaming::StreamingCompletionResponse;
use rig_core::telemetry::{CompletionOperation, SpanCombinator, completion_span};
use serde::{Deserialize, Serialize};
use tracing::Instrument;

use crate::completion::resolve_request_model;
use crate::types::{
    assistant_content::AwsConverseOutput,
    completion_request::AwsCompletionRequest,
    converse_output::InternalConverseOutput,
    errors::{AwsSdkConverseError, AwsSdkInvokeModelError},
    text_to_image::{TextToImageGeneration, TextToImageResponse},
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
/// [`client_from_config`] time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// AWS region (`None` defers to the SDK's default region resolution).
    pub region: Option<String>,
    /// Named AWS profile to load credentials from.
    pub profile: Option<String>,
    /// Custom endpoint URL override (local stacks, VPC endpoints).
    pub endpoint_url: Option<String>,
    /// Model identifier requests are built for.
    pub model: String,
    /// Enable Bedrock prompt caching, inserting cache points into requests.
    ///
    /// `CachePoint` blocks are placed after the serialized `system` content
    /// and after the final `messages` entry of each Converse request, so
    /// Bedrock can reuse repeated prompt prefixes. Cacheability and token
    /// thresholds are model-specific; too-short prefixes are ignored by
    /// Bedrock. Tool definitions are not cached yet.
    #[serde(default)]
    pub prompt_caching: bool,
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
            prompt_caching: false,
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

    /// Enable Bedrock prompt caching for requests built from this config.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }
}

/// Build an `aws_sdk_bedrockruntime::Client` from `cfg`.
///
/// Loads the AWS credential chain asynchronously: with no fields set this
/// behaves like `aws_config::load_from_env()`, while `profile`, `region`, and
/// `endpoint_url` override the corresponding SDK resolution steps.
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
pub async fn complete(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    complete_with_options(client, model, false, request).await
}

/// [`complete`] with Bedrock prompt caching controllable
/// ([`Config::prompt_caching`]).
pub async fn complete_with_options(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    prompt_caching: bool,
    completion_request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    let request_model = resolve_request_model(model, &completion_request);

    let span = completion_span(
        "aws_bedrock",
        &request_model,
        CompletionOperation::Chat,
        &completion_request,
    );

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

// ================================================================
// Embeddings
// ================================================================

/// Plain-data Bedrock embeddings configuration.
///
/// A sibling of [`Config`]: the same client-construction fields plus the
/// embedding model identifier and its optional dimension count.
/// [`Self::client_config`] projects the connection half so the host runtime
/// can share its cached AWS client machinery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingConfig {
    /// AWS region (`None` defers to the SDK's default region resolution).
    pub region: Option<String>,
    /// Named AWS profile to load credentials from.
    pub profile: Option<String>,
    /// Custom endpoint URL override (local stacks, VPC endpoints).
    pub endpoint_url: Option<String>,
    /// Embedding model identifier requests are built for.
    pub model: String,
    /// Requested embedding dimensions, forwarded as the Titan `dimensions`
    /// field (`None` sends the classic path's zero-valued default).
    pub ndims: Option<usize>,
}

impl EmbeddingConfig {
    /// Config for `model` using the SDK's default credential chain and
    /// region resolution.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            region: None,
            profile: None,
            endpoint_url: None,
            model: model.into(),
            ndims: None,
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

    /// Request `ndims`-sized embeddings.
    pub fn with_ndims(mut self, ndims: usize) -> Self {
        self.ndims = Some(ndims);
        self
    }

    /// The client-construction half of this config as a [`Config`], for
    /// sharing a host runtime's cached [`client_from_config`] machinery.
    pub fn client_config(&self) -> Config {
        Config {
            region: self.region.clone(),
            profile: self.profile.clone(),
            endpoint_url: self.endpoint_url.clone(),
            model: self.model.clone(),
            prompt_caching: false,
        }
    }
}

/// Embed `texts` with `model`, one `InvokeModel` call per document (the
/// Bedrock embedding API is single-document), summing reported input token
/// counts into usage.
///
/// Every document is attempted; the first failure is reported.
pub async fn embed(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    ndims: Option<usize>,
    texts: Vec<String>,
) -> Result<rig_core::embeddings::EmbeddingResponse, rig_core::embeddings::EmbeddingError> {
    use rig_core::embeddings::{Embedding, EmbeddingError};

    let mut results = Vec::with_capacity(texts.len());
    let mut errors: Vec<EmbeddingError> = Vec::new();
    let mut usage = rig_core::completion::Usage::new();

    for doc in texts {
        let request = crate::embedding::EmbeddingRequest {
            input_text: doc.clone(),
            dimensions: ndims.unwrap_or_default(),
            normalize: true,
        };
        match crate::embedding::invoke_embedding(client, model, request).await {
            Ok(response) => {
                usage.input_tokens += response.input_text_token_count as u64;
                usage.total_tokens += response.input_text_token_count as u64;
                results.push(Embedding {
                    document: doc,
                    vec: response.embedding,
                });
            }
            Err(err) => errors.push(err),
        }
    }

    match errors.as_slice() {
        [] => Ok(rig_core::embeddings::EmbeddingResponse {
            embeddings: results,
            usage,
        }),
        [err, ..] => Err(EmbeddingError::ResponseError(err.to_string())),
    }
}

/// Embed caller-defined batches, returning one order-aligned
/// [`rig_core::OneOrMany`] group per input batch plus summed usage.
pub async fn embed_batches(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    ndims: Option<usize>,
    texts: Vec<Vec<String>>,
) -> Result<
    (
        Vec<rig_core::OneOrMany<rig_core::embeddings::Embedding>>,
        rig_core::completion::Usage,
    ),
    rig_core::embeddings::EmbeddingError,
> {
    let counts: Vec<usize> = texts.iter().map(Vec::len).collect();
    let flat: Vec<String> = texts.into_iter().flatten().collect();
    let response = embed(client, model, ndims, flat).await?;
    let groups = rig_core::embeddings::batching::group_batches(&counts, response.embeddings)?;
    Ok((groups, response.usage))
}

// ================================================================
// Image generation
// ================================================================

/// Plain-data Bedrock image-generation configuration.
///
/// The image sibling of [`Config`] / [`EmbeddingConfig`]: the same
/// client-construction fields plus the image model identifier.
/// [`Self::client_config`] projects the connection half so a host runtime can
/// share its cached AWS client machinery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ImageConfig {
    /// AWS region (`None` defers to the SDK's default region resolution).
    pub region: Option<String>,
    /// Named AWS profile to load credentials from.
    pub profile: Option<String>,
    /// Custom endpoint URL override (local stacks, VPC endpoints).
    pub endpoint_url: Option<String>,
    /// Image-generation model identifier (see [`crate::image`]).
    pub model: String,
}

impl ImageConfig {
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

    /// The client-construction half of this config as a [`Config`], for
    /// sharing a host runtime's cached [`client_from_config`] machinery.
    pub fn client_config(&self) -> Config {
        Config {
            region: self.region.clone(),
            profile: self.profile.clone(),
            endpoint_url: self.endpoint_url.clone(),
            model: self.model.clone(),
            prompt_caching: false,
        }
    }
}

/// Generate an image with `model` through Bedrock's `InvokeModel` text-to-image
/// API, returning the decoded bytes alongside the raw Bedrock payload.
pub async fn generate_image(
    client: &aws_sdk_bedrockruntime::Client,
    model: &str,
    request: rig_core::image_generation::ImageGenerationRequest,
) -> Result<
    rig_core::image_generation::ImageGenerationResponse<TextToImageResponse>,
    rig_core::image_generation::ImageGenerationError,
> {
    use rig_core::image_generation::ImageGenerationError;

    let mut body = TextToImageGeneration::new(request.prompt);
    body.width(request.width);
    body.height(request.height);

    let body = serde_json::to_string(&body)?;
    let model_response = client
        .invoke_model()
        .model_id(model)
        .content_type("application/json")
        .accept("application/json")
        .body(Blob::new(body))
        .send()
        .await
        .map_err(|sdk_error| {
            Into::<ImageGenerationError>::into(AwsSdkInvokeModelError(sdk_error))
        })?;

    let response_str = String::from_utf8(model_response.body.into_inner())
        .map_err(|e| ImageGenerationError::ResponseError(e.to_string()))?;

    let result: TextToImageResponse = serde_json::from_str(&response_str)
        .map_err(|e| ImageGenerationError::ResponseError(e.to_string()))?;

    result.try_into()
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
