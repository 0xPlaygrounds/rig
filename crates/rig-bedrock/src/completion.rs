//! All supported models <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>

use crate::{
    client::Client,
    types::{
        assistant_content::AwsConverseOutput, completion_request::AwsCompletionRequest,
        converse_output::InternalConverseOutput, errors::AwsSdkConverseError,
    },
};

use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::completion::{self, CompletionError, CompletionRequest};
use rig_core::streaming::StreamingCompletionResponse;
use rig_core::telemetry::ProviderResponseExt;
use rig_core::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use tracing::Instrument;

// Model identifiers below were verified against `bedrock:ListFoundationModels`
// in us-east-1, us-west-2, eu-central-1 and ap-northeast-1, and each was
// invoked with `Converse` in us-east-1. Bedrock answers a retired identifier
// with `ResourceNotFoundException` ("This model version has reached the end of
// its life") and a profile-only identifier invoked bare with
// `ValidationException` ("Invocation of model ID … with on-demand throughput
// isn't supported"), so an identifier that fails either check is not shipped.
//
// A `us.` prefix marks a *cross-region inference profile*, which is the only
// invocable form for the models that carry one. The prefix names a region
// family: callers in Europe or Asia-Pacific substitute `eu.` or `apac.` (a
// `global.` profile also exists for some Anthropic models). See
// <https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html>.

/// `amazon.nova-lite-v1:0`
pub const AMAZON_NOVA_LITE: &str = "amazon.nova-lite-v1:0";
/// `amazon.nova-micro-v1:0`
pub const AMAZON_NOVA_MICRO: &str = "amazon.nova-micro-v1:0";
/// `amazon.nova-pro-v1:0`
pub const AMAZON_NOVA_PRO: &str = "amazon.nova-pro-v1:0";
/// `amazon.nova-canvas-v1:0` image generation model
pub const AMAZON_NOVA_CANVAS: &str = "amazon.nova-canvas-v1:0";
/// `amazon.nova-reel-v1:0` video generation model
pub const AMAZON_NOVA_REEL_V1_0: &str = "amazon.nova-reel-v1:0";
/// `amazon.nova-reel-v1:1` video generation model
pub const AMAZON_NOVA_REEL_V1_1: &str = "amazon.nova-reel-v1:1";
/// `amazon.nova-sonic-v1:0` speech model
pub const AMAZON_NOVA_SONIC: &str = "amazon.nova-sonic-v1:0";
/// `amazon.rerank-v1:0` rerank model
pub const AMAZON_RERANK_1_0: &str = "amazon.rerank-v1:0";
/// `amazon.titan-embed-text-v1` embedding model
pub const AMAZON_TITAN_EMBEDDINGS_G1_TEXT: &str = "amazon.titan-embed-text-v1";
/// `amazon.titan-embed-image-v1` multimodal embedding model
pub const AMAZON_TITAN_MULTIMODAL_EMBEDDINGS_G1: &str = "amazon.titan-embed-image-v1";
/// `amazon.titan-embed-text-v2:0` embedding model
pub const AMAZON_TITAN_TEXT_EMBEDDINGS_V2: &str = "amazon.titan-embed-text-v2:0";

/// `us.anthropic.claude-haiku-4-5-20251001-v1:0` (cross-region profile)
pub const ANTHROPIC_CLAUDE_HAIKU_4_5: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
/// `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (cross-region profile)
pub const ANTHROPIC_CLAUDE_SONNET_4_5: &str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0";
/// `us.anthropic.claude-opus-4-5-20251101-v1:0` (cross-region profile)
pub const ANTHROPIC_CLAUDE_OPUS_4_5: &str = "us.anthropic.claude-opus-4-5-20251101-v1:0";
/// `us.anthropic.claude-sonnet-4-6` (cross-region profile)
pub const ANTHROPIC_CLAUDE_SONNET_4_6: &str = "us.anthropic.claude-sonnet-4-6";
/// `us.anthropic.claude-sonnet-5` (cross-region profile)
pub const ANTHROPIC_CLAUDE_SONNET_5: &str = "us.anthropic.claude-sonnet-5";
/// `us.anthropic.claude-opus-5` (cross-region profile)
pub const ANTHROPIC_CLAUDE_OPUS_5: &str = "us.anthropic.claude-opus-5";

/// `cohere.embed-english-v3` embedding model
pub const COHERE_EMBED_ENGLISH: &str = "cohere.embed-english-v3";
/// `cohere.embed-multilingual-v3` embedding model
pub const COHERE_EMBED_MULTILINGUAL: &str = "cohere.embed-multilingual-v3";
/// `cohere.rerank-v3-5:0` rerank model
pub const COHERE_RERANK_V3_5: &str = "cohere.rerank-v3-5:0";

/// `us.deepseek.r1-v1:0` (cross-region profile)
pub const DEEPSEEK_R1: &str = "us.deepseek.r1-v1:0";

/// `luma.ray-v2:0` video generation model
pub const LUMA_RAY_V2_0: &str = "luma.ray-v2:0";

/// `meta.llama3-8b-instruct-v1:0`
pub const LLAMA_3_8B_INSTRUCT: &str = "meta.llama3-8b-instruct-v1:0";
/// `meta.llama3-70b-instruct-v1:0`
pub const LLAMA_3_70B_INSTRUCT: &str = "meta.llama3-70b-instruct-v1:0";
/// `meta.llama3-1-8b-instruct-v1:0`
pub const LLAMA_3_1_8B_INSTRUCT: &str = "meta.llama3-1-8b-instruct-v1:0";
/// `meta.llama3-1-70b-instruct-v1:0`
pub const LLAMA_3_1_70B_INSTRUCT: &str = "meta.llama3-1-70b-instruct-v1:0";
/// `us.meta.llama3-3-70b-instruct-v1:0` (cross-region profile)
pub const META_LLAMA_3_3_70B_INSTRUCT: &str = "us.meta.llama3-3-70b-instruct-v1:0";
/// `us.meta.llama4-maverick-17b-instruct-v1:0` (cross-region profile)
pub const META_LLAMA_4_MAVERICK_17B_INSTRUCT: &str = "us.meta.llama4-maverick-17b-instruct-v1:0";
/// `us.meta.llama4-scout-17b-instruct-v1:0` (cross-region profile)
pub const META_LLAMA_4_SCOUT_17B_INSTRUCT: &str = "us.meta.llama4-scout-17b-instruct-v1:0";

/// `mistral.mistral-7b-instruct-v0:2`
pub const MISTRAL_7B_INSTRUCT: &str = "mistral.mistral-7b-instruct-v0:2";
/// `mistral.mistral-large-2402-v1:0`
pub const MISTRAL_LARGE_24_02: &str = "mistral.mistral-large-2402-v1:0";
/// `mistral.mistral-small-2402-v1:0`
pub const MISTRAL_SMALL_24_02: &str = "mistral.mistral-small-2402-v1:0";
/// `mistral.mixtral-8x7b-instruct-v0:1`
pub const MISTRAL_MIXTRAL_8X7B_INSTRUCT_V0: &str = "mistral.mixtral-8x7b-instruct-v0:1";
/// `us.mistral.pixtral-large-2502-v1:0` (cross-region profile)
pub const MISTRAL_PIXTRAL_LARGE_2502: &str = "us.mistral.pixtral-large-2502-v1:0";

/// `stability.sd3-5-large-v1:0` image generation model
pub const STABILITY_SD3_5_LARGE: &str = "stability.sd3-5-large-v1:0";
/// `stability.stable-image-core-v1:1` image generation model
pub const STABILITY_STABLE_IMAGE_CORE_1_0: &str = "stability.stable-image-core-v1:1";
/// `stability.stable-image-ultra-v1:1` image generation model
pub const STABILITY_STABLE_IMAGE_ULTRA_1_0: &str = "stability.stable-image-ultra-v1:1";

/// `twelvelabs.pegasus-1-2-v1:0` video-understanding model
pub const TWELVELABS_PEGASUS_V1_2: &str = "twelvelabs.pegasus-1-2-v1:0";

/// `us.writer.palmyra-x4-v1:0` (cross-region profile)
pub const WRITER_PALMYRA_X4: &str = "us.writer.palmyra-x4-v1:0";
/// `us.writer.palmyra-x5-v1:0` (cross-region profile)
pub const WRITER_PALMYRA_X5: &str = "us.writer.palmyra-x5-v1:0";

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    pub model: String,
    /// When enabled, cache checkpoints are inserted into Converse API requests
    /// to take advantage of [Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html).
    /// A checkpoint is placed after the system prompt and after the last message
    /// in the chat history. Disabled by default.
    pub prompt_caching: bool,
    /// Guardrail applied to every Converse request from this model, if any.
    /// Set through [`CompletionModel::with_guardrail`].
    pub guardrail: Option<aws_bedrock::GuardrailConfiguration>,
}

impl CompletionModel {
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            prompt_caching: false,
            guardrail: None,
        }
    }

    /// Enable Bedrock prompt caching for this model.
    ///
    /// When enabled, `CachePoint` blocks are inserted after the serialized
    /// `system` content and after the final `messages` entry in each Converse
    /// API request. This allows Bedrock to cache and reuse repeated prompt
    /// prefixes, reducing both latency and input token costs.
    ///
    /// This currently covers the `system` and `messages` request fields only.
    /// Some Bedrock models also support caching `tools`, but that is not wired
    /// up here yet.
    ///
    /// Cacheability and token thresholds are model-specific. If the cached
    /// prefix is too short or the model does not support caching for a given
    /// field, Bedrock ignores the checkpoint. See the Bedrock prompt caching
    /// support table for current limits and field support.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Apply a [Bedrock guardrail](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
    /// to every Converse request this model issues.
    ///
    /// `identifier` is the guardrail id or ARN and `version` its version (or
    /// `DRAFT`). When `trace` is enabled, Bedrock returns its assessment on
    /// [`InternalConverseOutput::trace`](crate::types::converse_output::InternalConverseOutput::trace) —
    /// the only place it explains *why* a turn came back with
    /// [`StopReason::GuardrailIntervened`](crate::types::converse_output::StopReason::GuardrailIntervened),
    /// which the normalized response reports as a content filter and nothing
    /// more. Reach it through
    /// [`raw_completion`](CompletionModel::raw_completion).
    pub fn with_guardrail(
        mut self,
        identifier: impl Into<String>,
        version: impl Into<String>,
        trace: aws_bedrock::GuardrailTrace,
    ) -> Self {
        self.guardrail = Some(
            aws_bedrock::GuardrailConfiguration::builder()
                .guardrail_identifier(identifier)
                .guardrail_version(version)
                .trace(trace)
                .build(),
        );
        self
    }
}

pub(crate) fn resolve_request_model(
    default_model: &str,
    completion_request: &CompletionRequest,
) -> String {
    completion_request
        .model
        .clone()
        .unwrap_or_else(|| default_model.to_string())
}

impl CompletionModel {
    /// Execute a completion and return Bedrock's own Converse output.
    ///
    /// This is the escape hatch for fields rig does not normalize;
    /// [`completion::CompletionModel::completion`] calls it and maps the
    /// result, so there is exactly one request either way.
    pub async fn raw_completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<AwsConverseOutput, CompletionError> {
        let request_model = resolve_request_model(&self.model, &completion_request);

        let span =
            CompletionSpanBuilder::new("aws_bedrock", &request_model, CompletionOperation::Chat)
                .system_instructions(
                    completion_request.preamble.as_deref(),
                    completion_request.record_telemetry_content,
                )
                .build();

        let request = AwsCompletionRequest {
            inner: completion_request,
            prompt_caching: self.prompt_caching,
        };

        let mut converse_builder = self
            .client
            .get_inner()
            .await
            .converse()
            .model_id(request_model.clone());

        let tool_config = request.tools_config()?;
        let messages = request.messages()?;
        let output_config = request.output_config()?;
        converse_builder = converse_builder
            .set_additional_model_request_fields(request.additional_params())
            .set_inference_config(request.inference_config())
            .set_tool_config(tool_config)
            .set_system(request.system_prompt()?)
            .set_messages(Some(messages))
            .set_output_config(output_config)
            .set_guardrail_config(self.guardrail.clone());

        async move {
            let response = converse_builder.send().await.map_err(|sdk_error| {
                Into::<CompletionError>::into(AwsSdkConverseError(sdk_error))
            })?;

            let response: InternalConverseOutput = response.try_into().map_err(|x| {
                CompletionError::ProviderError(format!("Type conversion error: {x}"))
            })?;

            let aws_output = AwsConverseOutput(response);

            let span = tracing::Span::current();
            span.record_response_metadata(&aws_output);
            span.record_token_usage(&aws_output.get_usage().unwrap_or_default());

            Ok(aws_output)
        }
        .instrument(span)
        .await
    }
}

impl completion::CompletionModel for CompletionModel {
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        self.raw_completion(completion_request).await?.try_into()
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        CompletionModel::stream(self, request).await
    }
}
