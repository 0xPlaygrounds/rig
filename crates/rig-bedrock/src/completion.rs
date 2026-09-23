//! Bedrock Converse completion models and model identifiers.
//! Model availability and inference-profile support depend on the AWS region.
//!
//! ```no_run
//! use rig_bedrock::{client::Client, completion::{CompletionModel, AMAZON_NOVA_LITE}};
//!
//! let model = CompletionModel::new(Client::from_env()?, AMAZON_NOVA_LITE);
//! # Ok::<(), rig_core::client::ProviderClientError>(())
//! ```

use crate::{
    client::Client,
    types::{
        assistant_content::{AwsConverseOutput, completion_response},
        completion_request::AwsCompletionRequest,
        converse_output::InternalConverseOutput,
        errors::AwsSdkConverseError,
    },
};

use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::completion::{self, CompletionRequest};
use rig_core::error::ProviderError;
use rig_core::streaming::StreamingCompletionResponse;
use rig_core::telemetry::ProviderResponseExt;
use rig_core::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use tracing::Instrument;

// Profile identifiers with a us. prefix route inference within the US region
// family; callers elsewhere must select a supported regional profile.

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
    /// Marks system content and, when history contains no reasoning, the final
    /// message. Disabled by default.
    pub prompt_caching: bool,
    /// Guardrail applied to unary Converse requests from this model, if any.
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

    /// Enables checkpoints after system content and the final message.
    /// History containing reasoning suppresses the message checkpoint. Tool
    /// definitions are not marked; model-specific caching limits apply.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Apply a [Bedrock guardrail](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
    /// to unary Converse requests. Streaming requests do not apply this setting.
    ///
    /// `identifier` is the guardrail ID or ARN; `version` is a version or `DRAFT`.
    /// Enabled trace details are available through [`Self::raw_completion`] in
    /// [`InternalConverseOutput::trace`]. The normalized finish reason reports
    /// guardrail intervention as content filtering.
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
    /// Executes one Converse request and returns provider-native output.
    /// Returns request-conversion, SDK, or response-conversion errors.
    pub async fn raw_completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<AwsConverseOutput, ProviderError> {
        let request_model = resolve_request_model(&self.model, &completion_request);

        let span = SpanBuilder::new("aws_bedrock", &request_model, GenAiOperation::Chat)
            .system_instructions(
                completion_request.system_instructions(),
                completion_request.record_telemetry_content,
            )
            .build();

        let request = AwsCompletionRequest::for_model(
            completion_request,
            &request_model,
            self.prompt_caching,
        );

        let mut converse_builder = self
            .client
            .inner()
            .await
            .converse()
            .model_id(request_model.clone());

        let tool_config = request.tools_config()?;
        let output_config = request.output_config()?;
        let additional_params = request.additional_params();
        let inference_config = request.inference_config();
        let system_prompt = request.system_prompt()?;
        let messages = request.messages()?;
        converse_builder = converse_builder
            .set_additional_model_request_fields(additional_params)
            .set_inference_config(Some(inference_config))
            .set_tool_config(tool_config)
            .set_system(system_prompt)
            .set_messages(Some(messages))
            .set_output_config(output_config)
            .set_guardrail_config(self.guardrail.clone());

        async move {
            let response = converse_builder
                .send()
                .await
                .map_err(|sdk_error| Into::<ProviderError>::into(AwsSdkConverseError(sdk_error)))?;

            let response: InternalConverseOutput = response
                .try_into()
                .map_err(|x| ProviderError::Provider(format!("Type conversion error: {x}")))?;

            let aws_output = AwsConverseOutput(response);

            let span = tracing::Span::current();
            span.record_response(
                aws_output.response_id(),
                aws_output.response_model_name(),
                &aws_output.usage().unwrap_or_default(),
            );

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
    ) -> Result<completion::CompletionResponse, ProviderError> {
        let model = resolve_request_model(&self.model, &completion_request);
        completion_response(self.raw_completion(completion_request).await?, &model)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, ProviderError> {
        CompletionModel::stream(self, request).await
    }
}
