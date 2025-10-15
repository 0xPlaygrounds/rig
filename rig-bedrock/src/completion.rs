//! All supported models <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>

use crate::{
    client::Client,
    types::{
        assistant_content::AwsConverseOutput, completion_request::AwsCompletionRequest,
        converse_output::InternalConverseOutput, errors::AwsSdkConverseError,
    },
};

use rig::completion::{self, CompletionError, CompletionRequest};
use rig::streaming::StreamingCompletionResponse;

/// `ai21.jamba-1-5-large-v1:0`
pub const AI21_JAMBA_1_5_LARGE: &str = "ai21.jamba-1-5-large-v1:0";
/// `ai21.jamba-1-5-mini-v1:0`
pub const AI21_JAMBA_1_5_MINI: &str = "ai21.jamba-1-5-mini-v1:0";
/// `amazon.nova-canvas-v1:0`
pub const AMAZON_NOVA_CANVAS: &str = "amazon.nova-canvas-v1:0";
/// `amazon.nova-lite-v1:0`
pub const AMAZON_NOVA_LITE: &str = "amazon.nova-lite-v1:0";
/// `amazon.nova-micro-v1:0`
pub const AMAZON_NOVA_MICRO: &str = "amazon.nova-micro-v1:0";
/// `amazon.nova-premier-v1:0`
pub const AMAZON_NOVA_PREMIER: &str = "amazon.nova-premier-v1:0";
/// `amazon.nova-pro-v1:0`
pub const AMAZON_NOVA_PRO: &str = "amazon.nova-pro-v1:0";
/// `amazon.nova-reel-v1:0`
pub const AMAZON_NOVA_REEL_V1_0: &str = "amazon.nova-reel-v1:0";
/// `amazon.nova-reel-v1:1`
pub const AMAZON_NOVA_REEL_V1_1: &str = "amazon.nova-reel-v1:1";
/// `amazon.nova-sonic-v1:0`
pub const AMAZON_NOVA_SONIC: &str = "amazon.nova-sonic-v1:0";
/// `amazon.rerank-v1:0`
pub const AMAZON_RERANK_1_0: &str = "amazon.rerank-v1:0";
/// `amazon.titan-embed-text-v1`
pub const AMAZON_TITAN_EMBEDDINGS_G1_TEXT: &str = "amazon.titan-embed-text-v1";
/// `amazon.titan-image-generator-v2:0`
pub const AMAZON_TITAN_IMAGE_GENERATOR_G1_V2: &str = "amazon.titan-image-generator-v2:0";
/// `amazon.titan-image-generator-v1`
pub const AMAZON_TITAN_IMAGE_GENERATOR_G1: &str = "amazon.titan-image-generator-v1";
/// `amazon.titan-embed-image-v1`
pub const AMAZON_TITAN_MULTIMODAL_EMBEDDINGS_G1: &str = "amazon.titan-embed-image-v1";
/// `amazon.titan-embed-text-v2:0`
pub const AMAZON_TITAN_TEXT_EMBEDDINGS_V2: &str = "amazon.titan-embed-text-v2:0";
/// `amazon.titan-text-express-v1`
pub const AMAZON_TITAN_TEXT_EXPRESS_V1: &str = "amazon.titan-text-express-v1";
/// `amazon.titan-text-lite-v1`
pub const AMAZON_TITAN_TEXT_LITE_V1: &str = "amazon.titan-text-lite-v1";
/// `amazon.titan-text-premier-v1:0`
pub const AMAZON_TITAN_TEXT_PREMIER_V1_0: &str = "amazon.titan-text-premier-v1:0";
/// `anthropic.claude-3-haiku-20240307-v1:0`
pub const ANTHROPIC_CLAUDE_3_HAIKU: &str = "anthropic.claude-3-haiku-20240307-v1:0";
/// `anthropic.claude-3-opus-20240229-v1:0`
pub const ANTHROPIC_CLAUDE_3_OPUS: &str = "anthropic.claude-3-opus-20240229-v1:0";
/// `anthropic.claude-3-sonnet-20240229-v1:0`
pub const ANTHROPIC_CLAUDE_3_SONNET: &str = "anthropic.claude-3-sonnet-20240229-v1:0";
/// `anthropic.claude-3-5-haiku-20241022-v1:0`
pub const ANTHROPIC_CLAUDE_3_5_HAIKU: &str = "anthropic.claude-3-5-haiku-20241022-v1:0";
/// `anthropic.claude-3-5-sonnet-20241022-v2:0`
pub const ANTHROPIC_CLAUDE_3_5_SONNET_V2: &str = "anthropic.claude-3-5-sonnet-20241022-v2:0";
/// `anthropic.claude-3-5-sonnet-20240620-v1:0`
pub const ANTHROPIC_CLAUDE_3_5_SONNET: &str = "anthropic.claude-3-5-sonnet-20240620-v1:0";
/// `anthropic.claude-3-7-sonnet-20250219-v1:0`
pub const ANTHROPIC_CLAUDE_3_7_SONNET: &str = "anthropic.claude-3-7-sonnet-20250219-v1:0";
/// `anthropic.claude-opus-4-20250514-v1:0`
pub const ANTHROPIC_CLAUDE_OPUS_4: &str = "anthropic.claude-opus-4-20250514-v1:0";
/// `anthropic.claude-sonnet-4-20250514-v1:0`
pub const ANTHROPIC_CLAUDE_SONNET_4: &str = "anthropic.claude-sonnet-4-20250514-v1:0";
/// `cohere.command-light-text-v14`
pub const COHERE_COMMAND_LIGHT_TEXT: &str = "cohere.command-light-text-v14";
/// `cohere.command-r-plus-v1:0`
pub const COHERE_COMMAND_R_PLUS: &str = "cohere.command-r-plus-v1:0";
/// `cohere.command-r-v1:0`
pub const COHERE_COMMAND_R: &str = "cohere.command-r-v1:0";
/// `cohere.command-text-v14`
pub const COHERE_COMMAND: &str = "cohere.command-text-v14";
/// `cohere.embed-english-v3`
pub const COHERE_EMBED_ENGLISH: &str = "cohere.embed-english-v3";
/// `cohere.embed-multilingual-v3`
pub const COHERE_EMBED_MULTILINGUAL: &str = "cohere.embed-multilingual-v3";
/// `cohere.rerank-v3-5:0`
pub const COHERE_RERANK_V3_5: &str = "cohere.rerank-v3-5:0";
/// `deepseek.r1-v1:0`
pub const DEEPSEEK_R1: &str = "deepseek.r1-v1:0";
/// `luma.ray-v2:0`
pub const LUMA_RAY_V2_0: &str = "luma.ray-v2:0";
/// `meta.llama3-8b-instruct-v1:0`
pub const LLAMA_3_8B_INSTRUCT: &str = "meta.llama3-8b-instruct-v1:0";
/// `meta.llama3-70b-instruct-v1:0`
pub const LLAMA_3_70B_INSTRUCT: &str = "meta.llama3-70b-instruct-v1:0";
/// `meta.llama3-1-8b-instruct-v1:0`
pub const LLAMA_3_1_8B_INSTRUCT: &str = "meta.llama3-1-8b-instruct-v1:0";
/// `meta.llama3-1-70b-instruct-v1:0`
pub const LLAMA_3_1_70B_INSTRUCT: &str = "meta.llama3-1-70b-instruct-v1:0";
/// `meta.llama3-1-405b-instruct-v1:0`
pub const LLAMA_3_1_405B_INSTRUCT: &str = "meta.llama3-1-405b-instruct-v1:0";
/// `meta.llama3-2-1b-instruct-v1:0`
pub const LLAMA_3_2_1B_INSTRUCT: &str = "meta.llama3-2-1b-instruct-v1:0";
/// `meta.llama3-2-3b-instruct-v1:0`
pub const LLAMA_3_2_3B_INSTRUCT: &str = "meta.llama3-2-3b-instruct-v1:0";
/// `meta.llama3-2-11b-instruct-v1:0`
pub const LLAMA_3_2_11B_INSTRUCT: &str = "meta.llama3-2-11b-instruct-v1:0";
/// `meta.llama3-2-90b-instruct-v1:0`
pub const LLAMA_3_2_90B_INSTRUCT: &str = "meta.llama3-2-90b-instruct-v1:0";
/// `meta.llama3-3-70b-instruct-v1:0`
pub const META_LLAMA_3_3_70B_INSTRUCT: &str = "meta.llama3-3-70b-instruct-v1:0";
/// `meta.llama4-maverick-17b-instruct-v1:0`
pub const META_LLAMA_4_MAVERICK_17B_INSTRUCT: &str = "meta.llama4-maverick-17b-instruct-v1:0";
/// `meta.llama4-scout-17b-instruct-v1:0`
pub const META_LLAMA_4_SCOUT_17B_INSTRUCT: &str = "meta.llama4-scout-17b-instruct-v1:0";
/// `mistral.mistral-7b-instruct-v0:2`
pub const MISTRAL_7B_INSTRUCT: &str = "mistral.mistral-7b-instruct-v0:2";
/// `mistral.mistral-large-2402-v1:0`
pub const MISTRAL_LARGE_24_02: &str = "mistral.mistral-large-2402-v1:0";
/// `mistral.mistral-large-2407-v1:0`
pub const MISTRAL_LARGE_24_07: &str = "mistral.mistral-large-2407-v1:0";
/// `mistral.mistral-small-2402-v1:0`
pub const MISTRAL_SMALL_24_02: &str = "mistral.mistral-small-2402-v1:0";
/// `mistral.mixtral-8x7b-instruct-v0:1`
pub const MISTRAL_MIXTRAL_8X7B_INSTRUCT_V0: &str = "mistral.mixtral-8x7b-instruct-v0:1";
/// `mistral.pixtral-large-2502-v1:0`
pub const MISTRAL_PIXTRAL_LARGE_2502: &str = "mistral.pixtral-large-2502-v1:0";
/// `stability.sd3-5-large-v1:0`
pub const STABILITY_SD3_5_LARGE: &str = "stability.sd3-5-large-v1:0";
/// `stability.stable-image-core-v1:1`
pub const STABILITY_STABLE_IMAGE_CORE_1_0: &str = "stability.stable-image-core-v1:1";
/// `stability.stable-image-ultra-v1:1`
pub const STABILITY_STABLE_IMAGE_ULTRA_1_0: &str = "stability.stable-image-ultra-v1:1";
/// `twelvelabs.marengo-embed-2-7-v1:0`
pub const TWELVELABS_MARENGO_EMBED_V2_7: &str = "twelvelabs.marengo-embed-2-7-v1:0";
/// `twelvelabs.pegasus-1-2-v1:0`
pub const TWELVELABS_PEGASUS_V1_2: &str = "twelvelabs.pegasus-1-2-v1:0";
/// `writer.palmyra-x4-v1:0`
pub const WRITER_PALMYRA_X4: &str = "writer.palmyra-x4-v1:0";
/// `writer.palmyra-x5-v1:0`
pub const WRITER_PALMYRA_X5: &str = "writer.palmyra-x5-v1:0";
/// `ai21.jamba-instruct-v1:0`
pub const AI21_JAMBA_INSTRUCT: &str = "ai21.jamba-instruct-v1:0";
/// `anthropic.claude-v2:1`
pub const ANTHROPIC_CLAUDE_2_1: &str = "anthropic.claude-v2:1";
/// `anthropic.claude-v2`
pub const ANTHROPIC_CLAUDE_2: &str = "anthropic.claude-v2";
/// `anthropic.claude-instant-v1`
pub const ANTHROPIC_CLAUDE_INSTANT: &str = "anthropic.claude-instant-v1";
/// `anthropic.claude-instant-v1:2`
pub const ANTHROPIC_CLAUDE_INSTANT_V1_2: &str = "anthropic.claude-instant-v1:2";
/// `anthropic.claude-v2:0`
pub const ANTHROPIC_CLAUDE: &str = "anthropic.claude-v2:0";
/// `stability.sd3-large-v1:0`
pub const STABILITY_SD3_LARGE_1_0: &str = "stability.sd3-large-v1:0";
/// `stability.stable-diffusion-xl-v1`
pub const STABILITY_SDXL_1_0: &str = "stability.stable-diffusion-xl-v1";
/// `stability.stable-image-core-v1:0`
pub const STABILITY_STABLE_IMAGE_CORE_1_0_V1_0: &str = "stability.stable-image-core-v1:0";
/// `stability.stable-image-ultra-v1:0`
pub const STABILITY_STABLE_IMAGE_ULTRA_1_0_V1_0: &str = "stability.stable-image-ultra-v1:0";

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = AwsConverseOutput;
    type StreamingResponse = crate::streaming::BedrockStreamingResponse;

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<AwsConverseOutput>, CompletionError> {
        let request = AwsCompletionRequest(completion_request);

        let mut converse_builder = self
            .client
            .get_inner()
            .await
            .converse()
            .model_id(self.model.as_str());

        let tool_config = request.tools_config()?;
        let messages = request.messages()?;
        converse_builder = converse_builder
            .set_additional_model_request_fields(request.additional_params())
            .set_inference_config(request.inference_config())
            .set_tool_config(tool_config)
            .set_system(request.system_prompt())
            .set_messages(Some(messages));

        let response = converse_builder
            .send()
            .await
            .map_err(|sdk_error| Into::<CompletionError>::into(AwsSdkConverseError(sdk_error)))?;

        let response: InternalConverseOutput = response
            .try_into()
            .map_err(|x| CompletionError::ProviderError(format!("Type conversion error: {x}")))?;

        AwsConverseOutput(response).try_into()
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        CompletionModel::stream(self, request).await
    }
}
