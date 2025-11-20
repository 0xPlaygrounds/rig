//! All supported models <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>

use crate::{
    client::Client,
    types::{
        assistant_content::AwsConverseOutput, completion_request::AwsCompletionRequest,
        converse_output::InternalConverseOutput, errors::AwsSdkConverseError,
    },
};

use rig::streaming::StreamingCompletionResponse;
use rig::{
    completion::{self, CompletionError, CompletionRequest},
    models,
};

models! {
    #[allow(non_camel_case_types)]
    pub enum CompletionModels {
        /// `ai21.jamba-1-5-large-v1:0`
        AI21_JAMBA_1_5_LARGE => "ai21.jamba-1-5-large-v1:0",
        /// `ai21.jamba-1-5-mini-v1:0`
        AI21_JAMBA_1_5_MINI => "ai21.jamba-1-5-mini-v1:0",
        /// `amazon.nova-canvas-v1:0`
        AMAZON_NOVA_CANVAS => "amazon.nova-canvas-v1:0",
        /// `amazon.nova-lite-v1:0`
        AMAZON_NOVA_LITE => "amazon.nova-lite-v1:0",
        /// `amazon.nova-micro-v1:0`
        AMAZON_NOVA_MICRO => "amazon.nova-micro-v1:0",
        /// `amazon.nova-premier-v1:0`
        AMAZON_NOVA_PREMIER => "amazon.nova-premier-v1:0",
        /// `amazon.nova-pro-v1:0`
        AMAZON_NOVA_PRO => "amazon.nova-pro-v1:0",
        /// `amazon.nova-reel-v1:0`
        AMAZON_NOVA_REEL_V1_0 => "amazon.nova-reel-v1:0",
        /// `amazon.nova-reel-v1:1`
        AMAZON_NOVA_REEL_V1_1 => "amazon.nova-reel-v1:1",
        /// `amazon.nova-sonic-v1:0`
        AMAZON_NOVA_SONIC => "amazon.nova-sonic-v1:0",
        /// `amazon.rerank-v1:0`
        AMAZON_RERANK_1_0 => "amazon.rerank-v1:0",
        /// `amazon.titan-embed-text-v1`
        AMAZON_TITAN_EMBEDDINGS_G1_TEXT => "amazon.titan-embed-text-v1",
        /// `amazon.titan-image-generator-v2:0`
        AMAZON_TITAN_IMAGE_GENERATOR_G1_V2 => "amazon.titan-image-generator-v2:0",
        /// `amazon.titan-image-generator-v1`
        AMAZON_TITAN_IMAGE_GENERATOR_G1 => "amazon.titan-image-generator-v1",
        /// `amazon.titan-embed-image-v1`
        AMAZON_TITAN_MULTIMODAL_EMBEDDINGS_G1 => "amazon.titan-embed-image-v1",
        /// `amazon.titan-embed-text-v2:0`
        AMAZON_TITAN_TEXT_EMBEDDINGS_V2 => "amazon.titan-embed-text-v2:0",
        /// `amazon.titan-text-express-v1`
        AMAZON_TITAN_TEXT_EXPRESS_V1 => "amazon.titan-text-express-v1",
        /// `amazon.titan-text-lite-v1`
        AMAZON_TITAN_TEXT_LITE_V1 => "amazon.titan-text-lite-v1",
        /// `amazon.titan-text-premier-v1:0`
        AMAZON_TITAN_TEXT_PREMIER_V1_0 => "amazon.titan-text-premier-v1:0",
        /// `anthropic.claude-3-haiku-20240307-v1:0`
        ANTHROPIC_CLAUDE_3_HAIKU => "anthropic.claude-3-haiku-20240307-v1:0",
        /// `anthropic.claude-3-opus-20240229-v1:0`
        ANTHROPIC_CLAUDE_3_OPUS => "anthropic.claude-3-opus-20240229-v1:0",
        /// `anthropic.claude-3-sonnet-20240229-v1:0`
        ANTHROPIC_CLAUDE_3_SONNET => "anthropic.claude-3-sonnet-20240229-v1:0",
        /// `anthropic.claude-3-5-haiku-20241022-v1:0`
        ANTHROPIC_CLAUDE_3_5_HAIKU => "anthropic.claude-3-5-haiku-20241022-v1:0",
        /// `anthropic.claude-3-5-sonnet-20241022-v2:0`
        ANTHROPIC_CLAUDE_3_5_SONNET_V2 => "anthropic.claude-3-5-sonnet-20241022-v2:0",
        /// `anthropic.claude-3-5-sonnet-20240620-v1:0`
        ANTHROPIC_CLAUDE_3_5_SONNET => "anthropic.claude-3-5-sonnet-20240620-v1:0",
        /// `anthropic.claude-3-7-sonnet-20250219-v1:0`
        ANTHROPIC_CLAUDE_3_7_SONNET => "anthropic.claude-3-7-sonnet-20250219-v1:0",
        /// `anthropic.claude-opus-4-20250514-v1:0`
        ANTHROPIC_CLAUDE_OPUS_4 => "anthropic.claude-opus-4-20250514-v1:0",
        /// `anthropic.claude-sonnet-4-20250514-v1:0`
        ANTHROPIC_CLAUDE_SONNET_4 => "anthropic.claude-sonnet-4-20250514-v1:0",
        /// `cohere.command-light-text-v14`
        COHERE_COMMAND_LIGHT_TEXT => "cohere.command-light-text-v14",
        /// `cohere.command-r-plus-v1:0`
        COHERE_COMMAND_R_PLUS => "cohere.command-r-plus-v1:0",
        /// `cohere.command-r-v1:0`
        COHERE_COMMAND_R => "cohere.command-r-v1:0",
        /// `cohere.command-text-v14`
        COHERE_COMMAND => "cohere.command-text-v14",
        /// `cohere.embed-english-v3`
        COHERE_EMBED_ENGLISH => "cohere.embed-english-v3",
        /// `cohere.embed-multilingual-v3`
        COHERE_EMBED_MULTILINGUAL => "cohere.embed-multilingual-v3",
        /// `cohere.rerank-v3-5:0`
        COHERE_RERANK_V3_5 => "cohere.rerank-v3-5:0",
        /// `deepseek.r1-v1:0`
        DEEPSEEK_R1 => "deepseek.r1-v1:0",
        /// `luma.ray-v2:0`
        LUMA_RAY_V2_0 => "luma.ray-v2:0",
        /// `meta.llama3-8b-instruct-v1:0`
        LLAMA_3_8B_INSTRUCT => "meta.llama3-8b-instruct-v1:0",
        /// `meta.llama3-70b-instruct-v1:0`
        LLAMA_3_70B_INSTRUCT => "meta.llama3-70b-instruct-v1:0",
        /// `meta.llama3-1-8b-instruct-v1:0`
        LLAMA_3_1_8B_INSTRUCT => "meta.llama3-1-8b-instruct-v1:0",
        /// `meta.llama3-1-70b-instruct-v1:0`
        LLAMA_3_1_70B_INSTRUCT => "meta.llama3-1-70b-instruct-v1:0",
        /// `meta.llama3-1-405b-instruct-v1:0`
        LLAMA_3_1_405B_INSTRUCT => "meta.llama3-1-405b-instruct-v1:0",
        /// `meta.llama3-2-1b-instruct-v1:0`
        LLAMA_3_2_1B_INSTRUCT => "meta.llama3-2-1b-instruct-v1:0",
        /// `meta.llama3-2-3b-instruct-v1:0`
        LLAMA_3_2_3B_INSTRUCT => "meta.llama3-2-3b-instruct-v1:0",
        /// `meta.llama3-2-11b-instruct-v1:0`
        LLAMA_3_2_11B_INSTRUCT => "meta.llama3-2-11b-instruct-v1:0",
        /// `meta.llama3-2-90b-instruct-v1:0`
        LLAMA_3_2_90B_INSTRUCT => "meta.llama3-2-90b-instruct-v1:0",
        /// `meta.llama3-3-70b-instruct-v1:0`
        META_LLAMA_3_3_70B_INSTRUCT => "meta.llama3-3-70b-instruct-v1:0",
        /// `meta.llama4-maverick-17b-instruct-v1:0`
        META_LLAMA_4_MAVERICK_17B_INSTRUCT => "meta.llama4-maverick-17b-instruct-v1:0",
        /// `meta.llama4-scout-17b-instruct-v1:0`
        META_LLAMA_4_SCOUT_17B_INSTRUCT => "meta.llama4-scout-17b-instruct-v1:0",
        /// `mistral.mistral-7b-instruct-v0:2`
        MISTRAL_7B_INSTRUCT => "mistral.mistral-7b-instruct-v0:2",
        /// `mistral.mistral-large-2402-v1:0`
        MISTRAL_LARGE_24_02 => "mistral.mistral-large-2402-v1:0",
        /// `mistral.mistral-large-2407-v1:0`
        MISTRAL_LARGE_24_07 => "mistral.mistral-large-2407-v1:0",
        /// `mistral.mistral-small-2402-v1:0`
        MISTRAL_SMALL_24_02 => "mistral.mistral-small-2402-v1:0",
        /// `mistral.mixtral-8x7b-instruct-v0:1`
        MISTRAL_MIXTRAL_8X7B_INSTRUCT_V0 => "mistral.mixtral-8x7b-instruct-v0:1",
        /// `mistral.pixtral-large-2502-v1:0`
        MISTRAL_PIXTRAL_LARGE_2502 => "mistral.pixtral-large-2502-v1:0",
        /// `stability.sd3-5-large-v1:0`
        STABILITY_SD3_5_LARGE => "stability.sd3-5-large-v1:0",
        /// `stability.stable-image-core-v1:1`
        STABILITY_STABLE_IMAGE_CORE_1_0 => "stability.stable-image-core-v1:1",
        /// `stability.stable-image-ultra-v1:1`
        STABILITY_STABLE_IMAGE_ULTRA_1_0 => "stability.stable-image-ultra-v1:1",
        /// `twelvelabs.marengo-embed-2-7-v1:0`
        TWELVELABS_MARENGO_EMBED_V2_7 => "twelvelabs.marengo-embed-2-7-v1:0",
        /// `twelvelabs.pegasus-1-2-v1:0`
        TWELVELABS_PEGASUS_V1_2 => "twelvelabs.pegasus-1-2-v1:0",
        /// `writer.palmyra-x4-v1:0`
        WRITER_PALMYRA_X4 => "writer.palmyra-x4-v1:0",
        /// `writer.palmyra-x5-v1:0`
        WRITER_PALMYRA_X5 => "writer.palmyra-x5-v1:0",
        /// `ai21.jamba-instruct-v1:0`
        AI21_JAMBA_INSTRUCT => "ai21.jamba-instruct-v1:0",
        /// `anthropic.claude-v2:1`
        ANTHROPIC_CLAUDE_2_1 => "anthropic.claude-v2:1",
        /// `anthropic.claude-v2`
        ANTHROPIC_CLAUDE_2 => "anthropic.claude-v2",
        /// `anthropic.claude-instant-v1`
        ANTHROPIC_CLAUDE_INSTANT => "anthropic.claude-instant-v1",
        /// `anthropic.claude-instant-v1:2`
        ANTHROPIC_CLAUDE_INSTANT_V1_2 => "anthropic.claude-instant-v1:2",
        /// `anthropic.claude-v2:0`
        ANTHROPIC_CLAUDE => "anthropic.claude-v2:0",
        /// `stability.sd3-large-v1:0`
        STABILITY_SD3_LARGE_1_0 => "stability.sd3-large-v1:0",
        /// `stability.stable-diffusion-xl-v1`
        STABILITY_SDXL_1_0 => "stability.stable-diffusion-xl-v1",
        /// `stability.stable-image-core-v1:0`
        STABILITY_STABLE_IMAGE_CORE_1_0_V1_0 => "stability.stable-image-core-v1:0",
        /// `stability.stable-image-ultra-v1:0`
        STABILITY_STABLE_IMAGE_ULTRA_1_0_V1_0 => "stability.stable-image-ultra-v1:0",
    }
}
pub use CompletionModels::*;

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

    pub fn with_model(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = AwsConverseOutput;
    type StreamingResponse = crate::streaming::BedrockStreamingResponse;

    type Client = Client;
    type Models = CompletionModels;

    fn make(client: &Self::Client, model: impl Into<Self::Models>) -> Self {
        Self::new(client.clone(), model.into().into())
    }

    fn make_custom(client: &Self::Client, model: &str) -> Self {
        Self::with_model(client.clone(), model)
    }

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
