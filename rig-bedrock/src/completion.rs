//! All supported models <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>
use crate::{
    client::Client,
    types::{
        assistant_content::AwsConverseOutput, completion_request::AwsCompletionRequest,
        errors::AwsSdkConverseError,
    },
};

use rig::completion::{self, CompletionError};

/// `amazon.nova-canvas-v1:0`
pub const AMAZON_NOVA_CANVAS: &str = "amazon.nova-canvas-v1:0";
/// `amazon.nova-lite-v1:0`
pub const AMAZON_NOVA_LITE: &str = "amazon.nova-lite-v1:0";
/// `amazon.nova-micro-v1:0`
pub const AMAZON_NOVA_MICRO: &str = "amazon.nova-micro-v1:0";
/// `amazon.nova-pro-v1:0`
pub const AMAZON_NOVA_PRO: &str = "amazon.nova-pro-v1:0";
/// `amazon.rerank-v1:0`
pub const AMAZON_RERANK_1_0: &str = "amazon.rerank-v1:0";
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
/// `cohere.command-light-text-v14`
pub const COHERE_COMMAND_LIGHT_TEXT: &str = "cohere.command-light-text-v14";
/// `cohere.command-r-plus-v1:0`
pub const COHERE_COMMAND_R_PLUS: &str = "cohere.command-r-plus-v1:0";
/// `cohere.command-r-v1:0`
pub const COHERE_COMMAND_R: &str = "cohere.command-r-v1:0";
/// `cohere.command-text-v14`
pub const COHERE_COMMAND: &str = "cohere.command-text-v14";
/// `cohere.rerank-v3-5:0`
pub const COHERE_RERANK_V3_5: &str = "cohere.rerank-v3-5:0";
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
pub const LLAMA_3_2_70B_INSTRUCT: &str = "meta.llama3-3-70b-instruct-v1:0";
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
/// `stability.sd3-5-large-v1:0`
pub const STABILITY_SD3_5_LARGE: &str = "stability.sd3-5-large-v1:0";
/// `ai21.jamba-1-5-large-v1:0`
pub const JAMBA_1_5_LARGE: &str = "ai21.jamba-1-5-large-v1:0";
/// `ai21.jamba-1-5-mini-v1:0`
pub const JAMBA_1_5_MINI: &str = "ai21.jamba-1-5-mini-v1:0";

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

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<AwsConverseOutput>, CompletionError> {
        let request = AwsCompletionRequest(completion_request);

        let mut converse_builder = self
            .client
            .aws_client
            .converse()
            .model_id(self.model.as_str());

        let tool_config = request.tools_config()?;
        let prompt_with_history = request.prompt_with_history()?;
        converse_builder = converse_builder
            .set_additional_model_request_fields(request.additional_params())
            .set_inference_config(request.inference_config())
            .set_tool_config(tool_config)
            .set_system(request.system_prompt())
            .set_messages(Some(prompt_with_history));

        let response = converse_builder
            .send()
            .await
            .map_err(|sdk_error| Into::<CompletionError>::into(AwsSdkConverseError(sdk_error)))?;

        AwsConverseOutput(response).try_into()
    }
}
