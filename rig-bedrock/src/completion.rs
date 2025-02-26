use crate::{
    client::Client,
    types::{
        assistent_content::AwsConverseOutput, errors::AwsSdkConverseError, json::AwsDocument,
        message::MessageWithPrompt,
    },
};
use aws_sdk_bedrockruntime::types as aws_bedrock;

use aws_sdk_bedrockruntime::types::{
    InferenceConfiguration, SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema,
    ToolSpecification,
};
use rig::completion::{self, CompletionError};

// All supported models: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
/// `amazon.nova-lite-v1` foundational model
pub const AMAZON_NOVA_LITE_V1: &str = "amazon.nova-lite-v1:0";
/// `mistral.mixtral-8x7b-instruct-v0` foundational model
pub const MISTRAL_MIXTRAL_8X7B_INSTRUCT_V0: &str = "mistral.mixtral-8x7b-instruct-v0:1";

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
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
        mut completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<AwsConverseOutput>, CompletionError> {
        let mut full_history = Vec::new();
        full_history.append(&mut completion_request.chat_history);
        full_history.push(completion_request.prompt_with_context());

        let prompt_with_history = full_history
            .into_iter()
            .map(|message| {
                MessageWithPrompt {
                    message,
                    prompt: completion_request.preamble.clone(),
                }
                .try_into()
            })
            .collect::<Result<Vec<aws_bedrock::Message>, _>>()?;

        let mut converse_builder = self
            .client
            .aws_client
            .converse()
            .model_id(self.model.as_str());

        let mut inference_configuration = InferenceConfiguration::builder();

        if let Some(params) = completion_request.additional_params {
            let doc: AwsDocument = params.into();
            converse_builder = converse_builder.set_additional_model_request_fields(Some(doc.0));
        }

        if let Some(temperature) = completion_request.temperature {
            inference_configuration =
                inference_configuration.set_temperature(Some(temperature as f32));
        }

        if let Some(max_tokens) = completion_request.max_tokens {
            inference_configuration =
                inference_configuration.set_max_tokens(Some(max_tokens as i32));
        }

        converse_builder =
            converse_builder.set_inference_config(Some(inference_configuration.build()));

        let mut tools = vec![];
        for tool_definition in completion_request.tools.iter() {
            let doc: AwsDocument = tool_definition.parameters.clone().into();
            let schema = ToolInputSchema::Json(doc.0);
            let tool = Tool::ToolSpec(
                ToolSpecification::builder()
                    .name(tool_definition.name.clone())
                    .set_description(Some(tool_definition.description.clone()))
                    .set_input_schema(Some(schema))
                    .build()
                    .map_err(|e| CompletionError::RequestError(e.into()))?,
            );
            tools.push(tool);
        }

        if !tools.is_empty() {
            let config = ToolConfiguration::builder()
                .set_tools(Some(tools))
                .build()
                .map_err(|e| CompletionError::RequestError(e.into()))?;

            converse_builder = converse_builder.set_tool_config(Some(config));
        }

        if let Some(system_prompt) = completion_request.preamble {
            converse_builder =
                converse_builder.set_system(Some(vec![SystemContentBlock::Text(system_prompt)]));
        }

        let model_response = converse_builder
            .set_messages(Some(prompt_with_history))
            .send()
            .await;

        let response = model_response
            .map_err(|sdk_error| AwsSdkConverseError(sdk_error).into())
            .map_err(|e: CompletionError| e)?;

        AwsConverseOutput(response).try_into()
    }
}
