use crate::types::json::AwsDocument;
use crate::types::message::RigMessage;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use aws_sdk_bedrockruntime::types::{
    InferenceConfiguration, SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema,
    ToolSpecification,
};
use rig::completion::{CompletionError, Message};

pub struct AwsCompletionRequest(pub rig::completion::CompletionRequest);

impl AwsCompletionRequest {
    pub fn additional_params(&self) -> Option<aws_smithy_types::Document> {
        self.0
            .additional_params
            .to_owned()
            .map(|params| params.into())
            .map(|doc: AwsDocument| doc.0)
    }

    pub fn inference_config(&self) -> Option<InferenceConfiguration> {
        let mut inference_configuration = InferenceConfiguration::builder();

        if let Some(temperature) = &self.0.temperature {
            inference_configuration =
                inference_configuration.set_temperature(Some(*temperature as f32));
        }

        if let Some(max_tokens) = &self.0.max_tokens {
            inference_configuration =
                inference_configuration.set_max_tokens(Some(*max_tokens as i32));
        }

        Some(inference_configuration.build())
    }

    pub fn tools_config(&self) -> Result<Option<ToolConfiguration>, CompletionError> {
        let mut tools = vec![];
        for tool_definition in self.0.tools.iter() {
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

            Ok(Some(config))
        } else {
            Ok(None)
        }
    }

    pub fn system_prompt(&self) -> Option<Vec<SystemContentBlock>> {
        self.0
            .preamble
            .to_owned()
            .map(|system_prompt| vec![SystemContentBlock::Text(system_prompt)])
    }

    pub fn prompt_with_history(&self) -> Result<Vec<aws_bedrock::Message>, CompletionError> {
        let mut chat_history = self.0.chat_history.to_owned();
        let prompt_with_context = self.0.prompt_with_context();

        let mut full_history: Vec<Message> = Vec::new();
        full_history.append(&mut chat_history);
        full_history.push(prompt_with_context);

        full_history
            .into_iter()
            .map(|message| RigMessage(message).try_into())
            .collect::<Result<Vec<aws_bedrock::Message>, _>>()
    }
}
