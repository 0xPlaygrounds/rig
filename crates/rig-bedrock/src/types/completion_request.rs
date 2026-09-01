use crate::types::json::AwsDocument;
use crate::types::message::RigMessage;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use aws_sdk_bedrockruntime::types::{
    CachePointBlock, CachePointType, InferenceConfiguration, SystemContentBlock, Tool,
    ToolConfiguration, ToolInputSchema, ToolSpecification,
};
use rig_core::completion::{CompletionError, Message};
use rig_core::message::{DocumentMediaType, UserContent};

pub struct AwsCompletionRequest {
    pub inner: rig_core::completion::CompletionRequest,
    pub prompt_caching: bool,
}

fn cache_point_block() -> Result<CachePointBlock, CompletionError> {
    CachePointBlock::builder()
        .r#type(CachePointType::Default)
        .build()
        .map_err(|e| CompletionError::RequestError(e.into()))
}

impl AwsCompletionRequest {
    pub fn additional_params(&self) -> Option<aws_smithy_types::Document> {
        self.inner
            .additional_params
            .clone()
            .map(std::convert::Into::into)
            .map(|doc: AwsDocument| doc.0)
    }

    pub fn inference_config(&self) -> InferenceConfiguration {
        let mut inference_configuration = InferenceConfiguration::builder();

        if let Some(temperature) = &self.inner.temperature {
            inference_configuration =
                inference_configuration.set_temperature(Some(*temperature as f32));
        }

        if let Some(max_tokens) = &self.inner.max_tokens {
            inference_configuration =
                inference_configuration.set_max_tokens(Some(*max_tokens as i32));
        }

        inference_configuration.build()
    }

    pub fn tools_config(&self) -> Result<Option<ToolConfiguration>, CompletionError> {
        if self.inner.tool_choice == Some(rig_core::message::ToolChoice::None) {
            return Ok(None);
        }

        let tools = self
            .inner
            .tools
            .iter()
            .map(|tool_definition| {
                let doc: AwsDocument = tool_definition.parameters.clone().into();
                ToolSpecification::builder()
                    .name(tool_definition.name.clone())
                    .set_description(Some(tool_definition.description.clone()))
                    .set_input_schema(Some(ToolInputSchema::Json(doc.0)))
                    .build()
                    .map(Tool::ToolSpec)
                    .map_err(|e| CompletionError::RequestError(e.into()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if !tools.is_empty() {
            // Convert rig's ToolChoice to AWS Bedrock ToolChoice
            use aws_sdk_bedrockruntime::types as aws_bedrock;
            let tool_choice = self
                .inner
                .tool_choice
                .as_ref()
                .map(|choice| match choice {
                    rig_core::message::ToolChoice::Auto => Ok(Some(aws_bedrock::ToolChoice::Auto(
                        aws_bedrock::AutoToolChoice::builder().build(),
                    ))),
                    rig_core::message::ToolChoice::Required => Ok(Some(
                        aws_bedrock::ToolChoice::Any(aws_bedrock::AnyToolChoice::builder().build()),
                    )),
                    rig_core::message::ToolChoice::None => Ok(None),
                    rig_core::message::ToolChoice::Specific { function_names } => function_names
                        .first()
                        .map(|name| {
                            aws_bedrock::SpecificToolChoice::builder()
                                .name(name.clone())
                                .build()
                                .map(aws_bedrock::ToolChoice::Tool)
                                .map(Some)
                                .map_err(|e| CompletionError::RequestError(e.into()))
                        })
                        .transpose()
                        .map(Option::flatten),
                })
                .transpose()?
                .flatten();

            let config = ToolConfiguration::builder()
                .set_tools(Some(tools))
                .set_tool_choice(tool_choice)
                .build()
                .map_err(|e| CompletionError::RequestError(e.into()))?;

            Ok(Some(config))
        } else {
            Ok(None)
        }
    }

    /// Maps rig's `output_schema` to Bedrock's `OutputConfig` for structured output.
    pub fn output_config(&self) -> Result<Option<aws_bedrock::OutputConfig>, CompletionError> {
        let Some(schema) = self.inner.output_schema.as_ref() else {
            return Ok(None);
        };

        let schema_name = self
            .inner
            .output_schema_name()
            .unwrap_or_else(|| "response_schema".to_string());

        let schema_json = serde_json::to_string(schema.as_value())
            .map_err(|e| CompletionError::RequestError(e.into()))?;

        let json_schema_def = aws_bedrock::JsonSchemaDefinition::builder()
            .schema(schema_json)
            .name(schema_name)
            .build()
            .map_err(|e| CompletionError::RequestError(e.into()))?;

        let text_format = aws_bedrock::OutputFormat::builder()
            .r#type(aws_bedrock::OutputFormatType::JsonSchema)
            .structure(aws_bedrock::OutputFormatStructure::JsonSchema(
                json_schema_def,
            ))
            .build()
            .map_err(|e| CompletionError::RequestError(e.into()))?;

        Ok(Some(
            aws_bedrock::OutputConfig::builder()
                .text_format(text_format)
                .build(),
        ))
    }

    pub fn system_prompt(&self) -> Result<Option<Vec<SystemContentBlock>>, CompletionError> {
        let mut system_blocks = Vec::new();

        for message in self.inner.chat_history.iter() {
            if let Message::System { content } = message
                && !content.is_empty()
            {
                system_blocks.push(SystemContentBlock::Text(content.clone()));
            }
        }

        if system_blocks.is_empty() {
            Ok(None)
        } else {
            if self.prompt_caching {
                system_blocks.push(SystemContentBlock::CachePoint(cache_point_block()?));
            }
            Ok(Some(system_blocks))
        }
    }

    /// Consumes the request: this is the one accessor that needs the chat
    /// history by value, so call it after the borrowing accessors.
    pub fn messages(self) -> Result<Vec<aws_bedrock::Message>, CompletionError> {
        let mut full_history: Vec<Message> = Vec::new();

        if !self.inner.documents.is_empty() {
            let messages = self
                .inner
                .documents
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(" | ");

            let content = vec![UserContent::document(
                messages,
                Some(DocumentMediaType::TXT),
            )];

            full_history.push(Message::User { content });
        }

        // Compute before the history is moved below.
        let has_reasoning = self.inner.chat_history.iter().any(|message| match message {
            Message::Assistant { content, .. } => content
                .iter()
                .any(|c| matches!(c, rig_core::completion::AssistantContent::Reasoning(_))),
            _ => false,
        });

        full_history.extend(
            self.inner
                .chat_history
                .into_iter()
                .filter(|message| !matches!(message, Message::System { .. })),
        );

        let mut messages: Vec<aws_bedrock::Message> = full_history
            .into_iter()
            .map(|message| RigMessage(message).try_into())
            .collect::<Result<Vec<aws_bedrock::Message>, _>>()?;

        // Bedrock rejects cache points placed after reasoning blocks
        // ("Cache point cannot be inserted after reasoning block"). When the
        // request carries any reasoning content (round-tripped from a prior
        // turn), Anthropic's backend treats the trailing cache point as
        // following reasoning even when the literal previous block is a tool
        // result. Skip the message-level checkpoint in that case; the
        // system-prompt cache point still applies and captures the largest
        // stable prefix.
        if self.prompt_caching
            && !has_reasoning
            && let Some(last_msg) = messages.last_mut()
        {
            let mut content = std::mem::take(&mut last_msg.content);
            content.push(aws_bedrock::ContentBlock::CachePoint(cache_point_block()?));
            *last_msg = aws_bedrock::Message::builder()
                .role(last_msg.role.clone())
                .set_content(Some(content))
                .build()
                .map_err(|e| CompletionError::RequestError(e.into()))?;
        }

        Ok(messages)
    }
}

#[cfg(test)]
mod tests;
