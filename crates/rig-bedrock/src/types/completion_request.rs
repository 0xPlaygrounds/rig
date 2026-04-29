use crate::types::json::AwsDocument;
use crate::types::message::RigMessage;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use aws_sdk_bedrockruntime::types::{
    CachePointBlock, CachePointType, InferenceConfiguration, SystemContentBlock, Tool,
    ToolConfiguration, ToolInputSchema, ToolSpecification,
};
use rig::OneOrMany;
use rig::completion::{CompletionError, Message};
use rig::message::{DocumentMediaType, UserContent};

pub struct AwsCompletionRequest {
    pub inner: rig::completion::CompletionRequest,
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
            .to_owned()
            .map(|params| params.into())
            .map(|doc: AwsDocument| doc.0)
    }

    pub fn inference_config(&self) -> Option<InferenceConfiguration> {
        let mut inference_configuration = InferenceConfiguration::builder();

        if let Some(temperature) = &self.inner.temperature {
            inference_configuration =
                inference_configuration.set_temperature(Some(*temperature as f32));
        }

        if let Some(max_tokens) = &self.inner.max_tokens {
            inference_configuration =
                inference_configuration.set_max_tokens(Some(*max_tokens as i32));
        }

        Some(inference_configuration.build())
    }

    pub fn tools_config(&self) -> Result<Option<ToolConfiguration>, CompletionError> {
        let mut tools = vec![];
        for tool_definition in self.inner.tools.iter() {
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
            // Convert rig's ToolChoice to AWS Bedrock ToolChoice
            use aws_sdk_bedrockruntime::types as aws_bedrock;
            let tool_choice = self
                .inner
                .tool_choice
                .as_ref()
                .map(|choice| match choice {
                    rig::message::ToolChoice::Auto => Ok(Some(aws_bedrock::ToolChoice::Auto(
                        aws_bedrock::AutoToolChoice::builder().build(),
                    ))),
                    rig::message::ToolChoice::Required => Ok(Some(aws_bedrock::ToolChoice::Any(
                        aws_bedrock::AnyToolChoice::builder().build(),
                    ))),
                    rig::message::ToolChoice::None => Ok(None),
                    rig::message::ToolChoice::Specific { function_names } => function_names
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

    pub fn system_prompt(&self) -> Result<Option<Vec<SystemContentBlock>>, CompletionError> {
        let mut system_blocks = Vec::new();

        if let Some(system_prompt) = self.inner.preamble.to_owned()
            && !system_prompt.is_empty()
        {
            system_blocks.push(SystemContentBlock::Text(system_prompt));
        }

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

    pub fn messages(&self) -> Result<Vec<aws_bedrock::Message>, CompletionError> {
        let mut full_history: Vec<Message> = Vec::new();

        if !self.inner.documents.is_empty() {
            let messages = self
                .inner
                .documents
                .iter()
                .map(|doc| doc.to_string())
                .collect::<Vec<_>>()
                .join(" | ");

            let content = OneOrMany::one(UserContent::document(
                messages,
                Some(DocumentMediaType::TXT),
            ));

            full_history.push(Message::User { content });
        }

        self.inner.chat_history.iter().for_each(|message| {
            if !matches!(message, Message::System { .. }) {
                full_history.push(message.clone());
            }
        });

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
        let has_reasoning = self.inner.chat_history.iter().any(|message| match message {
            Message::Assistant { content, .. } => content
                .iter()
                .any(|c| matches!(c, rig::completion::AssistantContent::Reasoning(_))),
            _ => false,
        });

        if self.prompt_caching
            && !has_reasoning
            && let Some(last_msg) = messages.last_mut()
        {
            let mut content = last_msg.content.clone();
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
mod tests {
    use super::*;
    use rig::OneOrMany;
    use rig::completion::{CompletionRequest, ToolDefinition};
    use rig::message::{Message, Text, ToolChoice, UserContent};

    // Helper to create a minimal CompletionRequest for testing
    fn minimal_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::User {
                content: OneOrMany::one(UserContent::Text(Text {
                    text: "test".to_string(),
                })),
            }),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        }
    }

    fn aws_request(request: CompletionRequest, prompt_caching: bool) -> AwsCompletionRequest {
        AwsCompletionRequest {
            inner: request,
            prompt_caching,
        }
    }

    #[test]
    fn test_tool_choice_auto_conversion() {
        // Test that rig's ToolChoice::Auto converts to AWS Auto
        let request = CompletionRequest {
            model: None,
            tool_choice: Some(ToolChoice::Auto),
            tools: vec![ToolDefinition {
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }],
            ..minimal_request()
        };

        let aws_request = aws_request(request, false);
        let tool_config = aws_request
            .tools_config()
            .expect("Should build tool config");

        assert!(tool_config.is_some());

        let config = tool_config.unwrap();

        assert!(config.tool_choice().is_some());
        assert!(matches!(
            config.tool_choice().unwrap(),
            aws_bedrock::ToolChoice::Auto(_)
        ));
    }

    #[test]
    fn test_tool_choice_required_conversion() {
        // Test that rig's ToolChoice::Required converts to AWS Any
        let request = CompletionRequest {
            model: None,
            tool_choice: Some(ToolChoice::Required),
            tools: vec![ToolDefinition {
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }],
            ..minimal_request()
        };

        let aws_request = aws_request(request, false);
        let tool_config = aws_request
            .tools_config()
            .expect("Should build tool config");

        assert!(tool_config.is_some());
        let config = tool_config.unwrap();
        assert!(config.tool_choice().is_some());

        // Verify it's the Any variant
        assert!(matches!(
            config.tool_choice().unwrap(),
            aws_bedrock::ToolChoice::Any(_)
        ));
    }

    #[test]
    fn test_tool_choice_none_conversion() {
        // Test that rig's ToolChoice::None results in no tool_choice set
        let request = CompletionRequest {
            model: None,
            tool_choice: Some(ToolChoice::None),
            tools: vec![ToolDefinition {
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }],
            ..minimal_request()
        };

        let aws_request = aws_request(request, false);
        let tool_config = aws_request
            .tools_config()
            .expect("Should build tool config");

        assert!(tool_config.is_some());
        let config = tool_config.unwrap();
        // None should result in no tool_choice being set
        assert!(config.tool_choice().is_none());
    }

    #[test]
    fn test_tool_choice_specific_conversion() {
        // Test that rig's ToolChoice::Specific converts to AWS Tool
        let request = CompletionRequest {
            model: None,
            tool_choice: Some(ToolChoice::Specific {
                function_names: vec!["specific_tool".to_string()],
            }),
            tools: vec![ToolDefinition {
                name: "specific_tool".to_string(),
                description: "A specific tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }],
            ..minimal_request()
        };

        let aws_request = aws_request(request, false);
        let tool_config = aws_request
            .tools_config()
            .expect("Should build tool config");

        assert!(tool_config.is_some());

        let config = tool_config.unwrap();

        assert!(config.tool_choice().is_some());
        assert!(matches!(
            config.tool_choice().unwrap(),
            aws_bedrock::ToolChoice::Tool(specific) if specific.name() == "specific_tool"
        ));
    }

    #[test]
    fn test_no_tool_choice_when_not_specified() {
        // Test that when tool_choice is None (not set), it defaults to None in AWS
        let request = CompletionRequest {
            model: None,
            tool_choice: None, // Not set
            tools: vec![ToolDefinition {
                name: "test_tool".to_string(),
                description: "A test tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }],
            ..minimal_request()
        };

        let aws_request = aws_request(request, false);
        let tool_config = aws_request
            .tools_config()
            .expect("Should build tool config");

        assert!(tool_config.is_some());
        let config = tool_config.unwrap();
        // When not specified, should be None
        assert!(config.tool_choice().is_none());
    }

    #[test]
    fn test_tool_with_empty_parameters() {
        // Test that tools with empty parameters (like document_list) work correctly
        let request = CompletionRequest {
            model: None,
            tools: vec![ToolDefinition {
                name: "document_list".to_string(),
                description: "Lists all documents".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }],
            ..minimal_request()
        };

        let aws_request = aws_request(request, false);
        let tool_config = aws_request
            .tools_config()
            .expect("Should build tool config");

        assert!(tool_config.is_some());
        let config = tool_config.unwrap();
        assert_eq!(config.tools().len(), 1);

        // Verify the tool was created correctly
        assert!(
            matches!(&config.tools()[0], aws_bedrock::Tool::ToolSpec(spec)
                if spec.name() == "document_list"
                && spec.description() == Some("Lists all documents")
                && spec.input_schema().is_some()
            )
        );
    }

    #[test]
    fn test_tool_with_parameters() {
        // Test that tools with parameters work correctly
        let request = CompletionRequest {
            model: None,
            tools: vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: "Get weather for a location".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }),
            }],
            ..minimal_request()
        };

        let aws_request = aws_request(request, false);
        let tool_config = aws_request
            .tools_config()
            .expect("Should build tool config");

        assert!(tool_config.is_some());

        let config = tool_config.unwrap();

        assert_eq!(config.tools().len(), 1);
        assert!(
            matches!(&config.tools()[0], aws_bedrock::Tool::ToolSpec(spec)
                if spec.name() == "get_weather"
                && spec.description() == Some("Get weather for a location")
            )
        );
    }

    #[test]
    fn test_system_prompt_includes_system_history() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                Message::system("History system instruction"),
                Message::User {
                    content: OneOrMany::one(UserContent::Text(Text {
                        text: "test".to_string(),
                    })),
                },
            ])
            .expect("history should be non-empty"),
            ..minimal_request()
        };

        let aws_request = aws_request(request, false);
        let system_prompt = aws_request
            .system_prompt()
            .expect("system prompt should build")
            .expect("system prompt should exist");

        assert_eq!(system_prompt.len(), 1);
        assert_eq!(
            system_prompt.first(),
            Some(&aws_bedrock::SystemContentBlock::Text(
                "History system instruction".to_string()
            ))
        );
    }

    #[test]
    fn test_system_prompt_appends_cache_point_when_prompt_caching_enabled() {
        let request = CompletionRequest {
            preamble: Some("System prompt".to_string()),
            ..minimal_request()
        };

        let aws_request = aws_request(request, true);
        let system_prompt = aws_request
            .system_prompt()
            .expect("system prompt should build")
            .expect("system prompt should exist");

        assert_eq!(system_prompt.len(), 2);
        assert_eq!(
            system_prompt.first(),
            Some(&aws_bedrock::SystemContentBlock::Text(
                "System prompt".to_string()
            ))
        );
        assert!(matches!(
            system_prompt.last(),
            Some(aws_bedrock::SystemContentBlock::CachePoint(_))
        ));
    }

    #[test]
    fn test_messages_exclude_system_history() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                Message::system("History system instruction"),
                Message::User {
                    content: OneOrMany::one(UserContent::Text(Text {
                        text: "test".to_string(),
                    })),
                },
            ])
            .expect("history should be non-empty"),
            ..minimal_request()
        };

        let aws_request = aws_request(request, false);
        let messages = aws_request.messages().expect("messages should convert");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, aws_bedrock::ConversationRole::User);
    }

    #[test]
    fn test_messages_append_cache_point_when_prompt_caching_enabled() {
        let aws_request = aws_request(minimal_request(), true);

        let messages = aws_request.messages().expect("messages should convert");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, aws_bedrock::ConversationRole::User);
        assert_eq!(messages[0].content.len(), 2);
        assert!(matches!(
            messages[0].content.last(),
            Some(aws_bedrock::ContentBlock::CachePoint(_))
        ));
    }

    #[test]
    fn test_messages_skip_cache_point_when_history_contains_reasoning() {
        // Bedrock's Anthropic backend rejects "Cache point cannot be inserted
        // after reasoning block" whenever the chat history carries a prior
        // reasoning turn, even if the literal trailing block is a tool result.
        // Verify the message-level checkpoint is suppressed in that case.
        let reasoning =
            rig::message::Reasoning::new_with_signature("thinking", Some("sig".to_string()));
        let request = CompletionRequest {
            chat_history: OneOrMany::many(vec![
                Message::User {
                    content: OneOrMany::one(UserContent::Text(Text {
                        text: "user prompt".to_string(),
                    })),
                },
                Message::Assistant {
                    id: None,
                    content: OneOrMany::one(rig::completion::AssistantContent::Reasoning(
                        reasoning,
                    )),
                },
                Message::User {
                    content: OneOrMany::one(UserContent::Text(Text {
                        text: "follow up".to_string(),
                    })),
                },
            ])
            .expect("history should be non-empty"),
            ..minimal_request()
        };

        let aws_request = aws_request(request, true);
        let messages = aws_request.messages().expect("messages should convert");

        let last_message = messages.last().expect("messages should not be empty");
        assert!(
            !last_message
                .content
                .iter()
                .any(|c| matches!(c, aws_bedrock::ContentBlock::CachePoint(_))),
            "message-level cache point should be skipped when chat history contains reasoning"
        );

        // The system-prompt cache point path is independent and unaffected.
        let system_only = aws_request.system_prompt().expect("system prompt builds");
        assert!(system_only.is_none() || !system_only.unwrap().is_empty());
    }
}
