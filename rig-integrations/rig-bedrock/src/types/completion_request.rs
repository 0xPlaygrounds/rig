use crate::types::json::AwsDocument;
use crate::types::message::RigMessage;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use aws_sdk_bedrockruntime::types::{
    InferenceConfiguration, SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema,
    ToolSpecification,
};
use rig::OneOrMany;
use rig::completion::{CompletionError, Message};
use rig::message::{DocumentMediaType, UserContent};

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
            // Convert rig's ToolChoice to AWS Bedrock ToolChoice
            use aws_sdk_bedrockruntime::types as aws_bedrock;
            let tool_choice = self.0.tool_choice.as_ref().and_then(|choice| {
                match choice {
                    rig::message::ToolChoice::Auto => Some(aws_bedrock::ToolChoice::Auto(
                        aws_bedrock::AutoToolChoice::builder().build(),
                    )),
                    rig::message::ToolChoice::Required => Some(aws_bedrock::ToolChoice::Any(
                        aws_bedrock::AnyToolChoice::builder().build(),
                    )),
                    rig::message::ToolChoice::None => {
                        // Bedrock doesn't have a "None" option - just omit tool_choice
                        None
                    }
                    rig::message::ToolChoice::Specific { function_names } => {
                        // Use the first function name for Bedrock's specific tool choice
                        function_names.first().map(|name| {
                            aws_bedrock::ToolChoice::Tool(
                                aws_bedrock::SpecificToolChoice::builder()
                                    .name(name.clone())
                                    .build()
                                    .expect("Failed to build SpecificToolChoice"),
                            )
                        })
                    }
                }
            });

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

    pub fn system_prompt(&self) -> Option<Vec<SystemContentBlock>> {
        self.0
            .preamble
            .to_owned()
            .map(|system_prompt| vec![SystemContentBlock::Text(system_prompt)])
    }

    pub fn messages(&self) -> Result<Vec<aws_bedrock::Message>, CompletionError> {
        let mut full_history: Vec<Message> = Vec::new();

        if !self.0.documents.is_empty() {
            let messages = self
                .0
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

        self.0.chat_history.iter().for_each(|message| {
            full_history.push(message.clone());
        });

        full_history
            .into_iter()
            .map(|message| RigMessage(message).try_into())
            .collect::<Result<Vec<aws_bedrock::Message>, _>>()
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

    #[test]
    fn test_tool_choice_auto_conversion() {
        // Test that rig's ToolChoice::Auto converts to AWS Auto
        let request = CompletionRequest {
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

        let aws_request = AwsCompletionRequest(request);
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

        let aws_request = AwsCompletionRequest(request);
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

        let aws_request = AwsCompletionRequest(request);
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

        let aws_request = AwsCompletionRequest(request);
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

        let aws_request = AwsCompletionRequest(request);
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

        let aws_request = AwsCompletionRequest(request);
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

        let aws_request = AwsCompletionRequest(request);
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
}
