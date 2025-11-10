use crate::types::message::RigMessage;
use google_cloud_aiplatform_v1 as vertexai;
use rig::completion::CompletionError;

pub struct VertexCompletionRequest(pub rig::completion::CompletionRequest);

impl VertexCompletionRequest {
    pub fn contents(&self) -> Result<Vec<vertexai::model::Content>, CompletionError> {
        let mut contents = Vec::new();

        for message in self.0.chat_history.iter() {
            let content = RigMessage(message.clone()).try_into()?;
            contents.push(content);
        }

        Ok(contents)
    }

    pub fn system_instruction(&self) -> Option<vertexai::model::Content> {
        self.0.preamble.as_ref().map(|preamble| {
            vertexai::model::Content::new()
                .set_role("user")
                .set_parts([vertexai::model::Part::new().set_text(preamble.clone())])
        })
    }

    pub fn tools(&self) -> Option<vertexai::model::Tool> {
        if self.0.tools.is_empty() {
            return None;
        }

        let function_declarations: Vec<vertexai::model::FunctionDeclaration> = self
            .0
            .tools
            .iter()
            .map(|tool_def| {
                vertexai::model::FunctionDeclaration::new()
                    .set_name(tool_def.name.clone())
                    .set_description(tool_def.description.clone())
                    .set_parameters_json_schema(tool_def.parameters.clone())
            })
            .collect();

        Some(vertexai::model::Tool::new().set_function_declarations(function_declarations))
    }

    pub fn tool_config(&self) -> Option<vertexai::model::ToolConfig> {
        if self.0.tools.is_empty() {
            return None;
        }

        let function_calling_config = match self.0.tool_choice.as_ref() {
            Some(rig::message::ToolChoice::Auto) => vertexai::model::FunctionCallingConfig::new()
                .set_mode(vertexai::model::function_calling_config::Mode::Auto),
            Some(rig::message::ToolChoice::Required) => {
                vertexai::model::FunctionCallingConfig::new()
                    .set_mode(vertexai::model::function_calling_config::Mode::Any)
            }
            Some(rig::message::ToolChoice::None) => vertexai::model::FunctionCallingConfig::new()
                .set_mode(vertexai::model::function_calling_config::Mode::None),
            Some(rig::message::ToolChoice::Specific { function_names }) => {
                vertexai::model::FunctionCallingConfig::new()
                    .set_mode(vertexai::model::function_calling_config::Mode::Any)
                    .set_allowed_function_names(function_names.clone())
            }
            None => vertexai::model::FunctionCallingConfig::new()
                .set_mode(vertexai::model::function_calling_config::Mode::Auto),
        };

        Some(
            vertexai::model::ToolConfig::new().set_function_calling_config(function_calling_config),
        )
    }

    pub fn generation_config(&self) -> Option<vertexai::model::GenerationConfig> {
        let mut config = vertexai::model::GenerationConfig::new();

        if let Some(temperature) = self.0.temperature {
            config = config.set_temperature(temperature as f32);
        }

        if let Some(max_tokens) = self.0.max_tokens {
            config = config.set_max_output_tokens(max_tokens as i32);
        }

        config = config.set_candidate_count(1);

        Some(config)
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
        }
    }

    #[test]
    fn test_tool_choice_auto_conversion() {
        // Test that rig's ToolChoice::Auto converts to Vertex AI Auto mode
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

        let vertex_request = VertexCompletionRequest(request);
        let tool_config = vertex_request.tool_config();

        assert!(tool_config.is_some());
        let config = tool_config.unwrap();
        let function_calling_config = config.function_calling_config.as_ref();

        assert!(function_calling_config.is_some());
        assert_eq!(
            function_calling_config.unwrap().mode,
            vertexai::model::function_calling_config::Mode::Auto
        );
    }

    #[test]
    fn test_tool_choice_required_conversion() {
        // Test that rig's ToolChoice::Required converts to Vertex AI Any mode
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

        let vertex_request = VertexCompletionRequest(request);
        let tool_config = vertex_request.tool_config();

        assert!(tool_config.is_some());
        let config = tool_config.unwrap();
        let function_calling_config = config.function_calling_config.as_ref();

        assert!(function_calling_config.is_some());
        assert_eq!(
            function_calling_config.unwrap().mode,
            vertexai::model::function_calling_config::Mode::Any
        );
    }

    #[test]
    fn test_tool_choice_none_conversion() {
        // Test that rig's ToolChoice::None converts to Vertex AI None mode
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

        let vertex_request = VertexCompletionRequest(request);
        let tool_config = vertex_request.tool_config();

        assert!(tool_config.is_some());
        let config = tool_config.unwrap();
        let function_calling_config = config.function_calling_config.as_ref();

        assert!(function_calling_config.is_some());
        assert_eq!(
            function_calling_config.unwrap().mode,
            vertexai::model::function_calling_config::Mode::None
        );
    }

    #[test]
    fn test_tool_choice_specific_conversion() {
        // Test that rig's ToolChoice::Specific converts to Vertex AI Any mode with allowed function names
        let request = CompletionRequest {
            tool_choice: Some(ToolChoice::Specific {
                function_names: vec!["test_tool".to_string()],
            }),
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

        let vertex_request = VertexCompletionRequest(request);
        let tool_config = vertex_request.tool_config();

        assert!(tool_config.is_some());
        let config = tool_config.unwrap();
        let function_calling_config = config.function_calling_config.as_ref();

        assert!(function_calling_config.is_some());
        let fcc = function_calling_config.unwrap();
        assert_eq!(
            fcc.mode,
            vertexai::model::function_calling_config::Mode::Any
        );
        // Verify allowed function names are set
        assert!(!fcc.allowed_function_names.is_empty());
        assert_eq!(fcc.allowed_function_names.len(), 1);
        assert_eq!(fcc.allowed_function_names[0], "test_tool");
    }

    #[test]
    fn test_system_instruction_from_preamble() {
        // Test that preamble converts to system instruction
        let request = CompletionRequest {
            preamble: Some("You are a helpful assistant.".to_string()),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let system_instruction = vertex_request.system_instruction();

        assert!(system_instruction.is_some());
        let content = system_instruction.unwrap();
        assert_eq!(content.role.as_str(), "user");
        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0].text(),
            Some(&"You are a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_tools_conversion() {
        // Test that ToolDefinition converts to FunctionDeclaration
        let request = CompletionRequest {
            tools: vec![ToolDefinition {
                name: "add".to_string(),
                description: "Add two numbers".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    },
                    "required": ["x", "y"]
                }),
            }],
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let tools = vertex_request.tools();

        assert!(tools.is_some());
        let tool = tools.unwrap();
        // Verify function declarations exist
        assert!(!tool.function_declarations.is_empty());
        assert_eq!(tool.function_declarations.len(), 1);
        assert_eq!(tool.function_declarations[0].name.as_str(), "add");
        assert_eq!(
            tool.function_declarations[0].description.as_str(),
            "Add two numbers"
        );
    }

    #[test]
    fn test_no_tool_choice_when_not_specified() {
        // Test that when tool_choice is None (not set), it defaults to Auto in Vertex AI
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

        let vertex_request = VertexCompletionRequest(request);
        let tool_config = vertex_request.tool_config();

        assert!(tool_config.is_some());
        let config = tool_config.unwrap();
        let function_calling_config = config.function_calling_config.as_ref();

        assert!(function_calling_config.is_some());
        // When not specified, should default to Auto
        assert_eq!(
            function_calling_config.unwrap().mode,
            vertexai::model::function_calling_config::Mode::Auto
        );
    }

    #[test]
    fn test_tool_with_empty_parameters() {
        // Test that tools with empty parameters work correctly
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

        let vertex_request = VertexCompletionRequest(request);
        let tools = vertex_request.tools();

        assert!(tools.is_some());
        let tool = tools.unwrap();
        assert!(!tool.function_declarations.is_empty());
        assert_eq!(tool.function_declarations.len(), 1);
        assert_eq!(tool.function_declarations[0].name.as_str(), "document_list");
        assert_eq!(
            tool.function_declarations[0].description.as_str(),
            "Lists all documents"
        );
    }

    #[test]
    fn test_tool_with_parameters() {
        // Test that tools with complex parameters work correctly
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

        let vertex_request = VertexCompletionRequest(request);
        let tools = vertex_request.tools();

        assert!(tools.is_some());
        let tool = tools.unwrap();
        assert!(!tool.function_declarations.is_empty());
        assert_eq!(tool.function_declarations.len(), 1);
        assert_eq!(tool.function_declarations[0].name.as_str(), "get_weather");
        assert_eq!(
            tool.function_declarations[0].description.as_str(),
            "Get weather for a location"
        );
    }

    #[test]
    fn test_generation_config_with_temperature_and_max_tokens() {
        // Test that temperature and max_tokens convert to GenerationConfig
        let request = CompletionRequest {
            temperature: Some(0.7),
            max_tokens: Some(100),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let generation_config = vertex_request.generation_config();

        assert!(generation_config.is_some());
        let config = generation_config.unwrap();
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_output_tokens, Some(100));
        assert_eq!(config.candidate_count, Some(1));
    }
}
