use crate::types::message::RigMessage;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::CompletionError;
use rig_core::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, ThinkingConfig as GeminiThinkingConfig,
    ThinkingLevel as GeminiThinkingLevel,
};

pub struct VertexCompletionRequest(pub rig_core::completion::CompletionRequest);

impl VertexCompletionRequest {
    pub fn contents(&self) -> Result<Vec<vertexai::model::Content>, CompletionError> {
        let mut contents = Vec::new();

        for message in self.0.chat_history.iter() {
            if matches!(message, rig_core::completion::Message::System { .. }) {
                continue;
            }
            let content = RigMessage(message.clone()).try_into()?;
            contents.push(content);
        }

        Ok(contents)
    }

    pub fn system_instruction(&self) -> Option<vertexai::model::Content> {
        let mut system_texts = Vec::new();
        if let Some(preamble) = self.0.preamble.as_ref()
            && !preamble.is_empty()
        {
            system_texts.push(preamble.clone());
        }

        for message in self.0.chat_history.iter() {
            if let rig_core::completion::Message::System { content } = message
                && !content.is_empty()
            {
                system_texts.push(content.clone());
            }
        }

        if system_texts.is_empty() {
            return None;
        }

        Some(
            vertexai::model::Content::new()
                .set_role("user")
                .set_parts([vertexai::model::Part::new().set_text(system_texts.join("\n\n"))]),
        )
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
            Some(rig_core::message::ToolChoice::Auto) => {
                vertexai::model::FunctionCallingConfig::new()
                    .set_mode(vertexai::model::function_calling_config::Mode::Auto)
            }
            Some(rig_core::message::ToolChoice::Required) => {
                vertexai::model::FunctionCallingConfig::new()
                    .set_mode(vertexai::model::function_calling_config::Mode::Any)
            }
            Some(rig_core::message::ToolChoice::None) => {
                vertexai::model::FunctionCallingConfig::new()
                    .set_mode(vertexai::model::function_calling_config::Mode::None)
            }
            Some(rig_core::message::ToolChoice::Specific { function_names }) => {
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

    /// Builds the Vertex `GenerationConfig`, applying `additional_params.generation_config`
    /// (same shape and field names, camelCase, as `rig_core`'s Gemini provider
    /// `AdditionalParameters` — see `rig_core::providers::gemini::completion::gemini_api_types`)
    /// before the typed request fields, so `temperature`/`max_tokens` always win over any
    /// duplicate set through `additional_params` (the typed surface is authoritative).
    ///
    /// # Errors
    ///
    /// Returns `CompletionError::JsonError` if `additional_params` is present but does not
    /// deserialize into the expected shape.
    pub fn generation_config(
        &self,
    ) -> Result<Option<vertexai::model::GenerationConfig>, CompletionError> {
        let additional = match self.0.additional_params.clone() {
            Some(value) => serde_json::from_value::<AdditionalParameters>(value)?,
            None => AdditionalParameters::default(),
        };

        let mut config = vertexai::model::GenerationConfig::new();
        let mut candidate_count_set = false;

        if let Some(gemini_config) = additional.generation_config {
            if let Some(candidate_count) = gemini_config.candidate_count {
                config = config.set_candidate_count(candidate_count);
                candidate_count_set = true;
            }
            if let Some(temperature) = gemini_config.temperature {
                config = config.set_temperature(temperature as f32);
            }
            if let Some(max_output_tokens) = gemini_config.max_output_tokens {
                config = config.set_max_output_tokens(max_output_tokens as i32);
            }
            if let Some(thinking_config) = gemini_config.thinking_config {
                config = config.set_thinking_config(map_thinking_config(thinking_config));
            }
        }

        // Typed request fields win over additional_params duplicates.
        if let Some(temperature) = self.0.temperature {
            config = config.set_temperature(temperature as f32);
        }
        if let Some(max_tokens) = self.0.max_tokens {
            config = config.set_max_output_tokens(max_tokens as i32);
        }
        if !candidate_count_set {
            config = config.set_candidate_count(1);
        }

        Ok(Some(config))
    }
}

/// Maps the Gemini provider's `additional_params` thinking config onto Vertex's own
/// `generation_config::ThinkingConfig` — same fields (`thinking_budget`/`thinking_level`/
/// `include_thoughts`), no local validation of the `thinking_budget`/`thinking_level` mutual
/// exclusion the Gemini provider documents (that request-validity concern belongs to the API,
/// not this conversion, matching how the Gemini provider itself does not enforce it locally).
fn map_thinking_config(
    thinking_config: GeminiThinkingConfig,
) -> vertexai::model::generation_config::ThinkingConfig {
    let mut config = vertexai::model::generation_config::ThinkingConfig::new();

    if let Some(thinking_budget) = thinking_config.thinking_budget {
        config = config.set_thinking_budget(thinking_budget as i32);
    }
    if let Some(thinking_level) = thinking_config.thinking_level {
        config = config.set_thinking_level(map_thinking_level(thinking_level));
    }
    if let Some(include_thoughts) = thinking_config.include_thoughts {
        config = config.set_include_thoughts(include_thoughts);
    }

    config
}

fn map_thinking_level(
    thinking_level: GeminiThinkingLevel,
) -> vertexai::model::generation_config::thinking_config::ThinkingLevel {
    use vertexai::model::generation_config::thinking_config::ThinkingLevel as VertexThinkingLevel;

    match thinking_level {
        GeminiThinkingLevel::Minimal => VertexThinkingLevel::Minimal,
        GeminiThinkingLevel::Low => VertexThinkingLevel::Low,
        GeminiThinkingLevel::Medium => VertexThinkingLevel::Medium,
        GeminiThinkingLevel::High => VertexThinkingLevel::High,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::OneOrMany;
    use rig_core::completion::{CompletionRequest, ToolDefinition};
    use rig_core::message::{Message, Text, ToolChoice, UserContent};

    // Helper to create a minimal CompletionRequest for testing
    fn minimal_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(Message::User {
                content: OneOrMany::one(UserContent::Text(Text::new("test".to_string()))),
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
        // Test that rig's ToolChoice::Auto converts to Vertex AI Auto mode
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
            model: None,
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
            model: None,
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
    fn test_system_instruction_from_system_history_and_contents_skip_system() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                Message::system("System from history"),
                Message::User {
                    content: OneOrMany::one(UserContent::Text(Text::new("hello".to_string()))),
                },
            ])
            .expect("history should be non-empty"),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);

        let system_instruction = vertex_request.system_instruction();
        assert!(system_instruction.is_some());
        let system_instruction = system_instruction.unwrap();
        assert_eq!(system_instruction.parts.len(), 1);
        assert_eq!(
            system_instruction.parts[0].text(),
            Some(&"System from history".to_string())
        );

        let contents = vertex_request.contents().expect("contents should convert");
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_str(), "user");
    }

    #[test]
    fn test_tools_conversion() {
        // Test that ToolDefinition converts to FunctionDeclaration
        let request = CompletionRequest {
            model: None,
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
            model: None,
            temperature: Some(0.7),
            max_tokens: Some(100),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let generation_config = vertex_request
            .generation_config()
            .expect("no additional_params, so conversion cannot fail");

        assert!(generation_config.is_some());
        let config = generation_config.unwrap();
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_output_tokens, Some(100));
        assert_eq!(config.candidate_count, Some(1));
    }

    fn additional_params_generation_config(json: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "generationConfig": json })
    }

    #[test]
    fn test_generation_config_thinking_budget_from_additional_params() {
        let request = CompletionRequest {
            model: None,
            additional_params: Some(additional_params_generation_config(serde_json::json!({
                "thinkingConfig": { "thinkingBudget": 1024, "includeThoughts": true }
            }))),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let config = vertex_request
            .generation_config()
            .expect("valid additional_params")
            .expect("generation_config is always Some");

        let thinking = config
            .thinking_config
            .expect("thinking_config should be set");
        assert_eq!(thinking.thinking_budget, Some(1024));
        assert_eq!(thinking.include_thoughts, Some(true));
        assert_eq!(thinking.thinking_level, None);
    }

    #[test]
    fn test_generation_config_thinking_level_from_additional_params() {
        let request = CompletionRequest {
            model: None,
            additional_params: Some(additional_params_generation_config(serde_json::json!({
                "thinkingConfig": { "thinkingLevel": "high" }
            }))),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let config = vertex_request
            .generation_config()
            .expect("valid additional_params")
            .expect("generation_config is always Some");

        let thinking = config
            .thinking_config
            .expect("thinking_config should be set");
        assert_eq!(
            thinking.thinking_level,
            Some(vertexai::model::generation_config::thinking_config::ThinkingLevel::High)
        );
        assert_eq!(thinking.thinking_budget, None);
    }

    #[test]
    fn test_generation_config_thinking_budget_and_level_both_present_are_both_mapped() {
        // Mirrors the Gemini provider's own `ThinkingConfig`: the mutual-exclusion documented on
        // `thinking_budget`/`thinking_level` (Gemini 2.5 vs Gemini 3) is a REQUEST-VALIDITY concern
        // the provider API enforces, not something this conversion layer validates locally — same
        // as the Gemini provider itself, which maps both through unconditionally. Setting both
        // here proves the conversion does not silently drop either field.
        let request = CompletionRequest {
            model: None,
            additional_params: Some(additional_params_generation_config(serde_json::json!({
                "thinkingConfig": { "thinkingBudget": 512, "thinkingLevel": "low" }
            }))),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let config = vertex_request
            .generation_config()
            .expect("valid additional_params")
            .expect("generation_config is always Some");

        let thinking = config
            .thinking_config
            .expect("thinking_config should be set");
        assert_eq!(thinking.thinking_budget, Some(512));
        assert_eq!(
            thinking.thinking_level,
            Some(vertexai::model::generation_config::thinking_config::ThinkingLevel::Low)
        );
    }

    #[test]
    fn test_typed_temperature_and_max_tokens_win_over_additional_params_duplicates() {
        let request = CompletionRequest {
            model: None,
            temperature: Some(0.2),
            max_tokens: Some(50),
            additional_params: Some(additional_params_generation_config(serde_json::json!({
                "temperature": 0.9,
                "maxOutputTokens": 999
            }))),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let config = vertex_request
            .generation_config()
            .expect("valid additional_params")
            .expect("generation_config is always Some");

        assert_eq!(
            config.temperature,
            Some(0.2),
            "the typed request field must win over the additional_params duplicate"
        );
        assert_eq!(
            config.max_output_tokens,
            Some(50),
            "the typed request field must win over the additional_params duplicate"
        );
    }

    #[test]
    fn test_generation_config_candidate_count_from_additional_params_is_honored() {
        let request = CompletionRequest {
            model: None,
            additional_params: Some(additional_params_generation_config(serde_json::json!({
                "candidateCount": 2
            }))),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let config = vertex_request
            .generation_config()
            .expect("valid additional_params")
            .expect("generation_config is always Some");

        assert_eq!(
            config.candidate_count,
            Some(2),
            "an explicit candidateCount in additional_params must be honored, not overridden by \
             the default of 1"
        );
    }

    #[test]
    fn test_generation_config_invalid_additional_params_is_an_error() {
        let request = CompletionRequest {
            model: None,
            additional_params: Some(serde_json::json!({
                "generationConfig": { "thinkingConfig": { "thinkingLevel": "not-a-real-level" } }
            })),
            ..minimal_request()
        };

        let vertex_request = VertexCompletionRequest(request);
        let result = vertex_request.generation_config();
        assert!(
            result.is_err(),
            "an unrecognized thinkingLevel value must fail to deserialize, never be silently \
             dropped"
        );
    }
}
