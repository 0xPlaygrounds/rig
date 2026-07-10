use crate::types::message::RigMessage;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::CompletionError;
use rig_core::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig as GeminiGenerationConfig,
    ImageConfig as GeminiImageConfig, ResponseModality, ThinkingConfig as GeminiThinkingConfig,
    ThinkingLevel,
};
use serde_json::{Map, Value};

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

    pub fn generation_config(
        &self,
    ) -> Result<Option<vertexai::model::GenerationConfig>, CompletionError> {
        let additional_params = self
            .0
            .additional_params
            .clone()
            .unwrap_or_else(|| Value::Object(Map::new()));
        let AdditionalParameters {
            generation_config, ..
        } = serde_json::from_value(additional_params)?;

        let mut config = generation_config
            .map(|mut config| {
                // Typed max_tokens is authoritative, so an overridden provider value must not be
                // converted or range-validated before the typed value is applied below.
                if self.0.max_tokens.is_some() {
                    config.max_output_tokens = None;
                }
                // Typed temperature is likewise authoritative, so it must bypass provider
                // conversion and range validation before being applied below.
                if self.0.temperature.is_some() {
                    config.temperature = None;
                }
                vertex_generation_config(config)
            })
            .transpose()?
            .unwrap_or_else(vertexai::model::GenerationConfig::new);

        // The typed request surface is authoritative over provider-specific extras.
        if let Some(temperature) = self.0.temperature {
            config = config.set_temperature(vertex_f32(temperature, "temperature")?);
        }

        if let Some(max_tokens) = self.0.max_tokens {
            config = config.set_max_output_tokens(vertex_max_output_tokens(max_tokens)?);
        }

        // Rig's normalized response retains one candidate, so the request must not
        // ask Vertex to generate candidates that would be silently discarded.
        config = config.set_candidate_count(1);

        Ok(Some(config))
    }
}

fn vertex_max_output_tokens(max_output_tokens: u64) -> Result<i32, CompletionError> {
    i32::try_from(max_output_tokens).map_err(|_| {
        CompletionError::RequestError("max_output_tokens exceeds Vertex AI's i32 range".into())
    })
}

fn vertex_f32(value: f64, field: &str) -> Result<f32, CompletionError> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(CompletionError::RequestError(
            format!("{field} must be finite and within Vertex AI's f32 range").into(),
        ));
    }

    Ok(value as f32)
}

fn vertex_generation_config(
    config: GeminiGenerationConfig,
) -> Result<vertexai::model::GenerationConfig, CompletionError> {
    if config.response_schema.is_some()
        && (config.response_json_schema.is_some() || config._response_json_schema.is_some())
    {
        return Err(CompletionError::RequestError(
            "responseSchema cannot be combined with responseJsonSchema or _responseJsonSchema"
                .into(),
        ));
    }
    if config.response_json_schema.is_some() && config._response_json_schema.is_some() {
        return Err(CompletionError::RequestError(
            "responseJsonSchema cannot be combined with _responseJsonSchema".into(),
        ));
    }

    let mut vertex_config = vertexai::model::GenerationConfig::new();

    if let Some(stop_sequences) = config.stop_sequences {
        vertex_config = vertex_config.set_stop_sequences(stop_sequences);
    }
    if let Some(response_mime_type) = config.response_mime_type {
        vertex_config = vertex_config.set_response_mime_type(response_mime_type);
    }
    if let Some(response_schema) = config.response_schema {
        vertex_config.response_schema = Some(serde_json::from_value(serde_json::to_value(
            response_schema,
        )?)?);
    }
    if let Some(response_json_schema) = config.response_json_schema.or(config._response_json_schema)
    {
        vertex_config.response_json_schema = Some(serde_json::from_value(response_json_schema)?);
    }
    if let Some(max_output_tokens) = config.max_output_tokens {
        vertex_config =
            vertex_config.set_max_output_tokens(vertex_max_output_tokens(max_output_tokens)?);
    }
    if let Some(temperature) = config.temperature {
        vertex_config = vertex_config.set_temperature(vertex_f32(temperature, "temperature")?);
    }
    if let Some(top_p) = config.top_p {
        vertex_config = vertex_config.set_top_p(vertex_f32(top_p, "top_p")?);
    }
    if let Some(top_k) = config.top_k {
        vertex_config = vertex_config.set_top_k(vertex_f32(f64::from(top_k), "top_k")?);
    }
    if let Some(presence_penalty) = config.presence_penalty {
        vertex_config =
            vertex_config.set_presence_penalty(vertex_f32(presence_penalty, "presence_penalty")?);
    }
    if let Some(frequency_penalty) = config.frequency_penalty {
        vertex_config = vertex_config
            .set_frequency_penalty(vertex_f32(frequency_penalty, "frequency_penalty")?);
    }
    if let Some(response_logprobs) = config.response_logprobs {
        vertex_config = vertex_config.set_response_logprobs(response_logprobs);
    }
    if let Some(logprobs) = config.logprobs {
        vertex_config = vertex_config.set_logprobs(logprobs);
    }
    if let Some(thinking_config) = config.thinking_config {
        vertex_config = vertex_config.set_thinking_config(vertex_thinking_config(thinking_config)?);
    }
    if let Some(response_modalities) = config.response_modalities {
        let response_modalities = response_modalities
            .into_iter()
            .map(vertex_response_modality)
            .collect::<Result<Vec<_>, _>>()?;
        vertex_config = vertex_config.set_response_modalities(response_modalities);
    }
    if let Some(image_config) = config.image_config {
        vertex_config = vertex_config.set_image_config(vertex_image_config(image_config));
    }

    Ok(vertex_config)
}

fn vertex_thinking_config(
    config: GeminiThinkingConfig,
) -> Result<vertexai::model::generation_config::ThinkingConfig, CompletionError> {
    if config.thinking_budget.is_some() && config.thinking_level.is_some() {
        return Err(CompletionError::RequestError(
            "thinking_budget and thinking_level cannot both be set".into(),
        ));
    }

    let mut vertex_config = vertexai::model::generation_config::ThinkingConfig::new();
    if let Some(include_thoughts) = config.include_thoughts {
        vertex_config = vertex_config.set_include_thoughts(include_thoughts);
    }
    if let Some(thinking_budget) = config.thinking_budget {
        let thinking_budget = i32::try_from(thinking_budget).map_err(|_| {
            CompletionError::RequestError("thinking_budget exceeds Vertex AI's i32 range".into())
        })?;
        vertex_config = vertex_config.set_thinking_budget(thinking_budget);
    }
    if let Some(thinking_level) = config.thinking_level {
        vertex_config = vertex_config.set_thinking_level(match thinking_level {
            ThinkingLevel::Minimal => {
                vertexai::model::generation_config::thinking_config::ThinkingLevel::Minimal
            }
            ThinkingLevel::Low => {
                vertexai::model::generation_config::thinking_config::ThinkingLevel::Low
            }
            ThinkingLevel::Medium => {
                vertexai::model::generation_config::thinking_config::ThinkingLevel::Medium
            }
            ThinkingLevel::High => {
                vertexai::model::generation_config::thinking_config::ThinkingLevel::High
            }
        });
    }

    Ok(vertex_config)
}

fn vertex_response_modality(
    modality: ResponseModality,
) -> Result<vertexai::model::generation_config::Modality, CompletionError> {
    match modality {
        ResponseModality::Text => Ok(vertexai::model::generation_config::Modality::Text),
        ResponseModality::Image => Ok(vertexai::model::generation_config::Modality::Image),
        ResponseModality::Audio => Err(CompletionError::RequestError(
            "responseModalities AUDIO is unsupported because Rig cannot represent assistant audio responses"
                .into(),
        )),
    }
}

fn vertex_image_config(image_config: GeminiImageConfig) -> vertexai::model::ImageConfig {
    let mut vertex_config = vertexai::model::ImageConfig::new();
    if let Some(aspect_ratio) = image_config.aspect_ratio {
        vertex_config = vertex_config.set_aspect_ratio(aspect_ratio);
    }
    if let Some(image_size) = image_config.image_size {
        vertex_config = vertex_config.set_image_size(image_size);
    }
    vertex_config
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
            .expect("generation config should parse");

        assert!(generation_config.is_some());
        let config = generation_config.unwrap();
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_output_tokens, Some(100));
        assert_eq!(config.candidate_count, Some(1));
    }

    #[test]
    fn generation_config_maps_thinking_budget_and_include_thoughts() {
        let request = CompletionRequest {
            additional_params: Some(serde_json::json!({
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingBudget": 1024,
                        "includeThoughts": true
                    }
                }
            })),
            ..minimal_request()
        };

        let config = VertexCompletionRequest(request)
            .generation_config()
            .expect("generation config should parse")
            .expect("generation config should exist");
        let thinking = config
            .thinking_config
            .expect("thinking config should be mapped");

        assert_eq!(thinking.thinking_budget, Some(1024));
        assert_eq!(thinking.include_thoughts, Some(true));
        assert_eq!(thinking.thinking_level, None);
    }

    #[test]
    fn generation_config_maps_thinking_levels() {
        use vertexai::model::generation_config::thinking_config::ThinkingLevel as VertexThinkingLevel;

        for (level, expected) in [
            ("minimal", VertexThinkingLevel::Minimal),
            ("low", VertexThinkingLevel::Low),
            ("medium", VertexThinkingLevel::Medium),
            ("high", VertexThinkingLevel::High),
        ] {
            let request = CompletionRequest {
                additional_params: Some(serde_json::json!({
                    "generationConfig": {
                        "thinkingConfig": { "thinkingLevel": level }
                    }
                })),
                ..minimal_request()
            };

            let config = VertexCompletionRequest(request)
                .generation_config()
                .expect("generation config should parse")
                .expect("generation config should exist");
            let thinking = config
                .thinking_config
                .expect("thinking config should be mapped");

            assert_eq!(thinking.thinking_level, Some(expected));
        }
    }

    #[test]
    fn generation_config_maps_supported_fields_and_typed_fields_take_precedence() {
        let request = CompletionRequest {
            temperature: Some(0.7),
            max_tokens: Some(100),
            additional_params: Some(serde_json::json!({
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 20,
                    "candidateCount": 2,
                    "stopSequences": ["END"],
                    "responseMimeType": "application/json",
                    "responseJsonSchema": { "type": "object" },
                    "topP": 0.8,
                    "topK": 40,
                    "presencePenalty": 0.1,
                    "frequencyPenalty": 0.2,
                    "responseLogprobs": true,
                    "logprobs": 3,
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": "16:9",
                        "imageSize": "1K"
                    }
                }
            })),
            ..minimal_request()
        };

        let config = VertexCompletionRequest(request)
            .generation_config()
            .expect("generation config should parse")
            .expect("generation config should exist");

        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_output_tokens, Some(100));
        assert_eq!(config.candidate_count, Some(1));
        assert_eq!(config.stop_sequences, ["END"]);
        assert_eq!(config.response_mime_type, "application/json");
        assert_eq!(
            serde_json::to_value(
                config
                    .response_json_schema
                    .expect("JSON schema should be mapped")
            )
            .expect("Vertex JSON schema should serialize"),
            serde_json::json!({ "type": "object" })
        );
        assert_eq!(config.top_p, Some(0.8));
        assert_eq!(config.top_k, Some(40.0));
        assert_eq!(config.presence_penalty, Some(0.1));
        assert_eq!(config.frequency_penalty, Some(0.2));
        assert_eq!(config.response_logprobs, Some(true));
        assert_eq!(config.logprobs, Some(3));
        assert_eq!(
            config.response_modalities,
            [
                vertexai::model::generation_config::Modality::Text,
                vertexai::model::generation_config::Modality::Image,
            ]
        );
        let image = config.image_config.expect("image config should be mapped");
        assert_eq!(image.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(image.image_size.as_deref(), Some("1K"));
    }

    #[test]
    fn generation_config_rejects_audio_response_modality() {
        let request = CompletionRequest {
            additional_params: Some(serde_json::json!({
                "generationConfig": { "responseModalities": ["AUDIO"] }
            })),
            ..minimal_request()
        };

        let error = VertexCompletionRequest(request)
            .generation_config()
            .expect_err("audio output must fail before the API call");
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert!(error.to_string().contains("responseModalities AUDIO"));
    }

    #[test]
    fn generation_config_rejects_out_of_f32_range_typed_and_provider_values() {
        let typed_request = CompletionRequest {
            temperature: Some(f64::MAX),
            ..minimal_request()
        };
        let typed_error = VertexCompletionRequest(typed_request)
            .generation_config()
            .expect_err("typed temperature beyond f32 must fail");
        assert!(matches!(typed_error, CompletionError::RequestError(_)));
        assert!(typed_error.to_string().contains("temperature"));

        let provider_error = vertex_generation_config(GeminiGenerationConfig {
            top_p: Some(f64::MAX),
            ..Default::default()
        })
        .expect_err("provider top_p beyond f32 must fail");
        assert!(matches!(provider_error, CompletionError::RequestError(_)));
        assert!(provider_error.to_string().contains("top_p"));
    }

    #[test]
    fn generation_config_rejects_non_finite_typed_values() {
        let request = CompletionRequest {
            temperature: Some(f64::NAN),
            ..minimal_request()
        };

        let error = VertexCompletionRequest(request)
            .generation_config()
            .expect_err("non-finite typed temperature must fail");
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert!(error.to_string().contains("temperature"));
    }

    #[test]
    fn generation_config_maps_response_schema() {
        let request = CompletionRequest {
            additional_params: Some(serde_json::json!({
                "generationConfig": {
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "name": { "type": "STRING" }
                        }
                    }
                }
            })),
            ..minimal_request()
        };

        let config = VertexCompletionRequest(request)
            .generation_config()
            .expect("generation config should parse")
            .expect("generation config should exist");

        let schema = config
            .response_schema
            .expect("response schema should be mapped");
        assert_eq!(schema.r#type, vertexai::model::Type::Object);
        assert_eq!(
            schema
                .properties
                .get("name")
                .map(|property| &property.r#type),
            Some(&vertexai::model::Type::String)
        );
    }

    #[test]
    fn generation_config_typed_max_overrides_out_of_range_provider_max() {
        let request = CompletionRequest {
            max_tokens: Some(100),
            additional_params: Some(serde_json::json!({
                "generationConfig": { "maxOutputTokens": 2_147_483_648_u64 }
            })),
            ..minimal_request()
        };

        let config = VertexCompletionRequest(request)
            .generation_config()
            .expect("typed max should override an unrepresentable provider max")
            .expect("generation config should exist");
        assert_eq!(config.max_output_tokens, Some(100));
    }

    #[test]
    fn generation_config_typed_temperature_overrides_out_of_range_provider_temperature() {
        let request = CompletionRequest {
            temperature: Some(0.7),
            additional_params: Some(serde_json::json!({
                "generationConfig": { "temperature": f64::MAX }
            })),
            ..minimal_request()
        };

        let config = VertexCompletionRequest(request)
            .generation_config()
            .expect("typed temperature should override an unrepresentable provider temperature")
            .expect("generation config should exist");
        assert_eq!(config.temperature, Some(0.7));
    }

    #[test]
    fn generation_config_rejects_out_of_range_max_output_tokens_without_typed_override() {
        let provider_request = CompletionRequest {
            additional_params: Some(serde_json::json!({
                "generationConfig": { "maxOutputTokens": 2_147_483_648_u64 }
            })),
            ..minimal_request()
        };
        let provider_error = VertexCompletionRequest(provider_request)
            .generation_config()
            .expect_err("provider max output tokens beyond i32 must fail");
        assert!(
            provider_error
                .to_string()
                .contains("max_output_tokens exceeds Vertex AI's i32 range")
        );

        let typed_request = CompletionRequest {
            max_tokens: Some(2_147_483_648),
            ..minimal_request()
        };
        let typed_error = VertexCompletionRequest(typed_request)
            .generation_config()
            .expect_err("typed max output tokens beyond i32 must fail");
        assert!(
            typed_error
                .to_string()
                .contains("max_output_tokens exceeds Vertex AI's i32 range")
        );
    }

    #[test]
    fn generation_config_rejects_conflicting_response_schema_forms() {
        for json_schema_key in ["responseJsonSchema", "_responseJsonSchema"] {
            let request = CompletionRequest {
                additional_params: Some(serde_json::json!({
                    "generationConfig": {
                        "responseSchema": { "type": "OBJECT" },
                        (json_schema_key): { "type": "object" }
                    }
                })),
                ..minimal_request()
            };

            let error = VertexCompletionRequest(request)
                .generation_config()
                .expect_err("response schema forms cannot be combined");
            assert!(matches!(error, CompletionError::RequestError(_)));
            assert!(
                error
                    .to_string()
                    .contains("responseSchema cannot be combined")
            );
        }
    }

    #[test]
    fn generation_config_rejects_both_response_json_schema_aliases() {
        let request = CompletionRequest {
            additional_params: Some(serde_json::json!({
                "generationConfig": {
                    "responseJsonSchema": { "type": "object" },
                    "_responseJsonSchema": { "type": "object" }
                }
            })),
            ..minimal_request()
        };

        let error = VertexCompletionRequest(request)
            .generation_config()
            .expect_err("JSON schema aliases cannot be combined");
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert!(
            error
                .to_string()
                .contains("responseJsonSchema cannot be combined with _responseJsonSchema")
        );
    }

    #[test]
    fn generation_config_rejects_invalid_thinking_config() {
        let malformed_request = CompletionRequest {
            additional_params: Some(serde_json::json!({
                "generationConfig": { "thinkingConfig": { "thinkingBudget": "invalid" } }
            })),
            ..minimal_request()
        };
        assert!(
            VertexCompletionRequest(malformed_request)
                .generation_config()
                .is_err()
        );

        let conflicting_request = CompletionRequest {
            additional_params: Some(serde_json::json!({
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingBudget": 1024,
                        "thinkingLevel": "high"
                    }
                }
            })),
            ..minimal_request()
        };
        let error = VertexCompletionRequest(conflicting_request)
            .generation_config()
            .expect_err("mutually exclusive thinking values must fail");
        assert!(
            error
                .to_string()
                .contains("thinking_budget and thinking_level cannot both be set")
        );
    }
}
