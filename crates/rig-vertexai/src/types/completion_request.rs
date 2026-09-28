use crate::types::message::RigMessage;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::error::ProviderError;
use rig_core::providers::gemini::api::{
    GenerationConfig as GeminiGenerationConfig, ImageConfig as GeminiImageConfig,
    ResponseModalities as ResponseModality, ThinkingConfig as GeminiThinkingConfig, ThinkingLevel,
};

/// The `additional_params` Vertex reads: Gemini's `generationConfig`.
#[derive(Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct AdditionalParameters {
    #[serde(default)]
    generation_config: Option<GeminiGenerationConfig>,
}

pub struct VertexCompletionRequest(pub rig_core::completion::CompletionRequest);

impl VertexCompletionRequest {
    pub fn contents(self) -> Result<Vec<vertexai::model::Content>, ProviderError> {
        let history = self.0.chat_history;
        let mut contents = Vec::new();
        for message in history {
            if matches!(message, rig_core::completion::Message::System { .. }) {
                continue;
            }
            let content = RigMessage(message).try_into()?;
            contents.push(content);
        }

        Ok(contents)
    }

    pub fn system_instruction(&self) -> Option<vertexai::model::Content> {
        let mut system_texts = Vec::new();
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

        use vertexai::model::function_calling_config::Mode;

        let (mode, allowed_function_names) = match self.0.tool_choice.as_ref() {
            Some(rig_core::message::ToolChoice::Auto) | None => (Mode::Auto, Vec::new()),
            Some(rig_core::message::ToolChoice::Required) => (Mode::Any, Vec::new()),
            Some(rig_core::message::ToolChoice::None) => (Mode::None, Vec::new()),
            Some(rig_core::message::ToolChoice::Specific { function_names }) => {
                (Mode::Any, function_names.clone())
            }
        };

        let function_calling_config = vertexai::model::FunctionCallingConfig::new()
            .set_mode(mode)
            .set_allowed_function_names(allowed_function_names);

        Some(
            vertexai::model::ToolConfig::new().set_function_calling_config(function_calling_config),
        )
    }

    pub fn generation_config(
        &self,
    ) -> Result<Option<vertexai::model::GenerationConfig>, ProviderError> {
        let mut params = self
            .0
            .additional_params
            .clone()
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
        if let Some(config) = params
            .get_mut("generationConfig")
            .and_then(serde_json::Value::as_object_mut)
        {
            // Typed values are authoritative, so an overridden provider value
            // is neither converted nor range-checked.
            if self.0.max_tokens.is_some() {
                config.remove("maxOutputTokens");
            }
            if self.0.temperature.is_some() {
                config.remove("temperature");
            }
            if let Some(max) = config
                .get("maxOutputTokens")
                .and_then(serde_json::Value::as_u64)
            {
                vertex_max_output_tokens(max)?;
            }
        }
        let AdditionalParameters { generation_config } = serde_json::from_value(params)?;

        let mut config = generation_config
            .map(vertex_generation_config)
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

fn vertex_max_output_tokens(max_output_tokens: u64) -> Result<i32, ProviderError> {
    i32::try_from(max_output_tokens)
        .map_err(|_| ProviderError::request("max_output_tokens exceeds Vertex AI's i32 range"))
}

fn vertex_f32(value: f64, field: &str) -> Result<f32, ProviderError> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(ProviderError::request(format!(
            "{field} must be finite and within Vertex AI's f32 range"
        )));
    }

    Ok(value as f32)
}

fn vertex_generation_config(
    config: GeminiGenerationConfig,
) -> Result<vertexai::model::GenerationConfig, ProviderError> {
    if config.response_schema.is_some()
        && (config.response_json_schema.is_some() || config.legacy_response_json_schema.is_some())
    {
        return Err(ProviderError::request(
            "responseSchema cannot be combined with responseJsonSchema or _responseJsonSchema",
        ));
    }
    if config.response_json_schema.is_some() && config.legacy_response_json_schema.is_some() {
        return Err(ProviderError::request(
            "responseJsonSchema cannot be combined with _responseJsonSchema",
        ));
    }

    let mut vertex_config = vertexai::model::GenerationConfig::new();

    if !config.stop_sequences.is_empty() {
        vertex_config = vertex_config.set_stop_sequences(config.stop_sequences);
    }
    if let Some(response_mime_type) = config.response_mime_type {
        vertex_config = vertex_config.set_response_mime_type(response_mime_type);
    }
    if let Some(response_schema) = config.response_schema {
        vertex_config.response_schema = Some(serde_json::from_value(serde_json::to_value(
            response_schema,
        )?)?);
    }
    if let Some(response_json_schema) = config
        .response_json_schema
        .or(config.legacy_response_json_schema)
    {
        vertex_config.response_json_schema = Some(serde_json::from_value(response_json_schema)?);
    }
    if let Some(max_output_tokens) = config.max_output_tokens {
        vertex_config = vertex_config.set_max_output_tokens(max_output_tokens);
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
    if !config.response_modalities.is_empty() {
        let response_modalities = config
            .response_modalities
            .into_iter()
            .map(|modality| vertex_response_modality(&modality))
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
) -> Result<vertexai::model::generation_config::ThinkingConfig, ProviderError> {
    if config.thinking_budget.is_some() && config.thinking_level.is_some() {
        return Err(ProviderError::request(
            "thinking_budget and thinking_level cannot both be set",
        ));
    }

    let mut vertex_config = vertexai::model::generation_config::ThinkingConfig::new();
    if let Some(include_thoughts) = config.include_thoughts {
        vertex_config = vertex_config.set_include_thoughts(include_thoughts);
    }
    if let Some(thinking_budget) = config.thinking_budget {
        vertex_config = vertex_config.set_thinking_budget(thinking_budget);
    }
    if let Some(thinking_level) = config.thinking_level {
        use vertexai::model::generation_config::thinking_config::ThinkingLevel as Vertex;
        // Google reads the level in either case.
        let level = match thinking_level {
            ThinkingLevel::Unknown(level) => level.to_ascii_uppercase(),
            level => level.as_str().to_owned(),
        };
        vertex_config = vertex_config.set_thinking_level(match level.as_str() {
            "MINIMAL" => Vertex::Minimal,
            "LOW" => Vertex::Low,
            "MEDIUM" => Vertex::Medium,
            "HIGH" => Vertex::High,
            _ => {
                return Err(ProviderError::request(
                    "thinking_level must be minimal, low, medium or high",
                ));
            }
        });
    }

    Ok(vertex_config)
}

fn vertex_response_modality(
    modality: &ResponseModality,
) -> Result<vertexai::model::generation_config::Modality, ProviderError> {
    match modality {
        ResponseModality::Text => Ok(vertexai::model::generation_config::Modality::Text),
        ResponseModality::Image => Ok(vertexai::model::generation_config::Modality::Image),
        ResponseModality::Audio => Err(ProviderError::request(
            "responseModalities AUDIO is unsupported because Rig cannot represent assistant audio responses",
        )),
        ResponseModality::ModalityUnspecified | ResponseModality::Unknown(_) => Err(
            ProviderError::request("responseModalities must be TEXT or IMAGE"),
        ),
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
mod tests;
