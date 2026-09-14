use crate::types::message::RigMessage;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::CompletionError;
use rig_core::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig as GeminiGenerationConfig,
    ImageConfig as GeminiImageConfig, ResponseModality, ThinkingConfig as GeminiThinkingConfig,
    ThinkingLevel,
};

pub struct VertexCompletionRequest(pub rig_core::completion::CompletionRequest);

impl VertexCompletionRequest {
    pub fn contents(self) -> Result<Vec<vertexai::model::Content>, CompletionError> {
        // Vertex's `functionResponse.name` is the *function name*, not a
        // call identifier — `ToolResult::name` carries it as required data.
        // Consumes the request: this is the one accessor that needs the
        // history by value, so call it after the borrowing accessors.
        let mut history: Vec<rig_core::completion::Message> = self.0.chat_history;
        // Cross-provider ingested results arrive with an empty name and
        // their paired call carries it.
        rig_core::providers::internal::resolve_empty_tool_result_names(&mut history);

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
    ) -> Result<Option<vertexai::model::GenerationConfig>, CompletionError> {
        let AdditionalParameters {
            generation_config, ..
        } = match self.0.additional_params.as_ref() {
            Some(params) => serde::Deserialize::deserialize(params)?,
            None => AdditionalParameters::default(),
        };

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
    modality: &ResponseModality,
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
mod tests;
