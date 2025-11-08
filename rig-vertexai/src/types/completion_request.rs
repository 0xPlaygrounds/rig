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
