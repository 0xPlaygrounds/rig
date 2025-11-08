use crate::types::message::RigMessage;
use google_cloud_aiplatform_v1 as vertexai;
use rig::completion::CompletionError;

pub struct VertexCompletionRequest(pub rig::completion::CompletionRequest);

impl VertexCompletionRequest {
    pub fn contents(&self) -> Result<Vec<vertexai::model::Content>, CompletionError> {
        let mut contents = Vec::new();

        // Convert chat history to Vertex AI Content format
        for message in self.0.chat_history.iter() {
            let content = RigMessage(message.clone()).try_into()?;
            contents.push(content);
        }

        Ok(contents)
    }

    pub fn system_instruction(&self) -> Option<vertexai::model::Content> {
        self.0.preamble.as_ref().map(|preamble| {
            // System instructions don't need a role - the API handles them separately
            // But Content requires a role, so we use "user" (it's ignored for system_instruction)
            vertexai::model::Content::new()
                .set_role("user")
                .set_parts([vertexai::model::Part::new().set_text(preamble.clone())])
        })
    }

    pub fn generation_config(&self) -> Option<vertexai::model::GenerationConfig> {
        let mut config = vertexai::model::GenerationConfig::new();

        if let Some(temperature) = self.0.temperature {
            config = config.set_temperature(temperature as f32);
        }

        if let Some(max_tokens) = self.0.max_tokens {
            config = config.set_max_output_tokens(max_tokens as i32);
        }

        // Set candidate_count to 1 by default (Vertex AI can return multiple candidates)
        config = config.set_candidate_count(1);

        Some(config)
    }
}
