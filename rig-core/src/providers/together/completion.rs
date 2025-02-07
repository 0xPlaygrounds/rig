// ================================================================
//! Together AI Completion Integration
//! From [Together AI Reference](https://docs.together.ai/docs/chat-overview)
// ================================================================

use crate::{
    completion::{self, CompletionError},
    json_utils,
    providers::openai,
};

use serde_json::json;

use super::client::{together_ai_api_types::ApiResponse, Client};

/// Example model name for Together AI, you should replace this with actual model names provided by Together AI.
pub const TOGETHER_MODEL: &str = "meta-llama/Meta-Llama-3-8B-Instruct-Turbo";

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Clone)]
pub struct CompletionModel {
    client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = openai::CompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        
        let mut full_history: Vec<openai::Message> = match &completion_request.preamble {
            Some(preamble) => vec![openai::Message::system(preamble)],
            None => vec![],
        };

        // Convert prompt to user message
        let prompt: Vec<openai::Message> = completion_request.prompt_with_context().try_into()?;

        // Convert existing chat history
        let chat_history: Vec<openai::Message> = completion_request
            .chat_history
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<openai::Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        // Combine all messages into a single history
        full_history.extend(chat_history);
        full_history.extend(prompt);

        let mut request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(openai::ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": "auto",
            })
        };

        request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };
        
        let response = self
            .client
            .post("/v1/chat/completions")
            .json(&request)
            .send()
            .await?;

            if response.status().is_success() {
                let t = response.text().await?;
                tracing::debug!(target: "rig", "Together completion error: {}", t);
    
    
                match serde_json::from_str::<ApiResponse<openai::CompletionResponse>>(&t)? {
                    ApiResponse::Ok(response) => {
                        tracing::info!(target: "rig",
                            "Together completion token usage: {:?}",
                            response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                        );
                        response.try_into()
                    }
                    ApiResponse::Error(err) => Err(CompletionError::ProviderError(err.error)),
                }
            } else {
                Err(CompletionError::ProviderError(response.text().await?))
            }
    }
}
