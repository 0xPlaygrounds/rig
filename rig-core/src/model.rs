use crate::completion::{
    Completion, CompletionError, CompletionModel, CompletionRequestBuilder, CompletionResponse, Message, ModelChoice, Prompt, PromptError
};

/// A model that can be used to prompt completions from a completion model.
/// This is the simplest building block for creating an LLM powered application.
pub struct Model<M: CompletionModel> {
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    model: M,
    /// Temperature of the model
    temperature: Option<f64>,
}

impl<M: CompletionModel> Completion<M> for Model<M> {
    async fn completion(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        Ok(self
            .model
            .completion_request(prompt)
            .messages(chat_history)
            .temperature_opt(self.temperature))
    }
}

impl<M: CompletionModel> Prompt for Model<M> {
    async fn prompt(&self, prompt: &str, chat_history: Vec<Message>) -> Result<String, PromptError> {
        match self.completion(prompt, chat_history).await?.send().await? {
            CompletionResponse {
                choice: ModelChoice::Message(message),
                ..
            } => Ok(message),
            CompletionResponse {
                choice: ModelChoice::ToolCall(_, _),
                ..
            } => Err(PromptError::ToolCallError(
                "Tool calls are not supported by simple models in prompt mode".to_string(),
            )),
        }
    }
}

pub struct ModelBuilder<M: CompletionModel> {
    model: M,
    pub temperature: Option<f64>,
}

impl<M: CompletionModel> ModelBuilder<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            temperature: None,
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn build(self) -> Model<M> {
        Model {
            model: self.model,
            temperature: self.temperature,
        }
    }
}
