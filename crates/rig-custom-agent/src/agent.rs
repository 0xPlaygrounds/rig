use rig_core::{
    completion::{CompletionError, CompletionModel, CompletionRequest, Message, ToolDefinition},
    tool::{IntoToolOutput, PortableDynamicTool, ToolExecutionError, portable::PortableTool},
};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// Errors that can occur during the custom agent execution loop.
#[derive(Debug, thiserror::Error)]
pub enum CustomAgentError {
    /// An error returned by the underlying completion model.
    #[error("Completion error: {0}")]
    Completion(#[from] CompletionError),
    /// The maximum number of configured agent iterations was reached without yielding a final result.
    #[error("Max tool steps reached without final response")]
    MaxStepsReached,
}

/// A builder for configuring a [`CustomAgent`].
pub struct CustomAgentBuilder<M: CompletionModel + Clone> {
    model: M,
    tools: HashMap<String, PortableDynamicTool>,
    preamble: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
    documents: Vec<rig_core::completion::Document>,
    tool_choice: Option<rig_core::message::ToolChoice>,
    output_schema: Option<schemars::Schema>,
    record_telemetry_content: bool,
}

impl<M: CompletionModel + Clone> CustomAgentBuilder<M> {
    /// Initialize a new agent builder with the specified completion model.
    pub fn new(model: M) -> Self {
        Self {
            model,
            tools: HashMap::new(),
            preamble: None,
            temperature: None,
            max_tokens: None,
            additional_params: None,
            documents: Vec::new(),
            tool_choice: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    /// Add a portable tool to the agent's available toolset.
    pub fn tool<T: PortableTool + Send + Sync + 'static>(mut self, tool: T) -> Self {
        let tool = Arc::new(tool);
        let dynamic_tool = PortableDynamicTool::new(
            T::NAME,
            tool.description(),
            tool.parameters(),
            move |args| {
                let tool = tool.clone();
                Box::pin(async move {
                    match serde_json::from_value::<T::Args>(args) {
                        Ok(parsed) => match tool.call(parsed).await {
                            Ok(output) => output.into_tool_output(),
                            Err(e) => Err(ToolExecutionError::from_error(e)),
                        },
                        Err(e) => Err(ToolExecutionError::invalid_args(e.to_string())),
                    }
                })
            },
        );
        self.tools.insert(T::NAME.to_string(), dynamic_tool);
        self
    }

    /// Add a document for RAG context.
    pub fn document(mut self, document: rig_core::completion::Document) -> Self {
        self.documents.push(document);
        self
    }

    /// Set a system preamble (system prompt) for the agent.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Set the temperature for the completion model.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the max tokens for the completion model.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set additional parameters for the completion model.
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set the tool choice for the completion model.
    pub fn tool_choice(mut self, tool_choice: rig_core::message::ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set the output schema for the completion model.
    pub fn output_schema(mut self, schema: schemars::Schema) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Enable telemetry content recording for the completion model.
    pub fn record_telemetry_content(mut self, record: bool) -> Self {
        self.record_telemetry_content = record;
        self
    }

    /// Consume the builder and return the configured [`CustomAgent`].
    pub fn build(self) -> CustomAgent<M> {
        CustomAgent {
            model: self.model,
            tools: self.tools,
            preamble: self.preamble,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            documents: self.documents,
            tool_choice: self.tool_choice,
            output_schema: self.output_schema,
            record_telemetry_content: self.record_telemetry_content,
        }
    }
}

/// A custom orchestrator runtime for executing loops over completion models.
pub struct CustomAgent<M: CompletionModel + Clone> {
    model: M,
    tools: HashMap<String, PortableDynamicTool>,
    preamble: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
    documents: Vec<rig_core::completion::Document>,
    tool_choice: Option<rig_core::message::ToolChoice>,
    output_schema: Option<schemars::Schema>,
    record_telemetry_content: bool,
}

impl<M: CompletionModel + Clone> CustomAgent<M> {
    /// Execute a chat conversation with the agent, permitting up to `max_steps` tool invocations.
    #[instrument(skip_all, name = "custom_agent_chat", fields(user_prompt = %prompt))]
    pub async fn chat(&self, prompt: &str, max_steps: usize) -> Result<String, CustomAgentError> {
        self.chat_with_history(prompt, Vec::new(), max_steps).await
    }

    /// Execute a chat conversation with existing chat history, permitting up to `max_steps` tool invocations.
    #[instrument(skip_all, name = "custom_agent_chat_with_history", fields(user_prompt = %prompt))]
    pub async fn chat_with_history(
        &self,
        prompt: &str,
        mut chat_history: Vec<Message>,
        max_steps: usize,
    ) -> Result<String, CustomAgentError> {
        info!("Starting custom agent execution loop");

        chat_history.push(Message::User {
            content: vec![rig_core::message::UserContent::text(prompt)],
        });

        for step in 0..max_steps {
            debug!(step, "Preparing CompletionRequest");

            let mut tool_defs: Vec<ToolDefinition> = Vec::new();
            for tool in self.tools.values() {
                tool_defs.push(tool.definition());
            }

            let request = CompletionRequest {
                model: None,
                preamble: self.preamble.clone(),
                chat_history: chat_history.clone(),
                documents: self.documents.clone(),
                tools: tool_defs,
                temperature: self.temperature,
                max_tokens: self.max_tokens,
                tool_choice: self.tool_choice.clone(),
                additional_params: self.additional_params.clone(),
                output_schema: self.output_schema.clone(),
                record_telemetry_content: self.record_telemetry_content,
            };

            info!("Sending request to model");
            let response = self.model.clone().completion(request).await?;

            let choice = response.choice;

            let text = choice
                .iter()
                .filter_map(|c| {
                    if let rig_core::message::AssistantContent::Text(t) = c {
                        Some(t.text.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

            if !text.is_empty() {
                debug!(assistant_text = %text, "Model provided reasoning or response");
            }

            chat_history.push(Message::Assistant {
                id: None,
                content: choice.clone(),
            });

            let mut has_tool_calls = false;
            let mut tool_results = Vec::new();

            for content in choice {
                if let rig_core::message::AssistantContent::ToolCall(call) = content {
                    has_tool_calls = true;
                    info!(tool_name = %call.function.name, "Model requested a tool call");

                    let result = self
                        .execute_tool(&call.id, &call.function.name, &call.function.arguments)
                        .await;
                    tool_results.push(result);
                }
            }

            if !has_tool_calls {
                info!("Model returned final response");
                return Ok(text);
            }

            chat_history.push(Message::User {
                content: tool_results,
            });
        }

        warn!("Agent reached max steps without concluding");
        Err(CustomAgentError::MaxStepsReached)
    }

    #[instrument(skip(self), name = "execute_tool", fields(tool_name = %name))]
    async fn execute_tool(
        &self,
        call_id: &str,
        name: &str,
        args: &serde_json::Value,
    ) -> rig_core::message::UserContent {
        debug!(args = %args, "Executing tool");

        if let Some(tool) = self.tools.get(name) {
            match tool.execute(args.clone()).await {
                Ok(output) => {
                    debug!("Tool execution succeeded");
                    rig_core::message::UserContent::tool_result(
                        call_id,
                        name,
                        output.into_content(),
                    )
                }
                Err(e) => {
                    warn!(error = %e, "Tool execution failed");
                    rig_core::message::UserContent::tool_result(
                        call_id,
                        name,
                        vec![rig_core::message::ToolResultContent::text(format!(
                            "Error: {}",
                            e
                        ))],
                    )
                }
            }
        } else {
            warn!("Tool not found in registry");
            rig_core::message::UserContent::tool_result(
                call_id,
                name,
                vec![rig_core::message::ToolResultContent::text(format!(
                    "Error: Tool {} not found",
                    name
                ))],
            )
        }
    }
}
