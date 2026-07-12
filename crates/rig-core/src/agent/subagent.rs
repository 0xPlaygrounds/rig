//! Bounded child-agent orchestration built on the shared agent runner.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    completion::{CompletionModel, PromptError},
    tool::{Tool, ToolCallExtensions, ToolErrorReport, ToolFailure},
};

use super::{Agent, RunControlHandle, ToolCallContext};

/// Whether child tools inherit caller-provided runtime extensions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SubagentContextMode {
    /// Start with no caller extensions.
    Fresh,
    /// Preserve caller extensions and call ancestry.
    #[default]
    Inherited,
}

/// Typed handoff returned by a child agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentHandoff {
    /// Child's final model-visible output.
    pub output: String,
    /// Child run identity for trace/session correlation.
    pub run_id: String,
}

/// Subagent invocation arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentArgs {
    /// Task delegated to the child.
    pub prompt: String,
}

/// Child orchestration failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SubagentError {
    /// Configured ancestry depth was exceeded.
    #[error("subagent depth {depth} exceeds maximum {max_depth}")]
    DepthExceeded { depth: usize, max_depth: usize },
    /// Parent run was cancelled.
    #[error("subagent cancelled with parent: {0}")]
    Cancelled(String),
    /// Child runner failed.
    #[error(transparent)]
    Prompt(#[from] PromptError),
}

/// An agent exposed as a bounded, cancellation-aware child tool.
#[derive(Clone)]
pub struct Subagent<M>
where
    M: CompletionModel,
{
    agent: Agent<M>,
    max_depth: usize,
    max_turns: usize,
    context_mode: SubagentContextMode,
}

impl<M> Subagent<M>
where
    M: CompletionModel,
{
    /// Wrap an agent with conservative depth and turn budgets.
    pub fn new(agent: Agent<M>) -> Self {
        Self {
            agent,
            max_depth: 4,
            max_turns: 8,
            context_mode: SubagentContextMode::Inherited,
        }
    }

    /// Set the maximum parent/child call depth.
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the child model-call budget.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Choose fresh or inherited caller extensions.
    pub fn context_mode(mut self, mode: SubagentContextMode) -> Self {
        self.context_mode = mode;
        self
    }
}

impl<M> Tool for Subagent<M>
where
    M: CompletionModel + 'static,
{
    const NAME: &'static str = "subagent";

    type Error = SubagentError;
    type Args = SubagentArgs;
    type Output = SubagentHandoff;

    fn name(&self) -> String {
        self.agent
            .name
            .clone()
            .unwrap_or_else(|| Self::NAME.to_owned())
    }

    fn description(&self) -> String {
        self.agent
            .description
            .clone()
            .unwrap_or_else(|| "Delegate a bounded task to a child agent".to_owned())
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "prompt": { "type": "string" } },
            "required": ["prompt"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_with_extensions(args, &ToolCallExtensions::new())
            .await
    }

    async fn call_with_extensions(
        &self,
        args: Self::Args,
        extensions: &ToolCallExtensions,
    ) -> Result<Self::Output, Self::Error> {
        let parent = extensions.get::<ToolCallContext>().cloned();
        let depth = parent
            .as_ref()
            .map_or(0, |call| call.depth.saturating_add(1));
        if depth > self.max_depth {
            return Err(SubagentError::DepthExceeded {
                depth,
                max_depth: self.max_depth,
            });
        }

        let mut runner = self.agent.runner(args.prompt).max_turns(self.max_turns);
        if self.context_mode == SubagentContextMode::Inherited {
            runner = runner.tool_extensions(extensions.clone());
        }
        let child_control: RunControlHandle = runner.control_handle();
        let child_run_id = child_control.run_id().to_string();

        let response = if let Some(parent) = parent {
            let child_run = runner.run();
            tokio::pin!(child_run);
            tokio::select! {
                response = &mut child_run => response.map_err(SubagentError::from)?,
                reason = parent.run.control().cancelled() => {
                    child_control.cancel();
                    // Drive cancellation through the shared loop so run-scoped
                    // factories and in-flight provider/tool futures settle.
                    let _ = child_run.await;
                    return Err(SubagentError::Cancelled(reason.to_owned()));
                }
            }
        } else {
            runner.run().await?
        };

        Ok(SubagentHandoff {
            output: response.output,
            run_id: child_run_id,
        })
    }

    fn classify_error_report(&self, error: &Self::Error) -> ToolErrorReport {
        let failure = match error {
            SubagentError::DepthExceeded { .. } => {
                ToolFailure::permission_denied(error.to_string())
            }
            SubagentError::Cancelled(_) => ToolFailure::cancelled(error.to_string()),
            SubagentError::Prompt(_) => ToolFailure::other(error.to_string()),
        };
        ToolErrorReport::new(failure)
    }
}
