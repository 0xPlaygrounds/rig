//! Common imports for Rig's classic agent runtime.

pub use crate::agent::{Agent, MultiTurnStreamItem, StreamingResult};
pub use crate::client::{AgentClientExt, AgentModelExt};
pub use crate::completion::{
    Chat, CompletionError, CompletionModel, Prompt, PromptError, StructuredOutputError, TypedPrompt,
};
pub use crate::streaming::{StreamingChat, StreamingPrompt};
pub use crate::tool::{ContextualTool, ToolSet};
pub use rig_core::tool::Tool;
