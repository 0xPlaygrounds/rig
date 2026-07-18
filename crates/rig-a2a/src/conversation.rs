//! Keep Rig memory and the remote A2A thread on the same conversation key.
//!
//! ```no_run
//! use rig_a2a::A2AConversationExt;
//!
//! # async fn run(client: rig_a2a::A2AClient) -> anyhow::Result<()> {
//! let agent = client.agent().build();
//! agent
//!     .runner("what did we decide?")
//!     .a2a_conversation("user-42")
//!     .run()
//!     .await?;
//! # Ok(()) }
//! ```

use rig_agent::agent::{
    AgentBuilder, AgentRunner, PromptRequest, PromptType, StreamingPromptRequest,
};
use serde_json::{Map, Value, json};

use crate::model::THREAD_PARAMS_KEY;

/// Configure both Rig conversation memory and A2A remote threading.
///
/// Rig does not pass its memory conversation ID to completion models. This
/// extension sets that ID normally and also sends it to
/// [`A2AModel`](crate::A2AModel) through provider-specific request parameters.
pub trait A2AConversationExt: Sized {
    /// Continue `id` in both Rig memory and the remote A2A agent.
    fn a2a_conversation(self, id: impl Into<String>) -> Self;
}

fn conversation_params(id: String) -> Map<String, Value> {
    let mut params = Map::new();
    params.insert(THREAD_PARAMS_KEY.to_string(), json!({ "conversation": id }));
    params
}

impl<ToolState> A2AConversationExt for AgentBuilder<ToolState> {
    fn a2a_conversation(self, id: impl Into<String>) -> Self {
        let id = id.into();
        self.conversation(id.clone())
            .additional_params(Value::Object(conversation_params(id)))
    }
}

impl A2AConversationExt for AgentRunner {
    fn a2a_conversation(self, id: impl Into<String>) -> Self {
        let id = id.into();
        self.conversation(id.clone())
            .merge_additional_params(conversation_params(id))
    }
}

impl<S> A2AConversationExt for PromptRequest<S>
where
    S: PromptType,
{
    fn a2a_conversation(self, id: impl Into<String>) -> Self {
        let id = id.into();
        self.conversation(id.clone())
            .merge_additional_params(conversation_params(id))
    }
}

impl A2AConversationExt for StreamingPromptRequest {
    fn a2a_conversation(self, id: impl Into<String>) -> Self {
        let id = id.into();
        self.conversation(id.clone())
            .merge_additional_params(conversation_params(id))
    }
}
