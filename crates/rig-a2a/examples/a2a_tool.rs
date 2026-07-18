//! Give a local Rig agent access to a remote A2A agent.
//!
//! A whole remote agent becomes **one** tool taking a single `prompt` — A2A
//! carries no skill selector, so the skills a card declares are documentation
//! in the tool description rather than separate tools.
//!
//! Successive calls sharing a [`conversation_context`] continue one remote
//! conversation. The server-issued identifiers stay on the host: they never
//! enter the tool's schema or output, so the model can neither forge one nor
//! lose one, and a task the remote pauses is resumed automatically.
//!
//! The local model is scripted so this runs offline; swap in any provider
//! model and the wiring is unchanged.
//!
//! ```sh
//! cargo run --example a2a_tool -p rig-a2a
//! ```

use rig_a2a::{A2AAgentBuilderExt, A2AClient, A2AThreadInfo, conversation_context};
use rig_agent::agent::{AgentBuilder, AgentHook, HookContext, ToolResultAction, ToolResultEvent};
use rig_agent::completion::Prompt;
use rig_agent::test_utils::{MockCompletionModel, MockTurn};

#[path = "./fixtures/lib.rs"]
mod fixture;

/// The A2A identifiers reach the host, never the model. Read them off the
/// tool's result metadata to trace, log, or persist a remote conversation.
struct LogA2AIds;

impl AgentHook for LogA2AIds {
    async fn on_tool_result(
        &self,
        _context: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        if let Some(info) = event.tool_context.result::<A2AThreadInfo>() {
            println!(
                "    [host] contextId={:?} state={:?} resumable={}",
                info.context_id,
                info.state_label(),
                info.resumable
            );
        }
        ToolResultAction::Keep
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let stub = fixture::serve_stub_agent([
        fixture::Reply::input_required("Which file should I summarize?"),
        fixture::Reply::completed("README.md covers installation and usage."),
    ])
    .await;

    // Discover the remote agent, then bind it to a local agent as a tool.
    let remote = A2AClient::from_url(&stub.base_url).await?;
    println!(
        "remote agent {:?} declares {} skills, projected as one tool {:?}\n",
        remote.card().name,
        remote.card().skills.len(),
        remote.tool_name()
    );
    println!("{}\n", remote.tool().definition().description);

    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "call_1",
            remote.tool_name(),
            serde_json::json!({"prompt": "Summarize a file for me."}),
        ),
        MockTurn::tool_call(
            "call_2",
            remote.tool_name(),
            serde_json::json!({"prompt": "README.md"}),
        ),
        MockTurn::text("The README covers installation and usage."),
    ]);

    let agent = AgentBuilder::new(model)
        .name("orchestrator")
        .preamble("Delegate document questions to the remote agent.")
        .a2a_tool(&remote)
        .add_hook(LogA2AIds)
        .build();

    // Both tool calls share one `ConversationId`, so the second continues the
    // remote conversation the first opened — and resumes the task the remote
    // paused, without the model ever handling an identifier.
    let reply = agent
        .prompt("What does the README cover?")
        .tool_context(conversation_context("user-42"))
        .max_turns(4)
        .await?;
    println!("\nagent reply: {reply}");

    Ok(())
}
