//! Back a Rig agent with a remote A2A agent instead of an LLM provider.
//!
//! [`A2AModel`](rig_a2a::A2AModel) implements `CompletionModel` over A2A's
//! `message/send` and `message/stream`, so `A2AClient::agent` returns an
//! ordinary Rig [`Agent`]: it prompts, it streams, and it composes as a
//! sub-agent tool on an orchestrator.
//!
//! What it cannot do is carry a chat-completion API's whole surface. A remote
//! A2A agent owns its instructions and hides its tools, so `tools` and
//! `output_schema` on a request fail loudly rather than being dropped —
//! shown at the end.
//!
//! ```sh
//! cargo run --example a2a_agent -p rig-a2a
//! ```
//!
//! [`Agent`]: rig_agent::agent::Agent

use futures::StreamExt;
use rig_a2a::{A2AClient, A2AConversationExt};
use rig_agent::agent::AgentBuilder;
use rig_agent::completion::Prompt;
use rig_agent::test_utils::{MockCompletionModel, MockTurn};
use rig_core::completion::CompletionModel;
use rig_core::streaming::StreamedAssistantContent;

#[path = "./fixtures/lib.rs"]
mod fixture;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let stub = fixture::serve_stub_agent([
        fixture::Reply::completed("I index and answer questions about documents."),
        fixture::Reply::completed("Earlier you asked what I do."),
        fixture::Reply::completed("Streaming works the same way."),
        fixture::Reply::completed("Three documents mention the migration."),
    ])
    .await;
    let remote = A2AClient::from_url(&stub.base_url).await?;

    // The remote agent as a Rig agent. Its name and description come from the
    // card. Naming a conversation on the prompt threads it: one agent serves
    // every conversation, each with its own remote thread.
    let agent = remote.agent().build();
    println!("agent {:?}", agent.name());
    println!(
        "  turn 1: {}",
        agent
            .prompt("What do you do?")
            .a2a_conversation("user-42")
            .await?
    );
    println!(
        "  turn 2: {}\n",
        agent
            .prompt("What did I just ask?")
            .a2a_conversation("user-42")
            .await?
    );

    // Stream directly from the A2A completion model.
    let request = remote
        .model()
        .completion_request("Stream me a reply.")
        .build();
    let mut stream = remote.model().stream(request).await?;
    print!("streamed: ");
    while let Some(chunk) = stream.next().await {
        if let StreamedAssistantContent::Text(delta) = chunk? {
            print!("{}", delta.text);
        }
    }
    println!("\n");

    // Composition: an A2A-backed agent is an ordinary agent, so the sub-agent
    // bridge applies. Here a local orchestrator delegates to it.
    let researcher = remote.agent().a2a_conversation("research").build();
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "call_1",
            "librarian",
            serde_json::json!({"prompt": "Search for migration notes."}),
        ),
        MockTurn::text("Three documents mention the migration."),
    ]);
    let orchestrator = AgentBuilder::new(model)
        .preamble("Delegate research to the librarian sub-agent.")
        .dynamic_tool(researcher.into_tool())
        .build();
    println!(
        "orchestrator: {}\n",
        orchestrator
            .prompt("What do we know about the migration?")
            .max_turns(3)
            .await?
    );

    // A2A hides the remote's tools by design, so a remote agent never returns a
    // tool call. Registering local tools would advertise tools that can never
    // be invoked, so the request is refused instead.
    let mut with_tools = remote.model().completion_request("hi").build();
    with_tools.tools.push(rig_core::completion::ToolDefinition {
        name: "add".to_string(),
        description: "adds two numbers".to_string(),
        parameters: serde_json::json!({}),
    });
    match remote.model().completion(with_tools).await {
        Ok(_) => println!("unexpectedly accepted a tool-bearing request"),
        Err(error) => println!("refused, as intended: {error}"),
    }

    Ok(())
}
