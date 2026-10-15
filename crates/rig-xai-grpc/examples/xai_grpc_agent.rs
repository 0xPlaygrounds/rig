//! An agent with one tool on xAI's gRPC API: a prompt answered after a
//! tool call, then a streamed answer.
//!
//! ```sh
//! XAI_API_KEY=... cargo run -p rig-xai-grpc --example xai_grpc_agent
//! ```

use futures::StreamExt;
use rig_agent::agent::MultiTurnStreamItem;
use rig_agent::prelude::*;
use rig_agent::tool::ToolContext;
use rig_core::streaming::Item;
use rig_core::streaming::StreamEvent;
use rig_xai_grpc::XaiGrpc;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct AddArgs {
    x: i64,
    y: i64,
}

/// Adds two integers.
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = std::convert::Infallible;
    type Args = AddArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add the integers x and y.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "x": { "type": "integer" }, "y": { "type": "integer" } },
            "required": ["x", "y"]
        })
    }

    async fn call(&self, _: &mut ToolContext, args: AddArgs) -> Result<i64, Self::Error> {
        Ok(args.x.saturating_add(args.y))
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let transport = XaiGrpc::from_env().await?;
    let agent = AgentBuilder::new(transport.completion("grok-4.20-0309-non-reasoning"))
        .preamble("Be brief. Use the add tool for arithmetic.")
        .tool(Add)
        .max_tokens(200)
        .build();

    let response = agent.prompt("What is 1234 + 5678?").max_turns(3).await?;
    println!("unary: {}", response.output());

    let mut stream = agent
        .prompt("Name one moon of Jupiter, in one word.")
        .stream();
    print!("streamed: ");
    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                text,
                ..
            })) => print!("{text}"),
            MultiTurnStreamItem::FinalResponse(response) => {
                println!("\nusage: {:?}", response.usage());
            }
            _ => {}
        }
    }
    Ok(())
}
