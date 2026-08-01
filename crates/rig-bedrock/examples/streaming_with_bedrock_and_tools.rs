//! Streaming a tool-using Bedrock agent, draining the item stream by hand.
//!
//! The `stream_to_stdout` helper was example sugar and is gone; the drain loop
//! below is what it did, spelled out.
use futures::StreamExt;
use rig_agent::agent::{PromptResponse, Text};
use rig_agent::client::AgentClientExt;
use rig_agent::stream::{AgentRunStream, AgentStreamItem};
use rig_agent::streaming::StreamedAssistantContent;
use rig_bedrock::completion::AMAZON_NOVA_LITE;
mod common;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();
    // Create agent with a single context prompt and two tools
    let client = rig_bedrock::Client::from_env();
    let agent = client
        .agent(AMAZON_NOVA_LITE)
        .preamble(
            "You are a calculator here to help the user perform arithmetic
            operations. Use the tools provided to answer the user's question.
            make your answer long, so we can test the streaming functionality,
            like 20 words",
        )
        .max_tokens(1024)
        .tool(common::Adder)
        .build();

    println!("Calculate 2 + 5");
    let stream = agent.runner("Calculate 2 + 5").stream_run();
    let _ = drain_to_stdout(stream).await?;
    Ok(())
}

/// Drain a streamed run to stdout, returning the final [`PromptResponse`].
///
/// The old `stream_to_stdout` example helper is gone, so each example inlines
/// its own drain loop: print assistant text and reasoning deltas as they
/// arrive, keep the terminal `FinalResponse` for usage/output, and mark a
/// model-turn retry (text already written to stdout cannot be retracted).
async fn drain_to_stdout(mut stream: AgentRunStream) -> anyhow::Result<PromptResponse> {
    let mut final_response = PromptResponse::empty();
    print!("Response: ");
    while let Some(item) = stream.next().await {
        match item {
            Ok(AgentStreamItem::Assistant(StreamedAssistantContent::Text(Text {
                text, ..
            }))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(AgentStreamItem::Assistant(StreamedAssistantContent::Reasoning(reasoning))) => {
                print!("{}", reasoning.display_text());
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(AgentStreamItem::Final(response)) => final_response = response,
            Ok(AgentStreamItem::ModelTurnRetried { turn }) => {
                print!("\n[model turn {turn} rejected; retry requested]\nResponse: ");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(err) => eprintln!("Error: {err}"),
            _ => {}
        }
    }
    println!();
    Ok(final_response)
}
