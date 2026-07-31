//! Streaming with Bedrock, draining the item stream by hand.
//!
//! The `stream_to_stdout` helper was example sugar and is gone; the drain loop
//! below is what it did, spelled out.
use futures::StreamExt;
use rig_agent::agent::{AgentBuilder, PromptResponse, Text};
use rig_agent::stream::AgentStreamItem;
use rig_agent::streaming::StreamedAssistantContent;
use rig_bedrock::completion::AMAZON_NOVA_LITE;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = AgentBuilder::new(rig_bedrock::functions::Config::new(AMAZON_NOVA_LITE))
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .runner("When and where and what type is the next solar eclipse?")
        .stream_run();

    let _ = drain_to_stdout(&mut stream).await?;

    Ok(())
}

/// Drain a streamed run to stdout, returning the final [`PromptResponse`].
///
/// The old `stream_to_stdout` example helper is gone, so each example inlines
/// its own drain loop: print assistant text and reasoning deltas as they
/// arrive, keep the terminal `FinalResponse` for usage/output, and mark a
/// model-turn retry (text already written to stdout cannot be retracted).
async fn drain_to_stdout<S>(stream: &mut S) -> anyhow::Result<PromptResponse>
where
    S: futures::Stream<
            Item = Result<rig_agent::stream::AgentStreamItem, rig_agent::completion::PromptError>,
        > + Unpin,
{
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
