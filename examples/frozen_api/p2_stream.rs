use std::io::Write;

use futures::StreamExt;
use rig::providers::openai::{self, OpenAI};
use rig::streaming::{Item, StreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = OpenAI::from_env()?.completion(openai::GPT_5_2);
    let mut stream = model.stream("Write a haiku about the borrow checker.")?;

    // An `Err` item is always the last item: the stream ends after it.
    while let Some(item) = stream.next().await {
        match item? {
            Item::Event(StreamEvent::Text { text, .. }) => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            Item::Event(_) => {}
            Item::Unknown(payload) => eprintln!("\n[unmodeled provider event: {payload:?}]"),
        }
    }

    // The same response `call` would have returned for this reply.
    let response = stream.finish().await?;
    println!("\nusage: {:?}", response.usage);
    Ok(())
}
