//! Demonstrates fan-out structured extraction with `futures::try_join!`.
//! Requires `OPENAI_API_KEY`.
//! Run it to see one batch of text split into names, topics, and sentiment in parallel.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::client::ProviderClient;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Names {
    names: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Topics {
    topics: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Sentiment {
    sentiment: f64,
    confidence: f64,
}

fn sample_inputs() -> Vec<&'static str> {
    vec![
        "Screw you Putin!",
        "I love my dog, but I hate my cat.",
        "I'm going to the store to buy some milk.",
    ]
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;
    let names_extractor = client
        .extractor::<Names>(openai::GPT_4O_MINI)
        .preamble("Extract names from the given text.")
        .retries(2)
        .build();
    let topics_extractor = client
        .extractor::<Topics>(openai::GPT_4O_MINI)
        .preamble("Extract topics from the given text.")
        .retries(2)
        .build();
    let sentiment_extractor = client
        .extractor::<Sentiment>(openai::GPT_4O_MINI)
        .preamble("Extract sentiment and confidence from the given text.")
        .retries(2)
        .build();

    // Fan each input out to the three extractors concurrently (`try_join!`),
    // running up to four inputs at a time (`buffered`) — the same shape the
    // old `try_parallel!` + `try_batch_call(4, ..)` pipeline provided.
    let responses: Vec<String> = futures::stream::iter(sample_inputs())
        .map(|text| {
            let names_extractor = &names_extractor;
            let topics_extractor = &topics_extractor;
            let sentiment_extractor = &sentiment_extractor;
            async move {
                let (names, topics, sentiment) = futures::try_join!(
                    names_extractor.extract(text),
                    topics_extractor.extract(text),
                    sentiment_extractor.extract(text),
                )?;
                anyhow::Ok(format!(
                    "Extracted names: {}\nExtracted topics: {}\nExtracted sentiment: {} ({})",
                    names.names.join(", "),
                    topics.topics.join(", "),
                    sentiment.sentiment,
                    sentiment.confidence,
                ))
            }
        })
        .buffered(4)
        .try_collect()
        .await?;

    for (idx, response) in responses.iter().enumerate() {
        println!("batch item {}:\n{response}\n", idx + 1);
    }

    Ok(())
}
