use rig::prelude::*;
use rig::{
    pipeline::{self, TryOp, agent_ops},
    providers::openai,
    try_parallel,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A record containing extracted names
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
pub struct Names {
    /// The names extracted from the text
    pub names: Vec<String>,
}

/// A record containing extracted topics
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
pub struct Topics {
    /// The topics extracted from the text
    pub topics: Vec<String>,
}

/// A record containing extracted sentiment
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
pub struct Sentiment {
    /// The sentiment of the text (-1 being negative, 1 being positive)
    pub sentiment: f64,
    /// The confidence of the sentiment
    pub confidence: f64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let openai = openai::Client::from_env();

    let names_extractor = openai
        .extractor::<Names>(openai::GPT_4)
        .preamble("Extract names (e.g.: of people, places) from the given text.")
        .build();
    let topics_extractor = openai
        .extractor::<Topics>(openai::GPT_4)
        .preamble("Extract topics from the given text.")
        .build();
    let sentiment_extractor = openai
        .extractor::<Sentiment>(openai::GPT_4)
        .preamble(
            "Extract sentiment (and how confident you are of the sentiment) from the given text.",
        )
        .build();

    // Create a chain that extracts names, topics, and sentiment from a given text
    // using three different GPT-4 based extractors.
    // The chain will output a formatted string containing the extracted information.
    let chain = pipeline::new()
        .chain(try_parallel!(
            agent_ops::extract(names_extractor),
            agent_ops::extract(topics_extractor),
            agent_ops::extract(sentiment_extractor),
        ))
        .map_ok(|(names, topics, sentiment)| {
            format!(
                "Extracted names: {names}\nExtracted topics: {topics}\nExtracted sentiment: {sentiment}",
                names = names.names.join(", "),
                topics = topics.topics.join(", "),
                sentiment = sentiment.sentiment,
            )
        });

    // Batch call the chain with up to 4 inputs concurrently
    let response = chain
        .try_batch_call(
            4,
            vec![
                "Screw you Putin!",
                "I love my dog, but I hate my cat.",
                "I'm going to the store to buy some milk.",
            ],
        )
        .await?;

    for response in response {
        println!("Text analysis:\n{response}");
    }

    Ok(())
}
