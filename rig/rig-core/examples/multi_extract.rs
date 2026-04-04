use anyhow::Result;
use rig::client::ProviderClient;
use rig::pipeline::{self, TryOp, agent_ops};
use rig::providers::openai;
use rig::try_parallel;
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

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env();
    let names_extractor = client
        .extractor::<Names>(openai::GPT_4)
        .preamble("Extract names from the given text.")
        .build();
    let topics_extractor = client
        .extractor::<Topics>(openai::GPT_4)
        .preamble("Extract topics from the given text.")
        .build();
    let sentiment_extractor = client
        .extractor::<Sentiment>(openai::GPT_4)
        .preamble("Extract sentiment and confidence from the given text.")
        .build();

    let chain = pipeline::new()
        .chain(try_parallel!(
            agent_ops::extract(names_extractor),
            agent_ops::extract(topics_extractor),
            agent_ops::extract(sentiment_extractor),
        ))
        .map_ok(|(names, topics, sentiment)| {
            format!(
                "Extracted names: {}\nExtracted topics: {}\nExtracted sentiment: {} ({})",
                names.names.join(", "),
                topics.topics.join(", "),
                sentiment.sentiment,
                sentiment.confidence,
            )
        });

    let responses = chain
        .try_batch_call(
            4,
            vec![
                "Screw you Putin!",
                "I love my dog, but I hate my cat.",
                "I'm going to the store to buy some milk.",
            ],
        )
        .await?;

    for (idx, response) in responses.iter().enumerate() {
        println!("batch item {}:\n{response}\n", idx + 1);
    }

    Ok(())
}
