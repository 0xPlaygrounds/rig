//! Preserves the live multi-extract example as provider-local regression coverage.

use anyhow::Result;
use rig_core::client::ProviderClient;
use rig_core::pipeline::{self, TryOp, agent_ops};
use rig_core::providers::openai;
use rig_core::try_parallel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::assert_nonempty_response;

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

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn batch_multi_extract_chain() -> Result<()> {
    let client = openai::Client::from_env().expect("client should build");
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

    anyhow::ensure!(responses.len() == 3);
    for response in responses {
        assert_nonempty_response(&response);
    }

    Ok(())
}
