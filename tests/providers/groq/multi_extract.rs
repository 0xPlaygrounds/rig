//! Groq live coverage for batch multi-extract pipelines.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::groq;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::assert_nonempty_response;

use super::{MULTI_EXTRACT_NAMES_MODEL, MULTI_EXTRACT_SENTIMENT_MODEL, MULTI_EXTRACT_TOPICS_MODEL};

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
#[ignore = "requires GROQ_API_KEY"]
async fn batch_multi_extract_chain() -> Result<()> {
    let client = groq::Client::from_env().expect("client should build");
    let names_extractor = client
        .extractor::<Names>(MULTI_EXTRACT_NAMES_MODEL)
        .preamble("Extract names from the given text.")
        .retries(2)
        .build();
    let topics_extractor = client
        .extractor::<Topics>(MULTI_EXTRACT_TOPICS_MODEL)
        .preamble("Extract topics from the given text.")
        .retries(2)
        .build();
    let sentiment_extractor = client
        .extractor::<Sentiment>(MULTI_EXTRACT_SENTIMENT_MODEL)
        .preamble("Extract sentiment and confidence from the given text.")
        .retries(2)
        .build();

    let inputs = vec![
        "Ada Lovelace discussed analytical engines and early programming.",
        "I love my dog, but I hate rainy weather.",
        "I'm going to the store to buy milk and bread.",
    ];
    let responses: Vec<String> = futures::stream::iter(inputs)
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

    anyhow::ensure!(
        responses.len() == 3,
        "expected three responses, got {}",
        responses.len()
    );
    for response in responses {
        assert_nonempty_response(&response);
    }

    Ok(())
}
