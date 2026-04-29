//! xAI live coverage for batch multi-extract pipelines.

use anyhow::Result;
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::pipeline::{self, TryOp, agent_ops};
use rig_core::providers::xai;
use rig_core::try_parallel;
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

#[derive(Debug)]
struct CombinedExtract {
    names: Vec<String>,
    topics: Vec<String>,
    sentiment: f64,
    confidence: f64,
}

fn joined_lower(items: &[String]) -> String {
    items.join(" ").to_ascii_lowercase()
}

fn assert_contains_any(items: &[String], expected: &[&str], label: &str) {
    let joined = joined_lower(items);
    assert!(
        expected
            .iter()
            .any(|expected| joined.contains(&expected.to_ascii_lowercase())),
        "expected {label} to contain one of {expected:?}, got {items:?}",
    );
}

fn assert_sentiment_shape(extract: &CombinedExtract) {
    assert!(
        extract.sentiment.is_finite(),
        "sentiment should be finite, got {:?}",
        extract.sentiment
    );
    assert!(
        (-1.0..=1.0).contains(&extract.sentiment),
        "sentiment should be normalized to [-1.0, 1.0], got {:?}",
        extract.sentiment
    );
    assert!(
        extract.confidence.is_finite(),
        "confidence should be finite, got {:?}",
        extract.confidence
    );
    assert!(
        (0.0..=1.0).contains(&extract.confidence),
        "confidence should be normalized to [0.0, 1.0], got {:?}",
        extract.confidence
    );
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn batch_multi_extract_chain() -> Result<()> {
    let client = xai::Client::from_env().expect("client should build");
    let names_extractor = client
        .extractor::<Names>(xai::GROK_3_MINI)
        .preamble("Extract names from the given text.")
        .retries(2)
        .build();
    let topics_extractor = client
        .extractor::<Topics>(xai::GROK_3_MINI)
        .preamble("Extract topics from the given text.")
        .retries(2)
        .build();
    let sentiment_extractor = client
        .extractor::<Sentiment>(xai::GROK_3_MINI)
        .preamble(
            "Extract sentiment and confidence from the given text. \
             Return sentiment normalized to the range [-1.0, 1.0] and confidence normalized to [0.0, 1.0].",
        )
        .retries(2)
        .build();

    let chain = pipeline::new()
        .chain(try_parallel!(
            agent_ops::extract(names_extractor),
            agent_ops::extract(topics_extractor),
            agent_ops::extract(sentiment_extractor),
        ))
        .map_ok(|(names, topics, sentiment)| CombinedExtract {
            names: names.names,
            topics: topics.topics,
            sentiment: sentiment.sentiment,
            confidence: sentiment.confidence,
        });

    let responses = chain
        .try_batch_call(
            4,
            vec![
                "Ada Lovelace discussed analytical engines and early programming with Charles Babbage.",
                "Grace said she hates rainy weather but still walked her dog to the park.",
                "Linus is going to the store to buy milk and bread for dinner.",
            ],
        )
        .await?;

    anyhow::ensure!(responses.len() == 3);

    assert_contains_any(
        &responses[0].names,
        &["ada", "lovelace", "charles", "babbage"],
        "names",
    );
    assert_contains_any(
        &responses[0].topics,
        &["analytical", "engine", "programming"],
        "topics",
    );
    assert_sentiment_shape(&responses[0]);

    assert_contains_any(&responses[1].names, &["grace"], "names");
    assert_contains_any(
        &responses[1].topics,
        &["dog", "rain", "weather", "park"],
        "topics",
    );
    assert_sentiment_shape(&responses[1]);

    assert_contains_any(&responses[2].names, &["linus"], "names");
    assert_contains_any(
        &responses[2].topics,
        &["store", "milk", "bread", "dinner"],
        "topics",
    );
    assert_sentiment_shape(&responses[2]);

    Ok(())
}
