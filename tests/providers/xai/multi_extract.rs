//! xAI live coverage for batch multi-extract pipelines.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::client::CompletionClient;
use rig::providers::xai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::support::with_xai_cassette_result;
use crate::cassettes::CassetteSpec;

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
async fn batch_multi_extract_chain() -> Result<()> {
    with_xai_cassette_result(
        CassetteSpec::new("multi_extract/batch_multi_extract_chain").unordered(),
        |client| async move {
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

            let inputs = vec![
                "Ada Lovelace discussed analytical engines and early programming with Charles Babbage.",
                "Grace said she hates rainy weather but still walked her dog to the park.",
                "Linus is going to the store to buy milk and bread for dinner.",
            ];
            let responses: Vec<CombinedExtract> = futures::stream::iter(inputs)
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
                        anyhow::Ok(CombinedExtract {
                            names: names.names,
                            topics: topics.topics,
                            sentiment: sentiment.sentiment,
                            confidence: sentiment.confidence,
                        })
                    }
                })
                .buffered(4)
                .try_collect()
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
        },
    )
    .await
}
