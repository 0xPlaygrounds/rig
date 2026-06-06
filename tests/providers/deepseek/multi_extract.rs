//! DeepSeek live coverage for batch multi-extract pipelines.

use anyhow::Result;
use rig::client::CompletionClient;
use rig::pipeline::{self, TryOp, agent_ops};
use rig::providers::deepseek;
use rig::try_parallel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::support::with_deepseek_cassette_result;
use crate::cassettes::CassetteSpec;
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
async fn batch_multi_extract_chain() -> Result<()> {
    with_deepseek_cassette_result(
        CassetteSpec::new("multi_extract/batch_multi_extract_chain").unordered(),
        |client| async move {
            let names_extractor = client
                .extractor::<Names>(deepseek::DEEPSEEK_V4_FLASH)
                .preamble("Extract names from the given text.")
                .retries(2)
                .build();
            let topics_extractor = client
                .extractor::<Topics>(deepseek::DEEPSEEK_V4_FLASH)
                .preamble("Extract topics from the given text.")
                .retries(2)
                .build();
            let sentiment_extractor = client
                .extractor::<Sentiment>(deepseek::DEEPSEEK_V4_FLASH)
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
                        "Ada Lovelace discussed analytical engines and early programming.",
                        "I love my dog, but I hate rainy weather.",
                        "I'm going to the store to buy milk and bread.",
                    ],
                )
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
        },
    )
    .await
}
