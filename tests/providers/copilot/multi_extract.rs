//! Preserves the live multi-extract example as Copilot regression coverage.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::client::CompletionClient;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::copilot::{LIVE_LIGHT_MODEL, with_copilot_cassette_result};
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
    with_copilot_cassette_result(
        crate::cassettes::CassetteSpec::new("multi_extract/batch_multi_extract_chain").unordered(),
        |client| async move {
            let names_extractor = client
                .extractor::<Names>(LIVE_LIGHT_MODEL)
                .preamble("Extract names from the given text.")
                .retries(2)
                .build();
            let topics_extractor = client
                .extractor::<Topics>(LIVE_LIGHT_MODEL)
                .preamble("Extract topics from the given text.")
                .retries(2)
                .build();
            let sentiment_extractor = client
                .extractor::<Sentiment>(LIVE_LIGHT_MODEL)
                .preamble("Extract sentiment and confidence from the given text.")
                .retries(2)
                .build();

            let inputs = vec![
                "Screw you Putin!",
                "I love my dog, but I hate my cat.",
                "I'm going to the store to buy some milk.",
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

            anyhow::ensure!(responses.len() == 3);
            for response in responses {
                assert_nonempty_response(&response);
            }

            Ok(())
        },
    )
    .await
}
