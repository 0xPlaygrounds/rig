//! Preserves the live multi-extract example as Copilot regression coverage.

use rig::wire::Wire as _;
use std::future::IntoFuture;

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
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
            let names_extractor = rig::extractor::ExtractorBuilder::<Names>::new(client.completion(LIVE_LIGHT_MODEL).on(rig::transport()))
                .append_preamble("Extract names from the given text.")
                .retries(2)
                .build();
            let topics_extractor = rig::extractor::ExtractorBuilder::<Topics>::new(client.completion(LIVE_LIGHT_MODEL).on(rig::transport()))
                .append_preamble("Extract topics from the given text.")
                .retries(2)
                .build();
            let sentiment_extractor = rig::extractor::ExtractorBuilder::<Sentiment>::new(client.completion(LIVE_LIGHT_MODEL).on(rig::transport()))
                .append_preamble("Extract sentiment and confidence from the given text.")
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
                            names_extractor.extract(text).into_future(),
                            topics_extractor.extract(text).into_future(),
                            sentiment_extractor.extract(text).into_future(),
                        )?;
                        anyhow::Ok(format!(
                            "Extracted names: {}\nExtracted topics: {}\nExtracted sentiment: {} ({})",
                            names.output.names.join(", "),
                            topics.output.topics.join(", "),
                            sentiment.output.sentiment,
                            sentiment.output.confidence,
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
