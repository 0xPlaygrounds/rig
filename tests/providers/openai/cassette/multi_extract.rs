//! Preserves the live multi-extract example as provider-local regression coverage.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::super::support::with_openai_cassette_result;
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
    with_openai_cassette_result(
        CassetteSpec::new("multi_extract/batch_multi_extract_chain").unordered(),
        |client| async move {
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

            // Fan out each input to the three extractors concurrently
            // (`try_join!`), and run up to four inputs at a time
            // (`buffer_unordered`) — the same concurrency the pipeline's
            // `try_parallel!` + `try_batch_call(4, ..)` provided.
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
