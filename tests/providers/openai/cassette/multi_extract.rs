//! Preserves the live multi-extract example as provider-local regression coverage.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::extract::ExtractOptions;
use rig::prelude::*;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::super::support::with_openai_cassette_result;
use crate::cassettes::CassetteSpec;
use crate::support::assert_nonempty_response;

/// The classic `ExtractorBuilder::preamble(extra)` appended `extra` to the
/// pinned extraction preamble behind an `ADDITIONAL INSTRUCTIONS` banner. Keep
/// that byte-for-byte so the recorded requests still match.
fn classic_preamble(extra: &str) -> String {
    let options = ExtractOptions::classic_extractor();
    let base = options
        .preamble
        .clone()
        .expect("the classic extractor pins a preamble");

    format!("{base}\n\n=============== ADDITIONAL INSTRUCTIONS ===============\n{extra}")
}

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
            let agent = client.agent(openai::GPT_4O_MINI).build();

            let names_preamble = classic_preamble("Extract names from the given text.");
            let topics_preamble = classic_preamble("Extract topics from the given text.");
            let sentiment_preamble =
                classic_preamble("Extract sentiment and confidence from the given text.");

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
                    let agent = agent.clone();
                    let names_preamble = names_preamble.clone();
                    let topics_preamble = topics_preamble.clone();
                    let sentiment_preamble = sentiment_preamble.clone();
                    async move {
                        let (names, topics, sentiment) = futures::try_join!(
                            agent
                                .extractor(text)
                                .classic()
                                .retries(2)
                                .preamble(names_preamble)
                                .run::<Names>(),
                            agent
                                .extractor(text)
                                .classic()
                                .retries(2)
                                .preamble(topics_preamble)
                                .run::<Topics>(),
                            agent
                                .extractor(text)
                                .classic()
                                .retries(2)
                                .preamble(sentiment_preamble)
                                .run::<Sentiment>(),
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
