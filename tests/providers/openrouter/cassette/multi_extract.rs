//! Cassette-backed OpenRouter batch multi-extract pipelines.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::extract::ExtractOptions;
use rig::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::cassettes::CassetteSpec;
use crate::support::assert_nonempty_response;

use super::super::{DEFAULT_MODEL, support::with_openrouter_cassette_result};

/// `ExtractorBuilder::preamble(extra)` appended the extra instructions to the
/// pinned extraction preamble: `append_preamble` joined with a newline and the
/// appended block itself opened with one, so the separator is two newlines.
fn classic_options(extra: &str) -> ExtractOptions {
    let options = ExtractOptions::classic_extractor();
    let base = options
        .preamble
        .clone()
        .expect("classic_extractor() pins a preamble");
    options.with_preamble(format!(
        "{base}\n\n=============== ADDITIONAL INSTRUCTIONS ===============\n{extra}"
    ))
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
    with_openrouter_cassette_result(
        CassetteSpec::new("multi_extract/batch_multi_extract_chain").unordered(),
        |client| async move {
            let agent = client.agent(DEFAULT_MODEL).build();
            let names_options = classic_options("Extract names from the given text.").with_retries(2);
            let topics_options = classic_options("Extract topics from the given text.").with_retries(2);
            let sentiment_options = classic_options("Extract sentiment and confidence from the given text.").with_retries(2);

            let inputs = vec![
                "Ada Lovelace discussed analytical engines and early programming.",
                "I love my dog, but I hate rainy weather.",
                "I'm going to the store to buy milk and bread.",
            ];
            let responses: Vec<String> = futures::stream::iter(inputs)
                .map(|text| {
                    let agent = &agent;
                    let names_options = &names_options;
                    let topics_options = &topics_options;
                    let sentiment_options = &sentiment_options;
                    async move {
                        let (names, topics, sentiment) = futures::try_join!(
                            agent
                                .extractor(text)
                                .classic()
                                .retries(names_options.retries)
                                .preamble(names_options.preamble.clone().expect("preamble should exist"))
                                .run::<Names>(),
                            agent
                                .extractor(text)
                                .classic()
                                .retries(topics_options.retries)
                                .preamble(topics_options.preamble.clone().expect("preamble should exist"))
                                .run::<Topics>(),
                            agent
                                .extractor(text)
                                .classic()
                                .retries(sentiment_options.retries)
                                .preamble(
                                    sentiment_options
                                        .preamble
                                        .clone()
                                        .expect("preamble should exist"),
                                )
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
