//! Preserves the live multi-extract example as ChatGPT regression coverage.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::extract::ExtractOptions;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::chatgpt::{LIVE_MODEL, live_agent};
use crate::support::assert_nonempty_response;

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
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn batch_multi_extract_chain() -> Result<()> {
    let agent = live_agent(LIVE_MODEL).await.build();
    let names_options = classic_options("Extract names from the given text.").with_retries(2);
    let topics_options = classic_options("Extract topics from the given text.").with_retries(2);
    let sentiment_options =
        classic_options("Extract sentiment and confidence from the given text.").with_retries(2);

    let inputs = vec![
        "Screw you Putin!",
        "I love my dog, but I hate my cat.",
        "I'm going to the store to buy some milk.",
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
                        .retries(names_options.retries)
                        .preamble(
                            names_options
                                .preamble
                                .clone()
                                .expect("preamble should exist")
                        )
                        .run::<Names>(),
                    agent
                        .extractor(text)
                        .retries(topics_options.retries)
                        .preamble(
                            topics_options
                                .preamble
                                .clone()
                                .expect("preamble should exist")
                        )
                        .run::<Topics>(),
                    agent
                        .extractor(text)
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
