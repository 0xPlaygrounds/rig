//! Preserves the live multi-extract example as provider-local regression coverage.

use std::sync::Arc;

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::provider::Runtime;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::super::support::with_openai_cassette_result;
use crate::cassettes::CassetteSpec;
use crate::support::assert_nonempty_response;

/// The classic `ExtractorBuilder::preamble(extra)` appended `extra` to the
/// pinned extraction preamble behind an `ADDITIONAL INSTRUCTIONS` banner. Keep
/// that byte-for-byte so the recorded requests still match.
fn classic_options_with_extra_preamble(extra: &str, retries: usize) -> ExtractOptions {
    let options = ExtractOptions::classic_extractor();
    let base = options
        .preamble
        .clone()
        .expect("the classic extractor pins a preamble");

    options.with_retries(retries).with_preamble(format!(
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
    with_openai_cassette_result(
        CassetteSpec::new("multi_extract/batch_multi_extract_chain").unordered(),
        |client| async move {
            let provider = client.provider_config(openai::GPT_4O_MINI);
            let rt = Arc::new(Runtime::new());

            let names_options =
                classic_options_with_extra_preamble("Extract names from the given text.", 2);
            let topics_options =
                classic_options_with_extra_preamble("Extract topics from the given text.", 2);
            let sentiment_options = classic_options_with_extra_preamble(
                "Extract sentiment and confidence from the given text.",
                2,
            );

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
                    let provider = provider.clone();
                    let rt = rt.clone();
                    let names_options = names_options.clone();
                    let topics_options = topics_options.clone();
                    let sentiment_options = sentiment_options.clone();
                    async move {
                        let (names, topics, sentiment) = futures::try_join!(
                            extract_with_options::<Names>(
                                AgentConfig::new(),
                                provider.clone(),
                                rt.clone(),
                                text,
                                names_options,
                            ),
                            extract_with_options::<Topics>(
                                AgentConfig::new(),
                                provider.clone(),
                                rt.clone(),
                                text,
                                topics_options,
                            ),
                            extract_with_options::<Sentiment>(
                                AgentConfig::new(),
                                provider.clone(),
                                rt.clone(),
                                text,
                                sentiment_options,
                            ),
                        )?;
                        let (names, topics, sentiment) =
                            (names.value, topics.value, sentiment.value);
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
