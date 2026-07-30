//! Groq live coverage for batch multi-extract pipelines.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::provider::{ProviderConfig, Runtime};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::support::assert_nonempty_response;

use super::{MULTI_EXTRACT_NAMES_MODEL, MULTI_EXTRACT_SENTIMENT_MODEL, MULTI_EXTRACT_TOPICS_MODEL};

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

/// One classic-`Extractor<T>::extract` exchange through the free-function
/// extraction surface that replaced it.
async fn classic_extract_value<T>(
    provider: ProviderConfig,
    text: &str,
    options: ExtractOptions,
) -> anyhow::Result<T>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    Ok(extract_with_options::<T>(
        AgentConfig::new(),
        provider,
        Arc::new(Runtime::new()),
        text,
        options,
    )
    .await?
    .value)
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
#[ignore = "requires GROQ_API_KEY"]
async fn batch_multi_extract_chain() -> Result<()> {
    let names_provider = super::live(MULTI_EXTRACT_NAMES_MODEL);
    let topics_provider = super::live(MULTI_EXTRACT_TOPICS_MODEL);
    let sentiment_provider = super::live(MULTI_EXTRACT_SENTIMENT_MODEL);
    let names_options = classic_options("Extract names from the given text.").with_retries(2);
    let topics_options = classic_options("Extract topics from the given text.").with_retries(2);
    let sentiment_options =
        classic_options("Extract sentiment and confidence from the given text.").with_retries(2);

    let inputs = vec![
        "Ada Lovelace discussed analytical engines and early programming.",
        "I love my dog, but I hate rainy weather.",
        "I'm going to the store to buy milk and bread.",
    ];
    let responses: Vec<String> = futures::stream::iter(inputs)
        .map(|text| {
            let names_provider = &names_provider;
            let topics_provider = &topics_provider;
            let sentiment_provider = &sentiment_provider;
            let names_options = &names_options;
            let topics_options = &topics_options;
            let sentiment_options = &sentiment_options;
            async move {
                let (names, topics, sentiment) = futures::try_join!(
                    classic_extract_value::<Names>(
                        names_provider.clone(),
                        text,
                        names_options.clone()
                    ),
                    classic_extract_value::<Topics>(
                        topics_provider.clone(),
                        text,
                        topics_options.clone()
                    ),
                    classic_extract_value::<Sentiment>(
                        sentiment_provider.clone(),
                        text,
                        sentiment_options.clone(),
                    ),
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
