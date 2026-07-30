//! Llamafile live coverage for batch multi-extract pipelines.

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::provider::{ProviderConfig, Runtime};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::support::assert_nonempty_response;

use super::support;

/// The classic `ExtractorBuilder::preamble(extra)` appended the extra
/// instructions to the pinned extraction preamble; reproduce that byte for
/// byte on top of `classic_extractor()`.
fn classic_options_with_preamble(extra: &str) -> ExtractOptions {
    let options = ExtractOptions::classic_extractor().with_retries(2);
    let base = options
        .preamble
        .clone()
        .expect("classic_extractor() pins a preamble");
    // `append_preamble` joined with a newline and the appended block itself
    // opened with one, so the separator is two newlines.
    options.with_preamble(format!(
        "{base}\n\n=============== ADDITIONAL INSTRUCTIONS ===============\n{extra}"
    ))
}

/// One classic-extractor exchange through the free-function surface.
async fn classic_extract<T>(
    provider: ProviderConfig,
    text: &str,
    options: ExtractOptions,
) -> Result<T>
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
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn batch_multi_extract_chain() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let model = support::model_name();
    let provider = support::provider(model);
    let names_options = classic_options_with_preamble("Extract names from the given text.");
    let topics_options = classic_options_with_preamble("Extract topics from the given text.");
    let sentiment_options = classic_options_with_preamble(
        "Extract sentiment and confidence from the given text. \
         Return sentiment normalized to the range [-1.0, 1.0] and confidence normalized to [0.0, 1.0].",
    );

    let inputs = vec![
        "Ada Lovelace discussed analytical engines and early programming with Charles Babbage.",
        "Grace said she hates rainy weather but still walked her dog to the park.",
        "Linus is going to the store to buy milk and bread for dinner.",
    ];
    let responses: Vec<String> = futures::stream::iter(inputs)
        .map(|text| {
            let provider = &provider;
            let names_options = &names_options;
            let topics_options = &topics_options;
            let sentiment_options = &sentiment_options;
            async move {
                let (names, topics, sentiment) = futures::try_join!(
                    classic_extract::<Names>(provider.clone(), text, names_options.clone()),
                    classic_extract::<Topics>(provider.clone(), text, topics_options.clone()),
                    classic_extract::<Sentiment>(provider.clone(), text, sentiment_options.clone()),
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
}
