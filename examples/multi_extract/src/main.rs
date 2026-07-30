//! Demonstrates fan-out structured extraction with `futures::try_join!`.
//! Requires `OPENAI_API_KEY`.
//! Run it to see one batch of text split into names, topics, and sentiment in parallel.

use std::sync::Arc;

use anyhow::Result;
use futures::stream::{StreamExt, TryStreamExt};
use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::provider::{ProviderConfig, Runtime};
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

fn sample_inputs() -> Vec<&'static str> {
    vec![
        "Screw you Putin!",
        "I love my dog, but I hate my cat.",
        "I'm going to the store to buy some milk.",
    ]
}

#[tokio::main]
async fn main() -> Result<()> {
    // `client.extractor::<T>(model)` is gone: an extraction is a call to
    // `extract_with_options`, so the per-extractor "configuration" is just the
    // data it will be handed — the preamble on an `AgentConfig`, the retry
    // budget on `ExtractOptions`.
    let provider =
        ProviderConfig::OpenAi(openai::functions::Config::from_env(openai::GPT_4O_MINI)?);
    let rt = Arc::new(Runtime::new());

    let names_config = AgentConfig::new();
    let topics_config = AgentConfig::new();
    let sentiment_config = AgentConfig::new();

    // `.preamble(extra)` on the old builder appended to the extractor preamble;
    // spelling that out keeps the extraction protocol intact.
    let names_options = extractor_options("Extract names from the given text.");
    let topics_options = extractor_options("Extract topics from the given text.");
    let sentiment_options =
        extractor_options("Extract sentiment and confidence from the given text.");

    // Fan each input out to the three extractors concurrently (`try_join!`),
    // running up to four inputs at a time (`buffered`) — the same shape the
    // old `try_parallel!` + `try_batch_call(4, ..)` pipeline provided.
    let responses: Vec<String> = futures::stream::iter(sample_inputs())
        .map(|text| {
            let provider = &provider;
            let rt = &rt;
            let names_config = &names_config;
            let topics_config = &topics_config;
            let sentiment_config = &sentiment_config;
            let names_options = &names_options;
            let topics_options = &topics_options;
            let sentiment_options = &sentiment_options;
            async move {
                let (names, topics, sentiment) = futures::try_join!(
                    extract::<Names>(names_config, provider, rt, text, names_options),
                    extract::<Topics>(topics_config, provider, rt, text, topics_options),
                    extract::<Sentiment>(sentiment_config, provider, rt, text, sentiment_options),
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

    for (idx, response) in responses.iter().enumerate() {
        println!("batch item {}:\n{response}\n", idx + 1);
    }

    Ok(())
}

/// The classic extractor's exchange, with `retries(2)` and its extra
/// instructions appended to the extraction preamble the way the deleted
/// `ExtractorBuilder::preamble` did.
fn extractor_options(extra_instructions: &str) -> ExtractOptions {
    let classic = ExtractOptions::classic_extractor();
    let preamble = classic.preamble.clone().unwrap_or_default();
    classic.with_retries(2).with_preamble(format!(
        "{preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{extra_instructions}"
    ))
}

/// One extraction, borrowing the shared plain-data configuration.
async fn extract<T>(
    config: &AgentConfig,
    provider: &ProviderConfig,
    rt: &Arc<Runtime>,
    text: &str,
    options: &ExtractOptions,
) -> Result<T>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    let outcome = extract_with_options::<T>(
        config.clone(),
        provider.clone(),
        rt.clone(),
        text,
        options.clone(),
    )
    .await?;
    Ok(outcome.value)
}
