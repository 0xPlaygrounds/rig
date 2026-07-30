//! Demonstrates the smallest structured extraction for classification.
//! Requires `OPENAI_API_KEY`.
//! Run it to map a short sentence into a structured sentiment enum.
//!
//! `client.extractor::<T>(model).build().extract(text)` is gone; the whole
//! extractor is one call to [`rig::extract::extract_with_options`] over plain
//! data, with [`ExtractOptions::classic_extractor()`] supplying the classic
//! `submit`-tool protocol.

use std::sync::Arc;

use anyhow::Result;
use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::prelude::*;
use rig::provider::Runtime;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
/// An enum representing the sentiment of a document
enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct DocumentSentiment {
    /// The sentiment of the document
    sentiment: Sentiment,
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;

    let sentiment = extract_with_options::<DocumentSentiment>(
        AgentConfig::new(),
        client.provider_config(openai::GPT_4),
        Arc::new(Runtime::new()),
        "I am happy",
        ExtractOptions::classic_extractor(),
    )
    .await?
    .value;

    println!("GPT-4: {sentiment:?}");

    Ok(())
}
