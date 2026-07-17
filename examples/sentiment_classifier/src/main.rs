//! Demonstrates the smallest typed extractor for classification.
//! Requires `OPENAI_API_KEY`.
//! Run it to map a short sentence into a structured sentiment enum.

use anyhow::Result;
use rig::prelude::*;
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
    let extractor = openai::Client::from_env()?
        .extractor::<DocumentSentiment>(openai::GPT_4)
        .build();

    let sentiment = extractor.extract("I am happy").await?;

    println!("GPT-4: {sentiment:?}");

    Ok(())
}
