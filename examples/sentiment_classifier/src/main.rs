//! Demonstrates the smallest structured extraction for classification.
//! Requires `OPENAI_API_KEY`.
//! Run it to map a short sentence into a structured sentiment enum.
//!
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
    let client = openai::Client::from_env()?;
    let agent = client.agent(openai::GPT_4).build();
    let sentiment: DocumentSentiment = agent.extractor("I am happy").classic().run().await?;

    println!("GPT-4: {sentiment:?}");

    Ok(())
}
