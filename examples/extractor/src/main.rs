//! Demonstrates typed extraction and extraction with usage metadata.
//! Requires `OPENAI_API_KEY`.
//! Run it to compare a plain structured extraction with a usage-aware one.
//!
//! `Extractor<M, T>` and `client.extractor::<T>(model)` are gone. Extraction is
//! now the free function [`rig::extract::extract_with_options`]: it takes plain
//! data — an [`AgentConfig`], the client's [`ProviderConfig`], a shared
//! [`Runtime`] for transport handles — and returns an `ExtractionOutcome<T>`
//! carrying `.value` plus the accumulated `.usage`.
//! [`ExtractOptions::classic_extractor()`] reproduces the deleted extractor's
//! exchange exactly (a `submit` output tool, its preamble, `ToolChoice::Required`).

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
struct Person {
    #[schemars(required)]
    first_name: Option<String>,
    #[schemars(required)]
    last_name: Option<String>,
    #[schemars(required)]
    job: Option<String>,
}

const FIRST_INPUT: &str = "Hello my name is John Doe! I am a software engineer.";
const SECOND_INPUT: &str = "Jane Smith is a data scientist.";

#[tokio::main]
async fn main() -> Result<()> {
    // Plain data, built once and cloned per extraction.
    let provider = ProviderConfig::OpenAi(openai::functions::Config::from_env(openai::GPT_4)?);
    let rt = Arc::new(Runtime::new());

    let outcome = extract_with_options::<Person>(
        AgentConfig::new(),
        provider.clone(),
        rt.clone(),
        FIRST_INPUT,
        ExtractOptions::classic_extractor(),
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&outcome.value)?);

    // The same call already reports usage — there is no separate
    // `extract_with_usage` surface to reach for.
    let outcome = extract_with_options::<Person>(
        AgentConfig::new(),
        provider,
        rt,
        SECOND_INPUT,
        ExtractOptions::classic_extractor(),
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&outcome.value)?);
    println!("total tokens: {}", outcome.usage.total_tokens);

    Ok(())
}
