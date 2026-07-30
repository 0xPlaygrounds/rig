//! Orchestrator/worker/judge, all three built on structured extraction.
//!
//! `client.extractor::<T>(model).preamble(..).build()` is gone: each stage is
//! one [`extract_with_options`] call over plain data — an [`AgentConfig`], the
//! client's [`ProviderConfig`], a shared [`Runtime`] — with the stage's
//! instructions appended to the classic extraction preamble.
use std::sync::Arc;

use rig::agent::AgentConfig;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::prelude::*;
use rig::provider::{ProviderConfig, Runtime};
use rig::providers::openai;
use rig::providers::openai::client::Client;
use schemars::JsonSchema;

#[derive(serde::Deserialize, JsonSchema, serde::Serialize, Debug)]
struct Specification {
    tasks: Vec<Task>,
}

#[derive(serde::Deserialize, JsonSchema, serde::Serialize, Debug)]
struct Task {
    original_task: String,
    style: String,
    guidelines: String,
}

#[derive(serde::Deserialize, JsonSchema, serde::Serialize, Debug)]
struct TaskResults {
    style: String,
    response: String,
}

/// One extraction stage: the classic extractor protocol with `role` appended to
/// its preamble (what the deleted `ExtractorBuilder::preamble` did).
async fn extract_stage<T>(
    provider: &ProviderConfig,
    rt: &Arc<Runtime>,
    role: &str,
    text: &str,
) -> Result<T, anyhow::Error>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    let classic = ExtractOptions::classic_extractor();
    let preamble = classic.preamble.clone().unwrap_or_default();
    let options = classic.with_preamble(format!(
        "{preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{role}"
    ));
    let outcome = extract_with_options::<T>(
        AgentConfig::new(),
        provider.clone(),
        rt.clone(),
        text,
        options,
    )
    .await?;
    Ok(outcome.value)
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = Client::from_env()?;
    let provider = openai_client.provider_config(openai::GPT_4);
    let rt = Arc::new(Runtime::new());

    // Note that you can also create your own semantic router for this
    // that uses a vector store under the hood
    let specification: Specification = extract_stage(
        &provider,
        &rt,
        "
            Analyze the given task and break it down into 2-3 distinct approaches.

            Provide an Analysis:
            Explain your understanding of the task and which variations would be valuable.
            Focus on how each approach serves different aspects of the task.

            Along with the analysis, provide 2-3 approaches to tackle the task, each with a brief description:

            Formal style: Write technically and precisely, focusing on detailed specifications
            Conversational style: Write in a friendly and engaging way that connects with the reader
            Hybrid style: Tell a story that includes technical details, combining emotional elements with specifications

            Return only JSON output.
            ",
        "
        Write a product description for a new eco-friendly water bottle.
        The target_audience is environmentally conscious millennials and key product features are: plastic-free, insulated, lifetime warranty
        ",
    )
    .await?;

    const CONTENT_ROLE: &str = "
                Generate content based on the original task, style, and guidelines.

                Return only your response and the style you used as a JSON object.
                ";

    let mut vec: Vec<TaskResults> = Vec::new();
    for task in specification.tasks {
        let results: TaskResults = extract_stage(
            &provider,
            &rt,
            CONTENT_ROLE,
            &format!(
                "
            Task: {},
            Style: {},
            Guidelines: {}
            ",
                task.original_task, task.style, task.guidelines
            ),
        )
        .await?;
        vec.push(results);
    }

    let task_results_raw_json = serde_json::to_string_pretty(&vec)?;
    let results: Specification = extract_stage(
        &provider,
        &rt,
        "
            Analyze the given written materials and decide the best one, giving your reasoning.

            Return the style as well as the corresponding material you have chosen as a JSON object.
            ",
        &task_results_raw_json,
    )
    .await?;

    println!("Results: {results:?}");

    Ok(())
}
