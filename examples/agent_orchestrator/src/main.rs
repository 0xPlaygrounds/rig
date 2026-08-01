//! Orchestrator/worker/judge, all three built on structured extraction.
//!
//! Requires `OPENAI_API_KEY`.
use rig::agent::Agent;
use rig::extract::ExtractOptions;
use rig::prelude::*;
use rig::providers::openai;
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
async fn extract_stage<T>(agent: &Agent, role: &str, text: &str) -> Result<T, anyhow::Error>
where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
{
    let classic = ExtractOptions::classic_extractor();
    let preamble = classic.preamble.clone().unwrap_or_default();
    let preamble =
        format!("{preamble}\n=============== ADDITIONAL INSTRUCTIONS ===============\n{role}");
    Ok(agent
        .extractor(text)
        .classic()
        .preamble(preamble)
        .run::<T>()
        .await?)
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env()?;
    let agent = client.agent(openai::GPT_4).build();

    // Note that you can also create your own semantic router for this
    // that uses a vector store under the hood
    let specification: Specification = extract_stage(
        &agent,
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
            &agent,
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
        &agent,
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
