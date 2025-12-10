use rig::prelude::*;
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

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = Client::from_env();

    // Note that you can also create your own semantic router for this
    // that uses a vector store under the hood
    let classify_agent = openai_client.extractor::<Specification>(openai::GPT_4)
        .preamble("
            Analyze the given task and break it down into 2-3 distinct approaches.

            Provide an Analysis:
            Explain your understanding of the task and which variations would be valuable.
            Focus on how each approach serves different aspects of the task.

            Along with the analysis, provide 2-3 approaches to tackle the task, each with a brief description:

            Formal style: Write technically and precisely, focusing on detailed specifications
            Conversational style: Write in a friendly and engaging way that connects with the reader
            Hybrid style: Tell a story that includes technical details, combining emotional elements with specifications

            Return only JSON output.
            ")
        .build();

    let specification = classify_agent.extract("
        Write a product description for a new eco-friendly water bottle.
        The target_audience is environmentally conscious millennials and key product features are: plastic-free, insulated, lifetime warranty
        ").await.unwrap();

    let content_agent = openai_client
        .extractor::<TaskResults>(openai::GPT_4)
        .preamble(
            "
                Generate content based on the original task, style, and guidelines.

                Return only your response and the style you used as a JSON object.
                ",
        )
        .build();

    let mut vec: Vec<TaskResults> = Vec::new();
    for task in specification.tasks {
        let results = content_agent
            .extract(&format!(
                "
            Task: {},
            Style: {},
            Guidelines: {}
            ",
                task.original_task, task.style, task.guidelines
            ))
            .await
            .unwrap();
        vec.push(results);
    }

    let judge_agent = openai_client
        .extractor::<Specification>(openai::GPT_4)
        .preamble(
            "
            Analyze the given written materials and decide the best one, giving your reasoning.

            Return the style as well as the corresponding material you have chosen as a JSON object.
            ",
        )
        .build();

    let task_results_raw_json = serde_json::to_string_pretty(&vec).unwrap();
    let results = judge_agent.extract(&task_results_raw_json).await.unwrap();

    println!("Results: {results:?}");

    Ok(())
}
