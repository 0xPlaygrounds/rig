use std::env;

use rig::pipeline::agent_ops::extract;
use rig::{
    parallel,
    pipeline::{self, passthrough, Op},
    providers::openai::Client,
};
use schemars::JsonSchema;

#[derive(serde::Deserialize, JsonSchema, serde::Serialize)]
struct DocumentScore {
    /// The score of the document
    score: f32,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let manipulation_agent = openai_client
        .extractor::<DocumentScore>("gpt-4")
        .preamble(
            "
            Your role is to score a user's statement on how manipulative it sounds between 0 and 1.
        ",
        )
        .build();

    let depression_agent = openai_client
        .extractor::<DocumentScore>("gpt-4")
        .preamble(
            "
            Your role is to score a user's statement on how depressive it sounds between 0 and 1.
        ",
        )
        .build();

    let intelligent_agent = openai_client
        .extractor::<DocumentScore>("gpt-4")
        .preamble(
            "
            Your role is to score a user's statement on how intelligent it sounds between 0 and 1.
        ",
        )
        .build();

    let chain = pipeline::new()
        .chain(parallel!(
            passthrough(),
            extract(manipulation_agent),
            extract(depression_agent),
            extract(intelligent_agent)
        ))
        .map(|(statement, manip_score, dep_score, int_score)| {
            format!(
                "
                Original statement: {statement}
                Manipulation sentiment score: {}
                Depression sentiment score: {}
                Intelligence sentiment score: {}
                ",
                manip_score.unwrap().score,
                dep_score.unwrap().score,
                int_score.unwrap().score
            )
        });

    // Prompt the agent and print the response
    let response = chain
        .call("I hate swimming. The water always gets in my eyes.")
        .await;

    println!("Pipeline run: {response:?}");

    Ok(())
}
