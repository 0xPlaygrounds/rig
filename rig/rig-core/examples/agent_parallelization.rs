use rig::prelude::*;

use rig::pipeline::agent_ops::extract;

use rig::providers::openai;
use rig::providers::openai::client::Client;

use rig::{
    parallel,
    pipeline::{self, Op, passthrough},
};

use schemars::JsonSchema;

#[derive(Debug, serde::Deserialize, JsonSchema, serde::Serialize)]
struct DocumentScore {
    /// The score of the document
    score: f32,
}
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = Client::from_env()?;

    let manipulation_agent = openai_client
        .extractor::<DocumentScore>(openai::GPT_4)
        .preamble(
            "
            Your role is to score a user's statement on how manipulative it sounds between 0 and 1.
        ",
        )
        .build();

    let depression_agent = openai_client
        .extractor::<DocumentScore>(openai::GPT_4)
        .preamble(
            "
            Your role is to score a user's statement on how depressive it sounds between 0 and 1.
        ",
        )
        .build();

    let intelligent_agent = openai_client
        .extractor::<DocumentScore>(openai::GPT_4)
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
            match (manip_score, dep_score, int_score) {
                (Ok(manip_score), Ok(dep_score), Ok(int_score)) => format!(
                    "
                    Original statement: {statement}
                    Manipulation sentiment score: {}
                    Depression sentiment score: {}
                    Intelligence sentiment score: {}
                    ",
                    manip_score.score, dep_score.score, int_score.score
                ),
                (manip_score, dep_score, int_score) => format!(
                    "
                    Original statement: {statement}
                    Manipulation sentiment score: {manip_score:?}
                    Depression sentiment score: {dep_score:?}
                    Intelligence sentiment score: {int_score:?}
                    "
                ),
            }
        });

    // Prompt the agent and print the response
    let response = chain
        .call("I hate swimming. The water always gets in my eyes.")
        .await;

    println!("Pipeline run: {response:?}");

    Ok(())
}
