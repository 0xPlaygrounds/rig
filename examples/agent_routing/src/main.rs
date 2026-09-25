//! Demonstrates routing one prompt into different follow-up prompts.
//! Requires `OPENAI_API_KEY`.
//! Run it to see a classifier agent choose which second prompt should run.

use anyhow::{Result, bail};
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI, wire::OpenAiWire};

const INPUT_PROMPT: &str = "Sheep can self-medicate";
const ROUTER_PREAMBLE: &str = "
    Categorize the user's statement as exactly one of: sheep, cow, dog.
    Return only the category.
";

type Gpt4 = Model<OpenAiWire>;

fn build_router_agent(model: Gpt4) -> rig::agent::Agent {
    AgentBuilder::new(model).preamble(ROUTER_PREAMBLE).build()
}

fn build_response_agent(model: Gpt4) -> rig::agent::Agent {
    AgentBuilder::new(model).build()
}

fn follow_up_prompt(category: &str) -> Result<&'static str> {
    match category {
        "cow" => Ok("Tell me a fact about the United States of America."),
        "sheep" => Ok("Calculate 5+5 for me. Return only the number."),
        "dog" => Ok("Write me a poem about cashews."),
        other => bail!("could not process category: {other}"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let gpt4 = OpenAI::from_env()?
        .completion(openai::GPT_4)
        .on(rig::transport());
    let category = build_router_agent(gpt4.clone())
        .prompt(INPUT_PROMPT)
        .await?
        .output;
    let follow_up = follow_up_prompt(category.trim())?;
    let response = build_response_agent(gpt4).prompt(follow_up).await?.output;

    println!("Classifier chose: {}", category.trim());
    println!("Follow-up prompt: {follow_up}");
    println!("Response: {}", response.trim());

    Ok(())
}
