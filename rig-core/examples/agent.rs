use std::env;

use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client = providers::openai::Client::new(&env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"));

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent("gpt-4o")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // let client = providers::cohere::Client::new(
    //     &env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set"),
    // );
    // let comedian_agent = client
    //     .agent("command-r")
    //     .preamble("You are a comedian here to entertain the user using humour and jokes.")
    //     .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!", vec![]).await?;
    println!("{}", response);

    Ok(())
}
