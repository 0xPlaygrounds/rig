use rig::{completion::Prompt, providers::gemini};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the Google Gemini client
    // Create OpenAI client
    let client = gemini::Client::from_env();

    // Create agent with a single context prompt
    let agent = client
        .agent(gemini::completion::GEMINI_1_5_PRO)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .max_tokens(8192)
        .build();

    // Prompt the agent and print the response
    let response = agent
        .prompt("When and where and what type is the next solar eclipse?")
        .await?;
    println!("{}", response);

    Ok(())

}