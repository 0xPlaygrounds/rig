use rig_agent::prelude::*;
use rig_gemini_grpc::GeminiGrpc;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Connect with the API key in `GEMINI_API_KEY`.
    let transport = GeminiGrpc::from_env().await?;

    let agent = AgentBuilder::new(transport.completion("gemini-2.5-flash"))
        .preamble("Be creative and concise. Answer directly and clearly.")
        .temperature(0.5)
        .build();

    let response = agent
        .prompt("How much wood would a woodchuck chuck if a woodchuck could chuck wood? Infer an answer.")
        .await?;
    println!("{}", response.output());

    Ok(())
}
