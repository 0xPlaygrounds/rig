/// This example showcases using multiple clients by using a dynamic ClientBuilder to allow you to generate the client
/// without resorting to workarounds like generating an enum for every single provider you want to use.
/// In this example, we will use both OpenAI and Anthropic - so ensure you have your `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` set when using this example!
/// Note that DynClientBuilder does not only support agents - it supports every kind of client that can currently be used in Rig at the moment.
use rig::{
    client::builder::DynClientBuilder, completion::Prompt, providers::anthropic::CLAUDE_3_7_SONNET,
};

#[tokio::main]
async fn main() {
    let multi_client = DynClientBuilder::new();

    // set up OpenAI client
    let completion_openai = multi_client.agent("openai", "gpt-4o").unwrap();
    let agent_openai = completion_openai
        .preamble("You are a helpful assistant")
        .build();

    // set up Anthropic client
    let completion_anthropic = multi_client.agent("anthropic", CLAUDE_3_7_SONNET).unwrap();
    let agent_anthropic = completion_anthropic
        .preamble("You are a helpful assistant")
        .max_tokens(1024)
        .build();

    println!("Sending prompt: 'Hello world!'");

    let res_openai = agent_openai.prompt("Hello world!").await.unwrap();
    println!("Response from OpenAI (using gpt-4o): {res_openai}");

    let res_anthropic = agent_anthropic.prompt("Hello world!").await.unwrap();
    println!("Response from Anthropic (using Claude 3.7 Sonnet): {res_anthropic}");
}
