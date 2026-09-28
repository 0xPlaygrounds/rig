use rig::providers::openai::{self, OpenAI};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = OpenAI::from_env()?.completion(openai::GPT_5_2);
    let response = model.call("In one sentence, what is Rust?").await?;

    println!("{}", response.text());
    // `None` means the provider did not report the counter; it is never a fake zero.
    println!(
        "input tokens: {:?}, output tokens: {:?}",
        response.usage.input_tokens, response.usage.output_tokens
    );
    Ok(())
}
