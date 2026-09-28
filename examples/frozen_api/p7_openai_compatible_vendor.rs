use rig::providers::openai::OpenAIConfig;
use rig::providers::openai::wire::{Dialect, Quirks};

/// Acme's gateway speaks OpenAI Chat Completions, but rejects
/// `stream_options` and has no structured output. Without the usage chunk,
/// streamed usage reports `None`, not zero.
pub const ACME: Dialect = Dialect::gateway("acme", "https://api.acme.test/v1", "ACME_API_KEY")
    .with_quirks(
        Quirks::openai()
            .without_stream_usage()
            .without_response_format(),
    );

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let acme = OpenAIConfig::from_env_with(&ACME)?.client();
    let model = acme.completion("acme-large");

    let response = model.call("Say hello.").await?;
    println!(
        "{} ({:?} output tokens)",
        response.text(),
        response.usage.output_tokens
    );
    Ok(())
}
