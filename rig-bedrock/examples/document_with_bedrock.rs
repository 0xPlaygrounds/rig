use reqwest::Client;

use base64::{prelude::BASE64_STANDARD, Engine};
use rig::{
    completion::{Preamble, Prompt},
    providers::anthropic::ClientBuilder as AnthropicClientBuilder,
};
use rig::{
    completion::message::Document,
    message::{ContentFormat, DocumentMediaType},
};
use rig_bedrock::client::ClientBuilder;
use tracing::info;

const DOCUMENT_URL: &str =
    "https://www.inf.ed.ac.uk/teaching/courses/ai2/module4/small_slides/small-agents.pdf";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .without_time()
        .with_level(false)
        .with_target(false)
        .init();

    let client = ClientBuilder::new().build().await;
    let anthropic_client = AnthropicClientBuilder::new("").build();
    let completion_model = anthropic_client.completion_model("claude-3-5-sonnet-20240620-v1:0");
    let agent = client
        .agent(completion_model, "claude-3-5-sonnet-20240620-v1:0")
        .preamble(vec![Preamble::new("Describe this document".to_string())])
        .temperature(0.5)
        .build();

    let reqwest_client = Client::new();
    let response = reqwest_client.get(DOCUMENT_URL).send().await?;

    info!("Status: {}", response.status().as_str());
    info!("Content Type: {:?}", response.headers().get("Content-Type"));

    let document_bytes = response.bytes().await?;
    let bytes_base64 = BASE64_STANDARD.encode(document_bytes);

    let document = Document {
        data: bytes_base64,
        format: Some(ContentFormat::Base64),
        media_type: Some(DocumentMediaType::PDF),
    };

    let response = agent.prompt(document).await?;
    info!("{}", response);

    Ok(())
}
