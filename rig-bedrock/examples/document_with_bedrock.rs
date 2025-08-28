use rig::{
    completion::{Prompt, message::Document},
    message::{ContentFormat, DocumentMediaType},
};

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::client::{CompletionClient, ProviderClient};
use rig_bedrock::client::Client;
use rig_bedrock::completion::AMAZON_NOVA_LITE;
use tracing::info;

const DOCUMENT_URL: &str = "https://bitcoin.org/bitcoin.pdf";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .without_time()
        .with_level(false)
        .with_target(false)
        .init();

    let client = Client::from_env();
    let agent = client
        .agent(AMAZON_NOVA_LITE)
        .preamble("Describe this document")
        .temperature(0.5)
        .build();

    let reqwest_client = reqwest::Client::new();
    let response = reqwest_client.get(DOCUMENT_URL).send().await?;

    info!("Status: {}", response.status().as_str());
    info!("Content Type: {:?}", response.headers().get("Content-Type"));

    let document_bytes = response.bytes().await?;
    let bytes_base64 = BASE64_STANDARD.encode(document_bytes);

    let document = Document {
        data: bytes_base64,
        format: Some(ContentFormat::Base64),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    };

    let response = agent.prompt(document).await?;
    info!("{}", response);

    Ok(())
}
