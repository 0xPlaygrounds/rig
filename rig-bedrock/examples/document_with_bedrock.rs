use reqwest::Client;

use rig::{
    completion::{message::Document, Prompt},
    message::{ContentFormat, DocumentMediaType},
};

use base64::{prelude::BASE64_STANDARD, Engine};
use rig_bedrock::{client::ClientBuilder, completion::AMAZON_NOVA_LITE};
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
    let agent = client
        .agent(AMAZON_NOVA_LITE)
        .preamble("Describe this document but respond with json format only")
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
