use rig::{completion::Prompt, providers::openai};
use base64::{engine::general_purpose::STANDARD, Engine};
use reqwest;
use std::error::Error;

pub async fn download_image_as_base64(image_url: &str) -> Result<String, Box<dyn Error>> {
    let response = reqwest::get(image_url).await?;
    let image_data = response.bytes().await?;
    let base64_string = STANDARD.encode(&image_data);
    let data_uri = format!("data:{};base64,{}", "image/jpeg", base64_string);
    Ok(data_uri)
}

#[tokio::main]
async fn main() {
    // Create OpenAI client and model
    let openai_client = openai::Client::from_env();
    let image_base64 = download_image_as_base64("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg").await.expect("Failed to convert image to base64");
    let gpt4o = openai_client
        .agent("gpt-4o")
        .preamble("You are a helpful assistant.")
        .image_urls(vec![image_base64])
        .build();

    // Prompt the model and print its response
    let response = gpt4o
        .prompt("What is in this image?")
        .await
        .expect("Failed to prompt GPT-4o");

    println!("GPT-4o: {response}");
}
