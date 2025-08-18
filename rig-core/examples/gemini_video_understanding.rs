use rig::message::{ContentFormat, Message, UserContent, Video};
use rig::prelude::*;
use rig::providers::gemini::completion::gemini_api_types::AdditionalParameters;
use rig::{
    OneOrMany,
    completion::Prompt,
    providers::gemini::{self, completion::gemini_api_types::GenerationConfig},
};
use serde_json::json;

#[tracing::instrument(ret)]
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Initialize the Google Gemini client
    let client = gemini::Client::from_env();

    let gen_cfg = GenerationConfig {
        top_k: Some(1),
        top_p: Some(0.95),
        candidate_count: Some(1),
        ..Default::default()
    };
    let cfg = AdditionalParameters::default().with_config(gen_cfg);

    // Create agent with a single context prompt
    let agent = client
        .agent("gemini-2.5-pro")
        .preamble("Be creative and concise. Answer directly and clearly.")
        .temperature(0.5)
        // The `AdditionalParameters` utility struct helps construct a typesafe `additional_params`
        .additional_params(serde_json::to_value(cfg)?) // Unwrap the Result to get the Value
        .build();
    tracing::info!("Prompting the agent...");

    // Prompt the agent and print the response
    let response = agent
        .prompt(Message::User {
            content: OneOrMany::many(vec![
                UserContent::text("Summarize the video."),
                UserContent::Video(Video {
                    data: "https://www.youtube.com/watch?v=emtHJIxLwEc".to_string(),
                    format: Some(ContentFormat::String),
                    media_type: None,
                    additional_params: Some(json!({
                        "video_metadata": {
                            "fps": 0.2
                        }
                    })),
                }),
            ])?,
        })
        .await;

    tracing::info!("Response: {:?}", response);

    match response {
        Ok(response) => println!("{response}"),
        Err(e) => {
            tracing::error!("Error: {:?}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
