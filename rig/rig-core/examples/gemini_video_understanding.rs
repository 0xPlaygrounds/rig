//! Demonstrates Gemini video understanding with provider-specific request parameters.
//! Requires `GEMINI_API_KEY`.
//! Run it to see a single prompt combine text instructions with a video URL input.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::message::{Message, UserContent, Video};
use rig::providers::gemini::completion::gemini_api_types::AdditionalParameters;
use rig::{
    OneOrMany,
    completion::Prompt,
    providers::gemini::{self, completion::gemini_api_types::GenerationConfig},
};
use serde_json::json;

const MODEL: &str = gemini::completion::GEMINI_2_5_PRO_EXP_03_25;
const VIDEO_URL: &str = "https://www.youtube.com/watch?v=emtHJIxLwEc";

fn build_video_prompt() -> Result<Message> {
    Ok(Message::User {
        content: OneOrMany::many(vec![
            UserContent::text("Summarize the video."),
            UserContent::Video(Video {
                data: rig::message::DocumentSourceKind::Url(VIDEO_URL.to_string()),
                media_type: None,
                additional_params: Some(json!({
                    "video_metadata": {
                        "fps": 0.2
                    }
                })),
            }),
        ])?,
    })
}

fn build_additional_params() -> Result<serde_json::Value> {
    let generation_config = GenerationConfig {
        top_k: Some(1),
        top_p: Some(0.95),
        candidate_count: Some(1),
        ..Default::default()
    };
    Ok(serde_json::to_value(
        AdditionalParameters::default().with_config(generation_config),
    )?)
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = gemini::Client::from_env();
    let additional_params = build_additional_params()?;
    let agent = client
        .agent(MODEL)
        .preamble("Be creative and concise. Answer directly and clearly.")
        .temperature(0.5)
        .additional_params(additional_params)
        .build();

    println!("Sending a video-understanding request to Gemini...");
    let response = agent.prompt(build_video_prompt()?).await?;
    println!("Summary:\n{response}");

    Ok(())
}
