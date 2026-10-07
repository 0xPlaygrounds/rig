//! Demonstrates Gemini video understanding with provider-specific request parameters.
//! Requires `GEMINI_API_KEY`.
//! Run it to see a single prompt combine text instructions with a video URL input.

use anyhow::Result;
use rig::message::{Message, UserContent, Video};
use rig::prelude::*;
use rig::providers::gemini::extension::{CandidateCount, GeminiOptions};
use rig::providers::gemini::{self, Gemini};
use serde_json::json;

const MODEL: &str = gemini::completion::GEMINI_2_5_PRO_EXP_03_25;
const VIDEO_URL: &str = "https://www.youtube.com/watch?v=emtHJIxLwEc";

fn build_video_prompt() -> Result<Message> {
    Ok(Message::User {
        content: vec![
            UserContent::text("Summarize the video."),
            UserContent::Video(Video {
                data: rig::message::DocumentSourceKind::Url(VIDEO_URL.to_string()),
                media_type: None,
                additional_params: Some(json!({ "video_metadata": { "fps": 0.2 } })),
            }),
        ],
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Gemini::from_env()?;
    let agent = AgentBuilder::new(client.completion(MODEL))
        .preamble("Be creative and concise. Answer directly and clearly.")
        .temperature(0.5)
        .top_p(0.95)
        .provider_option(
            GeminiOptions::new()
                .candidate_count(CandidateCount::One)
                .top_k(1),
        )
        .build();

    println!("Sending a video-understanding request to Gemini...");
    let response = agent.prompt(build_video_prompt()?).await?.output();
    println!("Summary:\n{response}");

    Ok(())
}
