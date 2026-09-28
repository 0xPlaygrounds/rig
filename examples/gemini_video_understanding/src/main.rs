//! Demonstrates Gemini video understanding with typed request settings.
//! Requires `GEMINI_API_KEY`.
//! Run it to see a single prompt combine text instructions with a video URL input.

use anyhow::Result;
use rig::message::{MediaDetail, Message, UserContent, Video};
use rig::prelude::*;
use rig::providers::gemini::{self, Gemini, api};

const VIDEO_URL: &str = "https://www.youtube.com/watch?v=emtHJIxLwEc";

fn build_video_prompt() -> Message {
    Message::User {
        content: rig::NonEmpty::with_rest(
            UserContent::text("Summarize the video."),
            [UserContent::Video(Video {
                data: rig::message::DocumentSourceKind::Url(VIDEO_URL.to_string()),
                // Low resolution is enough to follow the action.
                detail: Some(MediaDetail::Low),
                ..Default::default()
            })],
        ),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Gemini::from_env()?;
    let model = client
        .completion(gemini::GEMINI_3_8_FLASH)
        .settings(api::RequestSettings {
            generation_config: api::GenerationSettings {
                thinking_config: Some(api::ThinkingConfig {
                    thinking_level: Some(api::ThinkingLevel::Low),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        });
    let agent = AgentBuilder::new(model)
        .preamble("Be concise. Answer directly and clearly.")
        .build();

    println!("Sending a video-understanding request to Gemini...");
    let response = agent.prompt(build_video_prompt()).await?.output;
    println!("Summary:\n{response}");

    Ok(())
}
