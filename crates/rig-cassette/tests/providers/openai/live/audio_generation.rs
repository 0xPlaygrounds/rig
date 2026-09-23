//! OpenAI audio generation smoke test.

use rig::audio_generation::AudioGenerationModel;

use rig::prelude::*;
use rig::providers::openai::{self, wire::OpenAI};

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn audio_generation_smoke() {
    let client = OpenAI::from_env()
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let model = client.audio_generation(openai::TTS_1);

    let response = model
        .audio_generation_request(AUDIO_TEXT, "alloy")
        .send()
        .await
        .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
