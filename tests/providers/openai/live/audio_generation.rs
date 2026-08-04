//! OpenAI audio generation smoke test.

use rig::audio_generation::AudioGenerationRequest;
use rig::providers::openai;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn audio_generation_smoke() {
    let cfg = openai::functions::Config::from_env(openai::TTS_1).expect("config should build");
    let rt = rig::http_runtime::HttpRuntime::new();

    let response = openai::functions::generate_audio(
        &cfg,
        &rt,
        AudioGenerationRequest::new(AUDIO_TEXT, "alloy"),
    )
    .await
    .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
