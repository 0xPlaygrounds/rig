//! OpenRouter audio generation (TTS) smoke test.

use rig::audio_generation::AudioGenerationRequest;
use rig::http_runtime::HttpRuntime;
use rig::providers::openrouter;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn audio_generation_smoke() {
    let cfg = openrouter::functions::Config::from_env(openrouter::GPT_4O_MINI_TTS)
        .expect("config should build");
    let response = openrouter::functions::generate_audio(
        &cfg,
        &HttpRuntime::new(),
        AudioGenerationRequest::new(AUDIO_TEXT, "alloy"),
    )
    .await
    .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
