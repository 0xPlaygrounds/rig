//! Hyperbolic audio generation smoke test.

use rig::audio_generation::AudioGenerationRequest;
use rig::http_runtime::HttpRuntime;
use rig::providers::hyperbolic;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn audio_generation_smoke() {
    // Hyperbolic's TTS route keys on a language rather than a model id.
    let cfg = hyperbolic::functions::Config::from_env("EN").expect("config should build");
    let rt = HttpRuntime::new();

    let response = hyperbolic::functions::generate_audio(
        &cfg,
        &rt,
        AudioGenerationRequest::new(AUDIO_TEXT, "EN-US"),
    )
    .await
    .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
