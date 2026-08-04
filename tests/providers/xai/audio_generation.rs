//! xAI audio generation smoke test covering provider-specific additional parameters.

use rig::audio_generation::AudioGenerationRequest;
use rig::http_runtime::HttpRuntime;
use rig::providers::xai;
use serde_json::json;

use crate::support::{AUDIO_TEXT, assert_nonempty_bytes};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn audio_generation_smoke() {
    let cfg = xai::functions::Config::from_env(xai::TTS_1).expect("config should build");
    let rt = HttpRuntime::new();

    let response = xai::functions::generate_audio(
        &cfg,
        &rt,
        AudioGenerationRequest::new(AUDIO_TEXT, "eve").with_additional_params(json!({
            "language": "en",
        })),
    )
    .await
    .expect("audio generation should succeed");

    assert_nonempty_bytes(&response.audio);
}
