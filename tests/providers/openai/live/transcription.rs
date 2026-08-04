//! Migrated from `examples/transcription.rs`.

use rig::providers::openai;
use rig::transcription::TranscriptionRequest;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn transcription_smoke() {
    let cfg = openai::functions::Config::from_env(openai::WHISPER_1).expect("config should build");
    let rt = rig::http_runtime::HttpRuntime::new();
    let response = openai::functions::transcribe(
        &cfg,
        &rt,
        TranscriptionRequest::from_file(AUDIO_FIXTURE_PATH)
            .expect("should be able to load audio fixture"),
    )
    .await
    .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
