//! Migrated from `examples/transcription.rs`.

use rig::http_runtime::HttpRuntime;
use rig::providers::azure;
use rig::transcription::TranscriptionRequest;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires AZURE_OPENAI_API_KEY and related Azure env vars"]
async fn transcription_smoke() {
    let cfg = azure::functions::Config::from_env("whisper").expect("config should build");
    let rt = HttpRuntime::new();
    let response = azure::functions::transcribe(
        &cfg,
        &rt,
        TranscriptionRequest::from_file(AUDIO_FIXTURE_PATH)
            .expect("should be able to load audio fixture"),
    )
    .await
    .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
