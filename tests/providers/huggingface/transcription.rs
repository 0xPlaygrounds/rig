//! Migrated from `examples/transcription.rs`.

use rig::http_runtime::HttpRuntime;
use rig::providers::huggingface;
use rig::transcription::TranscriptionRequest;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn transcription_smoke() {
    let cfg =
        huggingface::functions::Config::from_env("whisper-large-v3").expect("config should build");
    let rt = HttpRuntime::new();
    let response = huggingface::functions::transcribe(
        &cfg,
        &rt,
        TranscriptionRequest::from_file(AUDIO_FIXTURE_PATH)
            .expect("should be able to load audio fixture"),
    )
    .await
    .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
