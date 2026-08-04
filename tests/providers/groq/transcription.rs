//! Migrated from `examples/transcription.rs`.

use rig::http_runtime::HttpRuntime;
use rig::providers::groq;
use rig::transcription::TranscriptionRequest;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn transcription_smoke() {
    let cfg =
        groq::functions::Config::from_env(groq::WHISPER_LARGE_V3).expect("config should build");
    let rt = HttpRuntime::new();
    let response = groq::functions::transcribe(
        &cfg,
        &rt,
        TranscriptionRequest::from_file(AUDIO_FIXTURE_PATH)
            .expect("should be able to load audio fixture"),
    )
    .await
    .expect("transcription should succeed");

    assert_nonempty_response(&response.text);
}
