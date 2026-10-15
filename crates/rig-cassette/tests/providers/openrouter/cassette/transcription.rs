//! Cassette-backed OpenRouter transcription tests.

use rig::providers::openrouter;
use rig::transcription::TranscriptionRequestBuilder;
use serde_json::json;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

use super::super::support::with_openrouter_cassette;

#[tokio::test]
async fn transcription_smoke() {
    with_openrouter_cassette("transcription/transcription_smoke", |client| async move {
        let model = client.transcription(openrouter::WHISPER_1);
        let response = model
            .call(
                TranscriptionRequestBuilder::from_file(AUDIO_FIXTURE_PATH)
                    .expect("should be able to load audio fixture")
                    .build(),
            )
            .await
            .expect("transcription should succeed");

        assert_nonempty_response(&response.text);
    })
    .await;
}

/// A diarizing model asked for verbose output returns timed segments and
/// words labeled with speaker indices, and they reach the normalized response.
#[tokio::test]
async fn verbose_diarized_transcription_has_timed_speakers() {
    with_openrouter_cassette(
        "transcription/verbose_diarized_transcription_has_timed_speakers",
        |client| async move {
            let response = client
                .transcription("microsoft/mai-transcribe-2")
                .call(
                    TranscriptionRequestBuilder::from_file(AUDIO_FIXTURE_PATH)
                        .expect("should be able to load audio fixture")
                        .additional_params(json!({
                            "response_format": "verbose_json",
                            "timestamp_granularities": ["segment", "word"],
                            "provider": {"options": {"azure": {"diarization": {"enabled": true}}}},
                        }))
                        .build(),
                )
                .await
                .expect("transcription should succeed");

            assert_nonempty_response(&response.text);
            assert!(response.duration.is_some_and(|seconds| seconds > 0.0));
            assert!(!response.segments.is_empty(), "{response:?}");
            assert!(!response.words.is_empty(), "{response:?}");
            assert!(
                response
                    .segments
                    .iter()
                    .all(|segment| segment.start <= segment.end && segment.speaker.is_some()),
                "{:?}",
                response.segments
            );
            assert!(
                response.words.iter().all(|word| word.speaker.is_some()),
                "{:?}",
                response.words
            );
        },
    )
    .await;
}
