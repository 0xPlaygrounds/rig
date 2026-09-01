use super::*;
use serde_json::json;

/// The default body every provider inherits unless it overrides it.
struct DefaultAudioExt;

impl Provider for DefaultAudioExt {
    type Builder = crate::providers::openai::OpenAICompletionsExtBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl RawAudioGenerationProvider for DefaultAudioExt {
    const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";
    const PROVIDER_NAME: &'static str = "test";
}

fn body(additional_params: Option<serde_json::Value>) -> serde_json::Value {
    DefaultAudioExt::audio_generation_request_body(
        "tts-1",
        AudioGenerationRequest {
            text: "hello".to_owned(),
            voice: "alloy".to_owned(),
            speed: 1.0,
            additional_params,
        },
    )
    .expect("body should build")
}

#[test]
fn request_body_derives_the_documented_fields() {
    let body = body(None);

    assert_eq!(body["model"], json!("tts-1"));
    assert_eq!(body["input"], json!("hello"));
    assert_eq!(body["voice"], json!("alloy"));
    assert_eq!(body["speed"], json!(1.0));
}

/// The defect this default carried: the field reached no provider that
/// inherited the body, even though the endpoint acts on it.
#[test]
fn request_body_merges_additional_params_last() {
    let body = body(Some(
        json!({ "response_format": "wav", "instructions": "Speak slowly." }),
    ));

    assert_eq!(body["response_format"], json!("wav"));
    assert_eq!(body["instructions"], json!("Speak slowly."));
}

/// Merged last, so a caller can override a derived key.
#[test]
fn request_body_lets_additional_params_override_derived_keys() {
    let body = body(Some(
        json!({ "voice": "nova", "speed": 0.5, "input": "other" }),
    ));

    assert_eq!(body["voice"], json!("nova"));
    assert_eq!(body["speed"], json!(0.5));
    assert_eq!(body["input"], json!("other"));
}

#[test]
fn request_body_ignores_non_object_additional_params() {
    assert_eq!(body(Some(json!("not-an-object"))), body(None));
    assert_eq!(body(Some(json!(null))), body(None));
}
