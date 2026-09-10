use super::*;
use serde_json::json;

/// The default body every provider inherits unless it overrides it.
#[derive(Debug, Clone)]
struct DefaultAudioExt;

impl Provider for DefaultAudioExt {
    const NAME: &'static str = "test";
    const BASE_URL: &'static str = "";
    const VERIFY_PATH: &'static str = "/models";
    type ApiKey = crate::client::Nothing;
    type Config = ();
    type EnvInput = crate::client::Nothing;

    fn build(_: (), _: &crate::client::Nothing) -> crate::http_client::Result<Self> {
        Ok(DefaultAudioExt)
    }
    fn from_env<H: crate::http_client::HttpClientExt>(
        http: H,
    ) -> crate::client::ProviderClientResult<Client<Self, H>> {
        Client::new_with(crate::client::Nothing, http)
    }
    fn from_val<H: crate::http_client::HttpClientExt>(
        _: crate::client::Nothing,
        http: H,
    ) -> crate::client::ProviderClientResult<Client<Self, H>> {
        Client::new_with(crate::client::Nothing, http)
    }
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
