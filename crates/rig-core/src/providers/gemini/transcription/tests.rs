use super::*;
use crate::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

fn transcription_request() -> transcription::TranscriptionRequest {
    transcription::TranscriptionRequest {
        data: b"audio bytes".to_vec(),
        filename: "audio.mp3".to_string(),
        language: None,
        prompt: None,
        temperature: None,
        additional_params: None,
    }
}

/// The bytes one transcription puts on the wire, pinned by
/// `crates/rig-cassette/fixtures/cassettes/gemini/transcription/transcription_smoke.yaml`: the path,
/// the `key` query parameter and the whole JSON document, down to the `null`
/// fields Gemini's request shape keeps. The cassette's `inlineData.data` is a
/// 32 KB mp3; this request's audio is [`transcription_request`]'s, so only
/// that one string differs.
#[test]
fn the_wire_encodes_the_recorded_generate_content_request() {
    let wire = Transcriptions::new(
        crate::providers::gemini::GeminiConfig::new("test-key"),
        GEMINI_3_FLASH_PREVIEW,
    );

    let encoded = wire
        .encode(transcription_request(), Mode::Unary)
        .expect("the request encodes");

    let request = &encoded.request;
    assert_eq!(request.method().as_str(), "POST");
    assert_eq!(
        request.uri().path(),
        "/v1beta/models/gemini-3-flash-preview:generateContent"
    );
    assert_eq!(request.uri().query(), Some("key=test-key"));
    assert_eq!(
        request
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert_eq!(encoded.request_id_header, None);

    assert_eq!(
        crate::test_utils::json_body(request),
        serde_json::json!({
            "contents": [{
                "parts": [{
                    "inlineData": {
                        "data": "YXVkaW8gYnl0ZXM=",
                        "mimeType": "audio/mpeg"
                    },
                    "thought": false
                }],
                "role": "user"
            }],
            "generationConfig": {},
            "safetySettings": null,
            "systemInstruction": {
                "parts": [{ "text": TRANSCRIPTION_PREAMBLE, "thought": false }],
                "role": "model"
            },
            "toolConfig": null
        })
    );
}

/// `additional_params` become the generation config, so anything but an
/// object (or nothing) is refused before a request is sent.
#[test]
fn additional_params_that_are_not_an_object_are_refused() {
    let request = transcription::TranscriptionRequest {
        additional_params: Some(serde_json::json!([1])),
        ..transcription_request()
    };
    let error = transcription_body(request).expect_err("an array is no config");
    assert!(
        error.to_string().contains("should be an object, got [1]"),
        "{error}"
    );
}

/// This checks preservation of caller input at the request boundary, without
/// relying on a live model to follow the instruction or recording new audio.
#[test]
fn the_wire_preserves_prompt_and_output_language() {
    let wire = Transcriptions::new(
        crate::providers::gemini::GeminiConfig::new("test-key"),
        GEMINI_3_FLASH_PREVIEW,
    );
    let prompt = "Names: \"Rig\", café.\nKeep speaker labels.";
    for (prompt, language) in [
        (Some(prompt), None),
        (None, Some("fr")),
        (Some(prompt), Some("fr")),
    ] {
        let request = transcription::TranscriptionRequest {
            prompt: prompt.map(str::to_owned),
            language: language.map(str::to_owned),
            ..transcription_request()
        };
        let encoded = wire.encode(request, Mode::Unary).expect("request encodes");
        let body = crate::test_utils::json_body(&encoded.request);
        let parts = body
            .pointer("/contents/0/parts")
            .and_then(Value::as_array)
            .expect("user parts");
        assert_eq!(parts.len(), 1 + usize::from(prompt.is_some()));
        assert_eq!(
            parts.first(),
            Some(&json!({
                "inlineData": { "data": "YXVkaW8gYnl0ZXM=", "mimeType": "audio/mpeg" },
                "thought": false,
            }))
        );
        assert_eq!(
            parts.get(1),
            prompt
                .map(|text| json!({ "text": text, "thought": false }))
                .as_ref()
        );
        let instructions = body
            .pointer("/systemInstruction/parts")
            .and_then(Value::as_array)
            .expect("system instruction parts");
        assert_eq!(instructions.len(), 1 + usize::from(language.is_some()));
        assert_eq!(
            instructions.first(),
            Some(&json!({ "text": TRANSCRIPTION_PREAMBLE, "thought": false }))
        );
        assert_eq!(
            instructions.get(1),
            language
                .map(|language| json!({
                    "text": format!("Use {language} as the output language."),
                    "thought": false,
                }))
                .as_ref()
        );
        assert_eq!(body.get("generationConfig"), Some(&json!({})));
    }
}

/// Generation settings remain independent of the text instructions; this
/// request-only precedence rule does not require a provider response.
#[test]
fn transcription_options_preserve_generation_config_and_temperature_precedence() {
    let request = transcription::TranscriptionRequest {
        prompt: Some("Keep speaker labels.".to_owned()),
        language: Some("en".to_owned()),
        temperature: Some(0.2),
        additional_params: Some(json!({ "temperature": 0.8, "maxOutputTokens": 100 })),
        ..transcription_request()
    };
    let body: Value = serde_json::from_slice(&transcription_body(request).expect("options encode"))
        .expect("JSON body");
    assert_eq!(
        body.get("generationConfig"),
        Some(&json!({ "temperature": 0.2, "maxOutputTokens": 100 }))
    );
}
