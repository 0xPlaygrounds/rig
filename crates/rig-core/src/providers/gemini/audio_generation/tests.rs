use super::*;
use crate::audio_generation::AudioGenerationRequestBuilder;

fn wire() -> Speech {
    Speech::new(
        crate::providers::gemini::GeminiConfig::new("test-key"),
        GEMINI_3_8_FLASH_LITE_TTS,
    )
}

fn body(request: AudioGenerationRequest) -> Value {
    let encoded = wire()
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    assert_eq!(encoded.request.uri().path(), "/v1beta/interactions");
    assert_eq!(
        encoded.request.headers()[super::super::GeminiConfig::INTERACTIONS_KEY_HEADER],
        "test-key"
    );
    crate::test_utils::json_body(&encoded.request)
}

/// A plain request is one turn in its voice; with no voice, the API picks
/// one.
#[test]
fn a_request_speaks_its_text_in_its_voice() {
    assert_eq!(
        body(AudioGenerationRequestBuilder::new("Have a wonderful day!", "Kore").build()),
        json!({
            "model": "gemini-3.8-flash-lite-tts",
            "input": [{"type": "user_input", "content": [
                {"type": "text", "text": "Have a wonderful day!"}
            ]}],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": "Kore"}]}
        })
    );
    let unvoiced = body(AudioGenerationRequestBuilder::new("Hi", "").build());
    assert!(unvoiced.get("generation_config").is_none(), "{unvoiced}");
}

/// A two-speaker script sends each turn with its speaker and style, the
/// speakers as an object, and the format it asks for, as the speech
/// generation guide spells them.
#[test]
fn a_script_sends_its_turns_speakers_and_format() {
    let request = Script::speakers([("Joe", "Puck"), ("Jane", "Kore")])
        .turn(
            Turn::new("How's it going today Jane?")
                .speaker("Joe")
                .style("cheerful and friendly"),
        )
        .turn(Turn::new("Not too bad.").speaker("Jane"))
        .format(AudioFormat::pcm(24_000))
        .build();
    assert_eq!(request.text, "How's it going today Jane?\nNot too bad.");
    assert_eq!(
        body(request),
        json!({
            "model": "gemini-3.8-flash-lite-tts",
            "input": [{"type": "user_input", "content": [
                {"type": "text", "text": "How's it going today Jane?", "annotations": [
                    {"type": "speech_metadata", "speaker": "Joe", "style": "cheerful and friendly"}
                ]},
                {"type": "text", "text": "Not too bad.", "annotations": [
                    {"type": "speech_metadata", "speaker": "Jane"}
                ]}
            ]}],
            "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
            "generation_config": {"speech_config": {"speakers": [
                {"voice": "Puck", "speaker": "Joe"},
                {"voice": "Kore", "speaker": "Jane"}
            ]}}
        })
    );

    let styled = Script::voice("Kore")
        .turn(Turn::new("Wait... <short pause> did you hear that?").style("whispered"))
        .format(AudioFormat::new(AudioEncoding::Mulaw))
        .build();
    assert_eq!(styled.voice, "Kore");
    let styled = body(styled);
    assert_eq!(
        styled["generation_config"],
        json!({"speech_config": [{"voice": "Kore"}]})
    );
    assert_eq!(
        styled["response_format"],
        json!({"type": "audio", "mime_type": "audio/mulaw"})
    );
}

/// The API has no speed, so a request that sets one is refused.
#[test]
fn a_speed_is_refused() {
    let request = AudioGenerationRequestBuilder::new("Hi", "Kore")
        .speed(1.5)
        .build();
    assert!(wire().encode(request, Mode::Unary).is_err());
}

/// A reply from `gemini-3.8-flash-lite-tts` asked for `audio/l16`, with its
/// base64 cut to its first group.
const RECORDED_SPEECH_REPLY: &str = r#"{"id":"v1_speech","status":"completed","usage":{"total_tokens":39,"total_input_tokens":4,"total_cached_tokens":0,"total_output_tokens":35,"total_tool_use_tokens":0,"total_thought_tokens":0},"steps":[{"content":[{"data":"AAECAw==","channels":1,"sample_rate":24000,"mime_type":"audio/l16; rate=24000; channels=1","type":"audio"}],"type":"model_output"}],"object":"interaction","model":"gemini-3.8-flash-lite-tts"}"#;

#[test]
fn the_wire_decodes_a_speech_reply_and_its_format() {
    let reply: Value = serde_json::from_str(RECORDED_SPEECH_REPLY).expect("the reply is JSON");
    let response = crate::test_utils::decode_reply(
        &wire(),
        &AudioGenerationRequestBuilder::new("Hi there!", "Kore").build(),
        Mode::Unary,
        [WireFrame::Text(RECORDED_SPEECH_REPLY.to_owned())],
        reply,
    )
    .expect("one whole reply decodes to one response");
    assert_eq!(response.audio, [0, 1, 2, 3]);
    assert_eq!(response.model.as_deref(), Some(GEMINI_3_8_FLASH_LITE_TTS));
    assert_eq!(response.response_id.as_deref(), Some("v1_speech"));
    assert_eq!(response.usage.input_tokens, Some(4));
    assert_eq!(response.usage.output_tokens, Some(35));
    assert_eq!(
        format_of(&response),
        Some(SpeechFormat {
            mime_type: Some("audio/l16; rate=24000; channels=1".to_owned()),
            sample_rate: Some(24_000),
            channels: Some(1),
        })
    );
}

/// An interaction that ended without audio is a response error naming its
/// status.
#[test]
fn a_reply_without_audio_is_an_error() {
    let reply = json!({"id": "v1_x", "status": "failed", "steps": []}).to_string();
    let err = crate::test_utils::decode_reply(
        &wire(),
        &AudioGenerationRequestBuilder::new("Hi", "Kore").build(),
        Mode::Unary,
        [WireFrame::Text(reply)],
        Value::Null,
    )
    .expect_err("no audio");
    assert!(
        matches!(&err, ProviderError::Response(message) if message.contains("failed")),
        "{err:?}"
    );
}
