use serde_json::json;

use super::*;
use crate::providers::gemini::GeminiConfig;

fn wire() -> CountTokens {
    CountTokens {
        generate: GenerateContent::new(GeminiConfig::new("test-key"), "gemini-3.8-flash"),
    }
}

#[test]
fn the_count_wraps_the_generate_content_body() {
    let encoded = wire()
        .encode(
            CompletionRequest::new("How long is this?").preamble("Be brief."),
            Mode::Unary,
        )
        .expect("encodes");
    assert_eq!(
        encoded.request.uri().path(),
        "/v1beta/models/gemini-3.8-flash:countTokens"
    );
    let crate::wire::Body::Bytes(bytes) = encoded.request.body() else {
        panic!("a JSON body");
    };
    let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
    assert_eq!(
        body,
        json!({"generateContentRequest": {
            "model": "models/gemini-3.8-flash",
            "contents": [{"role": "user", "parts": [{"text": "How long is this?"}]}],
            "systemInstruction": {"parts": [{"text": "Be brief."}]}
        }})
    );
}

#[test]
fn a_count_reply_decodes() {
    let count = crate::test_utils::decode_reply(
        &wire(),
        &CompletionRequest::new("hi"),
        Mode::Unary,
        [WireFrame::Text(
            r#"{"totalTokens":12,"cachedContentTokenCount":4,"promptTokensDetails":[{"modality":"TEXT","tokenCount":12}]}"#
                .to_owned(),
        )],
        serde_json::Value::Null,
    )
    .expect("decodes");
    assert_eq!(count.total_tokens, Some(12));
    assert_eq!(count.cached_content_token_count, Some(4));
}
